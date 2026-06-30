// Lifecycle owner for the `omlx serve` child process.
//
// State machine
//   stopped ─start()→ starting ─/health 200→ running ─/health fail×3→ unresponsive
//                       │                       │ ↑                       │
//                       │                       │ └─/health or status OK──┘
//                       │                       │
//                       │                       └─process exit → auto-restart
//                       └─process exit during startup → auto-restart
//
//   stop()  : * → stopping → SIGTERM → wait ≤10s → SIGKILL → stopped
//   forceRestart() : * → SIGKILL → start()
//   crashes : auto-restart with 5s/10s/20s backoff, max 3 attempts, counter
//             resets after 60s of stable .running
//
// Spawn invocation:
//   <python> -m omlx.cli serve --base-path <base> --port <port>
//   stdout+stderr → ~/Library/Application Support/oMLX/logs/server.log
//   PATH = parent + Homebrew prefixes
//
// Dev override: OMLX_DEV_SERVER_SCRIPT spawns <python> <script> --port …
// instead, used by Scripts/dev_server.py to exercise the spawn path without
// the full omlx stack.
//
// State changes posted via NotificationCenter so MenubarController and (in
// PR 6) the AppView shell can react without owning the lifecycle.

import Foundation
import Darwin

// @unchecked Sendable: state mutations either happen on the main thread
// (start, stop, force restart, callbacks dispatched via main) or inside
// the @MainActor health-check Task. Process termination handler bounces
// to main before touching state.
final class ServerProcess: @unchecked Sendable {
    enum State: Equatable, Sendable {
        case stopped
        case starting
        case running(pid: Int32)
        case attached(pid: Int32?)
        case stopping
        case unresponsive(pid: Int32)
        case failed(message: String)

        var isRunningLike: Bool {
            switch self {
            case .running, .attached, .unresponsive: return true
            default:                      return false
            }
        }
    }

    enum StartResult: Sendable {
        case started
        case alreadyRunning
        case portConflict(PortConflict)
    }

    enum StartError: Error, CustomStringConvertible {
        case spawnFailed(String)

        var description: String {
            switch self {
            case .spawnFailed(let m): return "Spawn failed: \(m)"
            }
        }
    }

    static let stateDidChangeNotification = Notification.Name("OMLXServerProcessStateDidChange")
    static let portConflictNotification   = Notification.Name("OMLXServerPortConflict")

    // Inputs

    private(set) var bindAddress: String
    /// The connectable host — normalises `0.0.0.0` → `127.0.0.1` because
    /// `0.0.0.0` is a bind wildcard, not a connectable address.
    var host: String {
        AppConfig.connectableHost(for: bindAddress)
    }
    private(set) var port: Int
    private(set) var basePath: URL
    private let runtime: PythonRuntime
    private var resolver: PortConflictResolver

    /// Apply a new host/port/basePath. Only legal while the server is
    /// stopped — won't take effect until the next `start()`. Caller is
    /// responsible for stopping first; we throw if asked to mutate a live
    /// process so a stale resolver / spawn args can never reach a running
    /// uvicorn.
    enum ReconfigureError: Error { case serverIsLive }
    func reconfigure(bindAddress: String? = nil, port: Int? = nil, basePath: URL? = nil) throws {
        switch state {
        case .running, .attached, .starting, .stopping, .unresponsive:
            throw ReconfigureError.serverIsLive
        case .stopped, .failed:
            break
        }
        if let bindAddress { self.bindAddress = bindAddress }
        if let port { self.port = port }
        if let basePath { self.basePath = basePath }
        self.resolver = PortConflictResolver(host: self.host, port: self.port)
    }

    // Tunables (mirror server_manager.py)

    private let healthCheckInterval: TimeInterval = 5
    private let maxHealthFailures = 3
    private let auxiliaryHealthFreshness: TimeInterval = 15
    private let maxAutoRestarts   = 3
    private let stableThreshold: TimeInterval = 60   // seconds before counter resets
    private let stopGraceSeconds: TimeInterval = 10

    // State

    private(set) var state: State = .stopped
    private var process: Process?
    private var logHandle: FileHandle?
    private var healthTask: Task<Void, Never>?
    private var consecutiveFailures = 0
    private var autoRestartCount    = 0
    private var lastHealthyAt: Date?
    private var lastAuxiliaryHealthyAt: Date?
    private var expectingExit       = false   // set by stop()/forceRestart() so terminationHandler doesn't trigger auto-restart
    private var attachedExternally  = false
    private var attachedOwnerPID: pid_t?
    private let logURL: URL

    init(
        runtime: PythonRuntime,
        bindAddress: String = "127.0.0.1",
        port: Int = 8000,
        basePath: URL = ServerProcess.defaultBasePath()
    ) {
        self.runtime  = runtime
        self.bindAddress = bindAddress
        self.port     = port
        self.basePath = basePath
        self.logURL   = ServerProcess.defaultLogURL()
        self.resolver = PortConflictResolver(
            host: AppConfig.connectableHost(for: bindAddress),
            port: port
        )
    }

    // MARK: - Public surface

    var isRunning: Bool {
        if case .running = state { return true }
        if case .attached = state { return true }
        if case .unresponsive = state { return true }
        return process?.isRunning == true
    }

    var pid: Int32? {
        if let processPID = process?.processIdentifier { return processPID }
        return attachedOwnerPID
    }
    var serverLogURL: URL { logURL }

    /// Start the server. Returns .started on success, .alreadyRunning if
    /// already up, or .portConflict if the port is busy. Throws only on
    /// spawn-syscall failure.
    @discardableResult
    func start() throws -> StartResult {
        switch state {
        case .running, .attached, .starting, .unresponsive:
            return .alreadyRunning
        default:
            break
        }

        // Sync probe — fast enough on local connect refused.
        if resolver.isPortInUseSync() {
            let conflict = PortConflict(
                pid: resolver.findOwnerPIDSync(),
                isOMLX: resolver.isOMLXOnPortSync()
            )
            if conflict.isOMLX {
                attachExternal(pid: conflict.pid)
                return .alreadyRunning
            }
            update(.failed(message: "Port \(port) in use" +
                           (conflict.isOMLX ? " (oMLX server already running)" : "")))
            postPortConflict(conflict)
            return .portConflict(conflict)
        }

        try doStart()
        return .started
    }

    /// Graceful stop: SIGTERM → wait ≤ stopGraceSeconds → SIGKILL.
    func stop(timeout: TimeInterval? = nil, stopAttachedExternal: Bool = true) async {
        guard isRunning || state == .starting else { return }

        if isAttachedToExternal {
            guard stopAttachedExternal else {
                cancelHealthLoop()
                attachedExternally = false
                attachedOwnerPID = nil
                update(.stopped)
                return
            }
            update(.stopping)
            cancelHealthLoop()
            let ownerPID = attachedOwnerPID ?? resolver.findOwnerPIDSync()
            let stopped = await stopExternalOwner(pid: ownerPID)
            attachedExternally = false
            attachedOwnerPID = nil
            if stopped {
                update(.stopped)
            } else {
                let conflict = PortConflict(
                    pid: resolver.findOwnerPIDSync(),
                    isOMLX: resolver.isOMLXOnPortSync()
                )
                update(.failed(message: "Could not stop external oMLX on port \(port)"))
                postPortConflict(conflict)
            }
            return
        }

        update(.stopping)
        expectingExit = true
        cancelHealthLoop()

        let timeout = timeout ?? stopGraceSeconds
        if let proc = process, proc.isRunning {
            kill(proc.processIdentifier, SIGTERM)

            let deadline = Date().addingTimeInterval(timeout)
            while proc.isRunning && Date() < deadline {
                try? await Task.sleep(for: .milliseconds(100))
            }
            if proc.isRunning {
                kill(proc.processIdentifier, SIGKILL)
                try? await Task.sleep(for: .seconds(0.5))
            }
        }

        // terminationHandler updates state to .stopped; force in case it
        // didn't fire yet.
        if state != .stopped {
            update(.stopped)
        }
        expectingExit = false
        process = nil
        lastAuxiliaryHealthyAt = nil
        closeLog()
    }

    /// Force-restart: SIGKILL the child without waiting, reset counters,
    /// then start() fresh.
    @discardableResult
    func forceRestart() async throws -> StartResult {
        if isAttachedToExternal {
            update(.stopping)
            cancelHealthLoop()
            let ownerPID = attachedOwnerPID ?? resolver.findOwnerPIDSync()
            let stopped = await stopExternalOwner(pid: ownerPID)
            attachedExternally = false
            attachedOwnerPID = nil
            autoRestartCount = 0
            consecutiveFailures = 0
            update(.stopped)
            guard stopped else {
                let conflict = PortConflict(
                    pid: resolver.findOwnerPIDSync(),
                    isOMLX: resolver.isOMLXOnPortSync()
                )
                update(.failed(message: "Could not restart external oMLX on port \(port)"))
                postPortConflict(conflict)
                return .portConflict(conflict)
            }
            return try start()
        }

        expectingExit = true
        cancelHealthLoop()
        if let proc = process, proc.isRunning {
            kill(proc.processIdentifier, SIGKILL)
            let deadline = Date().addingTimeInterval(2)
            while proc.isRunning && Date() < deadline {
                try? await Task.sleep(for: .milliseconds(50))
            }
        }
        process = nil
        closeLog()
        autoRestartCount = 0
        consecutiveFailures = 0
        lastAuxiliaryHealthyAt = nil
        expectingExit = false
        update(.stopped)
        return try start()
    }

    /// Called by lightweight menubar status polling when the server answers
    /// `/api/status`. Under heavy generation load this keeps the UI from
    /// declaring the managed process unresponsive solely because `/health`
    /// was delayed.
    @MainActor
    func recordAuxiliaryHealthSuccess(at date: Date = Date()) {
        lastAuxiliaryHealthyAt = date
        lastHealthyAt = date
        consecutiveFailures = 0
        switch state {
        case .starting:
            if let pid = process?.processIdentifier {
                update(.running(pid: pid))
            }
        case .unresponsive(let pid):
            update(.running(pid: pid))
        default:
            break
        }
    }

    /// Synchronous SIGTERM-then-SIGKILL of the child, used by signal
    /// handlers (which can't await). Returns when the kernel reports
    /// the PID as gone or after `timeout` seconds.
    func reapSync(timeout: TimeInterval = 5) {
        guard let proc = process, proc.isRunning else { return }
        let pid = proc.processIdentifier
        kill(pid, SIGTERM)
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if kill(pid, 0) != 0 { return }     // process gone
            usleep(100_000)                     // 100 ms
        }
        kill(pid, SIGKILL)
    }

    // MARK: - Internal — spawn

    private func doStart() throws {
        attachedExternally = false
        attachedOwnerPID = nil
        try ensureDir(basePath)
        try ensureDir(logURL.deletingLastPathComponent())
        consecutiveFailures = 0
        lastAuxiliaryHealthyAt = nil

        if !FileManager.default.fileExists(atPath: logURL.path) {
            FileManager.default.createFile(atPath: logURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: logURL)
        try handle.seekToEnd()
        logHandle = handle

        let proc = Process()
        proc.executableURL = runtime.executable
        proc.arguments = makeArguments()
        proc.environment = runtime.makeEnvironment()
        proc.standardOutput = handle
        proc.standardError  = handle
        proc.terminationHandler = { [weak self] term in
            DispatchQueue.main.async {
                self?.handleProcessExit(code: term.terminationStatus)
            }
        }

        update(.starting)
        do {
            try proc.run()
        } catch {
            closeLog()
            update(.failed(message: "spawn failed: \(error.localizedDescription)"))
            throw StartError.spawnFailed(error.localizedDescription)
        }
        process = proc
        startHealthCheckLoop()
    }

    private func handleProcessExit(code: Int32) {
        let wasExpectingExit = expectingExit
        expectingExit = false
        attachedExternally = false
        attachedOwnerPID = nil
        process = nil
        closeLog()

        if wasExpectingExit {
            update(.stopped)
            return
        }

        switch state {
        case .starting:
            tryAutoRestart(reason: "Server exited with code \(code) during startup")
        case .running, .unresponsive:
            tryAutoRestart(reason: "Server exited with code \(code)")
        default:
            // Unexpected — log and stop.
            update(.stopped)
        }
    }

    private func tryAutoRestart(reason: String) {
        // Reset counter if last healthy was > stableThreshold ago.
        if let last = lastHealthyAt,
           Date().timeIntervalSince(last) >= stableThreshold {
            autoRestartCount = 0
        }

        if autoRestartCount >= maxAutoRestarts {
            update(.failed(message: "\(reason). Auto-restart failed after \(maxAutoRestarts) attempts."))
            return
        }

        autoRestartCount += 1
        consecutiveFailures = 0
        lastAuxiliaryHealthyAt = nil
        let attempt = autoRestartCount
        let backoff = TimeInterval(5 * (1 << (attempt - 1)))   // 5, 10, 20s

        NSLog("oMLX: auto-restart \(attempt)/\(maxAutoRestarts) in \(Int(backoff))s — \(reason)")
        update(.starting)

        Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(backoff))
            guard let self else { return }
            // If the user (or stop) intervened during backoff, abort.
            guard case .starting = self.state else { return }

            do {
                try self.doStart()
            } catch {
                self.update(.failed(message: "Auto-restart failed: \(error)"))
            }
        }
    }

    // MARK: - Internal — health check

    private func startHealthCheckLoop() {
        cancelHealthLoop()
        healthTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                await self.tickHealth()
                try? await Task.sleep(for: .seconds(self.healthCheckInterval))
            }
        }
    }

    private func cancelHealthLoop() {
        healthTask?.cancel()
        healthTask = nil
    }

    @MainActor
    private func tickHealth() async {
        switch state {
        case .starting, .running, .attached, .unresponsive:
            break
        default:
            return
        }

        let probe = await resolver.probeHealth()
        let now = Date()
        switch state {
        case .starting:
            if probe.ok || hasRecentAuxiliaryHealth(now: now) {
                let pid = process?.processIdentifier ?? 0
                markHealthy(pid: pid, at: now)
            } else {
                logHealthProbeFailure(probe, failures: consecutiveFailures, suppressed: false)
            }
        case .attached(let pid):
            if probe.ok {
                consecutiveFailures = 0
                lastHealthyAt = now
                attachedOwnerPID = pid ?? attachedOwnerPID
            } else if hasRecentAuxiliaryHealth(now: now) {
                logHealthProbeFailure(probe, failures: consecutiveFailures, suppressed: true)
                consecutiveFailures = 0
                lastHealthyAt = now
                attachedOwnerPID = pid ?? attachedOwnerPID
            } else {
                consecutiveFailures += 1
                logHealthProbeFailure(probe, failures: consecutiveFailures, suppressed: false)
                if consecutiveFailures >= maxHealthFailures {
                    update(.unresponsive(pid: attachedOwnerPID ?? pid ?? 0))
                }
            }
        case .running(let pid), .unresponsive(let pid):
            if probe.ok {
                markHealthy(pid: pid, at: now)
            } else if hasRecentAuxiliaryHealth(now: now) {
                logHealthProbeFailure(probe, failures: consecutiveFailures, suppressed: true)
                markHealthy(pid: pid, at: now)
            } else {
                consecutiveFailures += 1
                logHealthProbeFailure(probe, failures: consecutiveFailures, suppressed: false)
                if consecutiveFailures >= maxHealthFailures,
                   case .running = state {
                    update(.unresponsive(pid: pid))
                }
            }
        default:
            return
        }
    }

    @MainActor
    private func markHealthy(pid: Int32, at date: Date) {
        consecutiveFailures = 0
        lastHealthyAt = date
        switch state {
        case .starting:
            update(.running(pid: pid))
        case .unresponsive:
            if attachedExternally {
                update(.attached(pid: attachedOwnerPID ?? pid))
            } else {
                update(.running(pid: pid))
            }
        default:
            break
        }
    }

    private func hasRecentAuxiliaryHealth(now: Date) -> Bool {
        guard let lastAuxiliaryHealthyAt else { return false }
        return now.timeIntervalSince(lastAuxiliaryHealthyAt) <= auxiliaryHealthFreshness
    }

    private func logHealthProbeFailure(
        _ result: HealthProbeResult,
        failures: Int,
        suppressed: Bool
    ) {
        let status = result.statusCode.map(String.init) ?? "none"
        let error = result.errorDescription ?? "none"
        NSLog(
            "oMLX: health probe failed url=\(result.url) latency_ms=\(result.latencyMs) status=\(status) error=\(error) failures=\(failures) suppressed_by_recent_status=\(suppressed)"
        )
    }

    // MARK: - Internal — helpers

    private func makeArguments() -> [String] {
        let env = ProcessInfo.processInfo.environment
        if let dev = env["OMLX_DEV_SERVER_SCRIPT"], !dev.isEmpty {
            return [dev, "--host", bindAddress, "--port", String(port)]
        }
        return [
            "-m", "omlx.cli", "serve",
            "--base-path", basePath.path,
            "--port", String(port),
        ]
    }

    private func attachExternal(pid: pid_t?) {
        process = nil
        closeLog()
        expectingExit = false
        attachedExternally = true
        attachedOwnerPID = pid
        consecutiveFailures = 0
        autoRestartCount = 0
        lastHealthyAt = Date()
        update(.attached(pid: pid))
        startHealthCheckLoop()
    }

    private var isAttachedToExternal: Bool {
        process == nil && attachedExternally
    }

    private func stopExternalOwner(pid: pid_t?) async -> Bool {
        if let pid, isHomebrewOMLXOwner(pid: pid) {
            if await runBrewServices(action: "stop"),
               await waitUntilPortFree(timeout: 12) {
                return true
            }
        }

        if let pid {
            _ = await resolver.killExternal(pid, timeout: 8)
        }
        return await waitUntilPortFree(timeout: 8)
    }

    private func waitUntilPortFree(timeout: TimeInterval) async -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if !resolver.isPortInUseSync() { return true }
            try? await Task.sleep(for: .milliseconds(200))
        }
        return !resolver.isPortInUseSync()
    }

    private func isHomebrewOMLXOwner(pid: pid_t) -> Bool {
        guard let command = commandLine(for: pid) else { return false }
        return command.contains("/opt/homebrew/")
            && (command.contains("/omlx")
                || command.contains("omlx.cli")
                || command.contains("omlx serve"))
    }

    private func commandLine(for pid: pid_t) -> String? {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/ps")
        proc.arguments = ["-p", String(pid), "-o", "command="]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = Pipe()
        do {
            try proc.run()
        } catch {
            return nil
        }
        proc.waitUntilExit()
        guard proc.terminationStatus == 0 else { return nil }
        let data = (try? pipe.fileHandleForReading.readToEnd()) ?? Data()
        return String(data: data, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func runBrewServices(action: String) async -> Bool {
        let candidates = [
            "/opt/homebrew/bin/brew",
            "/usr/local/bin/brew",
        ]
        guard let brew = candidates.first(where: {
            FileManager.default.isExecutableFile(atPath: $0)
        }) else { return false }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: brew)
        proc.arguments = ["services", action, "jason5545/omlx/omlx"]
        proc.standardOutput = Pipe()
        proc.standardError = Pipe()
        do {
            try proc.run()
        } catch {
            return false
        }

        let deadline = Date().addingTimeInterval(15)
        while proc.isRunning && Date() < deadline {
            try? await Task.sleep(for: .milliseconds(200))
        }
        if proc.isRunning {
            proc.terminate()
            return false
        }
        return proc.terminationStatus == 0
    }

    private func update(_ next: State) {
        guard state != next else { return }
        state = next
        // Observers (MenubarController.serverStateChanged, AppServices) are
        // `@MainActor`. We can be called from the cooperative executor pool
        // (via `await stop()` / `await forceRestart()` / `tickHealth`), so a
        // direct synchronous post trips Swift 6's actor-isolation check and
        // crashes the parent — see crash report 2026-05-09. Hop to main.
        let note = Self.stateDidChangeNotification
        if Thread.isMainThread {
            NotificationCenter.default.post(name: note, object: self)
        } else {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                NotificationCenter.default.post(name: note, object: self)
            }
        }
    }

    private func postPortConflict(_ conflict: PortConflict) {
        let note = Self.portConflictNotification
        if Thread.isMainThread {
            NotificationCenter.default.post(name: note, object: self, userInfo: ["conflict": conflict])
        } else {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                NotificationCenter.default.post(name: note, object: self, userInfo: ["conflict": conflict])
            }
        }
    }

    private func ensureDir(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    private func closeLog() {
        try? logHandle?.close()
        logHandle = nil
    }

    static func defaultBasePath() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".omlx", isDirectory: true)
    }

    static func defaultLogURL() -> URL {
        AppConfig.appSupportURL().appendingPathComponent("logs/server.log")
    }
}

// MARK: - Port conflict payload

struct PortConflict: Sendable, Equatable {
    let pid: pid_t?
    let isOMLX: Bool
}
