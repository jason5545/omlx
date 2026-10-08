#!/usr/bin/env bash
# Mac app 部署：從目前的 checkout 建 release app、用 Jason 的 Apple Development 憑證重簽、
# 換掉 /Applications/oMLX.app，再確認 app 有 attach 到 brew service。步驟與理由見
# AGENTS.md「Mac app 操作」。brew service 和 Mac app 一律同步部署，deploy_fast.sh 部署完
# 會自動接著跑這支。
#
#   scripts/deploy_app.sh
#
# app 內建的 omlx 是 build.sh 直接從 working tree 複製的，所以 omlx/ 必須跟 brew 部署中的
# commit 相同，而且沒有未 commit 的改動，兩邊才是同一份程式。建好的 app 會在
# Contents/Resources/omlx-source-commit 記下來源 commit，deploy_fast.sh --status 用它比對。
#
# 變數一律寫成 ${VAR}（理由見 deploy_fast.sh 開頭）。
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd -P)
APP_DIR=${REPO}/apps/omlx-mac
STAGE=${APP_DIR}/build/Stage/oMLX.app
DEST=/Applications/oMLX.app
DEST_NEW=/Applications/.oMLX.app.deploy
# Apple Development: Jui Chen Chien (4L22S63983)；SHA-1 固定，當身分選擇器用可以避開引號。
IDENTITY=F309AB3C905A91376F00359AEF88CE3650E8E18C
TEAM=MW4GWYGX56
ENTITLEMENTS=${APP_DIR}/Resources/oMLX.entitlements
MARKER=Contents/Resources/omlx-source-commit
HEALTH_URL=http://127.0.0.1:8000/health
BUILD_LOG=${APP_DIR}/build/deploy_app-build.log

T0=$(date +%s)
say() { printf '[deploy_app +%ds] %s\n' "$(($(date +%s) - T0))" "$*"; }
die() { say "停止：$*" >&2; exit 1; }

listener() { lsof -tiTCP:8000 -sTCP:LISTEN 2>/dev/null | head -1 || true; }

# 1. 確認要打包的 omlx/ 就是 brew 正在跑的那份。
DEPLOYED=$("${REPO}/scripts/deploy_fast.sh" --commit)
HEAD_SHA=$(git -C "${REPO}" rev-parse HEAD)
DIRTY=$(git -C "${REPO}" status --porcelain -- omlx apps packaging)
[[ -z ${DIRTY} ]] || die "omlx/、apps/、packaging/ 有未 commit 的改動，app 會帶上不在任何 commit 裡的程式：
${DIRTY}"
git -C "${REPO}" diff --quiet "${DEPLOYED}" "${HEAD_SHA}" -- omlx \
  || die "HEAD ${HEAD_SHA:0:8} 的 omlx/ 跟 brew 部署中的 ${DEPLOYED:0:8} 不同。先用 scripts/deploy_fast.sh 部署 HEAD，或 checkout ${DEPLOYED:0:8} 再跑"
say "來源 ${HEAD_SHA}（omlx/ 與 brew 部署中的 ${DEPLOYED:0:8} 相同）"

# 2. 憑證：看不到不代表憑證不見，通常是這個 shell 的 sandbox 讀不到 Keychain。
security find-identity -v -p codesigning 2>/dev/null | grep -q "${IDENTITY}" \
  || die "這個 shell 看不到簽章憑證 ${IDENTITY:0:8}…（多半是 sandbox 讀不到 Keychain）。換一個看得到憑證的 shell 再跑，不要重新登入或重建憑證"

# 3. build：donor 指紋沒變時會沿用 packaging/_export，只重貼 framework-mlx-base 和 omlx。
# build.sh 用 PYTHON_BIN 編 custom kernel，ABI 必須跟 app 內附的 CPython 相同；沒指定時它抓
# PATH 上的 python3，Homebrew 把 python3 升到 3.14 後就對不上 app 的 3.11（2026-10-08）。
# 沒指定 PYTHON_BIN 就照 packaging/venvstacks.toml 的版本找同版的 python。這支 Python 還要
# import 得到 nanobind，版本照 pyproject 的 build pin（build.sh 會檢查並印出安裝指令）。
if [[ -z ${PYTHON_BIN:-} ]]; then
  PY_MM=$(sed -n 's/^python_implementation = "cpython@\([0-9]*\.[0-9]*\)\..*/\1/p' \
    "${REPO}/packaging/venvstacks.toml" | head -1)
  [[ -n ${PY_MM} ]] || die "讀不到 packaging/venvstacks.toml 的 CPython 版本，用 PYTHON_BIN 指定編 kernel 的 Python"
  PYTHON_BIN=$(command -v "python${PY_MM}" || true)
  [[ -n ${PYTHON_BIN} ]] || die "找不到 python${PY_MM}（app 內附 CPython ${PY_MM}，編 kernel 要同版）：brew install python@${PY_MM}，或用 PYTHON_BIN 指定"
fi
export PYTHON_BIN
say "編 kernel 的 Python：${PYTHON_BIN}"
say "build.sh release（log：${BUILD_LOG}）"
mkdir -p "$(dirname "${BUILD_LOG}")"
if ! "${APP_DIR}/Scripts/build.sh" release >"${BUILD_LOG}" 2>&1; then
  tail -20 "${BUILD_LOG}" >&2
  die "build.sh release 失敗"
fi
grep -E "Using donor|Rebuilding donor|Copying framework-mlx-base" "${BUILD_LOG}" \
  | sed -e $'s/\x1b\\[[0-9;]*m//g' -e 's/^/  /' || true
[[ -d ${STAGE} ]] || die "找不到 ${STAGE}"
echo "${HEAD_SHA}" >"${STAGE}/${MARKER}"

# 4. build.sh 每次都 ad-hoc 簽，要重簽。broken symlink 會讓 codesign 報
#    No such file or directory，有才清，沒有就不動。
BROKEN=$(find "${STAGE}" -type l ! -exec test -e {} \; -print)
if [[ -n ${BROKEN} ]]; then
  say "清掉 $(printf '%s\n' "${BROKEN}" | wc -l | tr -d ' ') 個 broken symlink："
  printf '%s\n' "${BROKEN}" | sed 's/^/  /'
  printf '%s\n' "${BROKEN}" | while IFS= read -r link; do rm -f "${link}"; done
else
  say "broken symlink：0 個"
fi

say "簽 Contents/Resources/Python 裡的 embedded Mach-O"
SIGN_ERR=$(mktemp)
trap 'rm -f "${SIGN_ERR}"' EXIT
signed=0
failed=0
while IFS= read -r -d '' file; do
  signed=$((signed + 1))
  codesign --force --sign "${IDENTITY}" --timestamp=none --options runtime "${file}" \
    2>>"${SIGN_ERR}" || failed=$((failed + 1))
done < <(find "${STAGE}/Contents/Resources/Python" -type f \
  \( -name '*.so' -o -name '*.dylib' -o -perm -u+x \) -print0)
NOISE=$(grep -v "replacing existing signature" "${SIGN_ERR}" || true)
say "embedded Mach-O：簽 ${signed} 個，失敗 ${failed} 個"
if [[ -n ${NOISE} ]]; then
  printf '%s\n' "${NOISE}" | sort | uniq -c | sort -rn | head -10 | sed 's/^/  /'
fi
((failed == 0)) || die "有 ${failed} 個檔案簽不起來"

codesign --force --sign "${IDENTITY}" --timestamp=none --options runtime \
  --entitlements "${ENTITLEMENTS}" "${STAGE}" 2>>"${SIGN_ERR}"
INFO=$(codesign -dvvv "${STAGE}" 2>&1)
grep -q "Authority=Apple Development: Jui Chen Chien" <<<"${INFO}" \
  && grep -q "TeamIdentifier=${TEAM}" <<<"${INFO}" \
  || die "外層簽章不是 Jason 的 Apple Development 憑證"
VERIFY=$(codesign --verify --deep --strict --verbose=4 "${STAGE}" 2>&1) \
  || { printf '%s\n' "${VERIFY}" | tail -5 >&2; die "staged app 驗證失敗"; }
say "staged app：$(grep -cE 'valid on disk|satisfies its Designated Requirement' <<<"${VERIFY}")/2 項驗證通過"
if grep -q "No such file or directory" "${SIGN_ERR}"; then
  say "注意：簽章過程出現 No such file or directory（macho sign fault），但 verify 已通過"
fi

# 5. 替換 /Applications/oMLX.app：先關 app（否則 rm 會留下跑著已刪檔案的程序），
#    先複製到旁邊再換名字，縮短 app 不存在的時間。關 app 不會停 attach 的 brew service。
OWNER_BEFORE=$(listener)
#    直接 pkill（Jason 2026-10-06 定）：osascript 的 quit 每次都被 app 的確認框擋下，
#    白等 20 秒才 pkill。SIGTERM 5 秒內沒結束才送 SIGKILL。
if pgrep -x oMLX >/dev/null; then
  say "關閉 oMLX app（pkill）"
  pkill -x oMLX || true
  for _ in $(seq 10); do
    pgrep -x oMLX >/dev/null || break
    sleep 0.5
  done
  if pgrep -x oMLX >/dev/null; then
    say "SIGTERM 5 秒內沒結束，改送 SIGKILL"
    pkill -9 -x oMLX || true
    sleep 1
  fi
  ! pgrep -x oMLX >/dev/null || die "oMLX 關不掉，沒有替換 ${DEST}"
fi

rm -rf "${DEST_NEW}"
ditto "${STAGE}" "${DEST_NEW}"
xattr -dr com.apple.quarantine "${DEST_NEW}" 2>/dev/null || true
rm -rf "${DEST}"
mv "${DEST_NEW}" "${DEST}"
codesign --verify --deep --strict "${DEST}" 2>/dev/null || die "${DEST} 驗證失敗"
say "已替換 ${DEST}"

# 6. 開 app，確認 attach 成立：app 在跑、8000 的 owner 沒變（brew 的 omlx-server、
#    PPID 1）、/health healthy。
open "${DEST}"
for _ in $(seq 30); do
  pgrep -x oMLX >/dev/null && break
  sleep 1
done
pgrep -x oMLX >/dev/null || die "oMLX app 30 秒內沒起來"
sleep 5
OWNER_AFTER=$(listener)
[[ -n ${OWNER_AFTER} ]] || die "8000 沒有 listener"
read -r OWNER_PPID OWNER_CMD < <(ps -o ppid=,command= -p "${OWNER_AFTER}")
if [[ -n ${OWNER_BEFORE} && ${OWNER_AFTER} != "${OWNER_BEFORE}" ]]; then
  die "8000 的 owner 從 pid ${OWNER_BEFORE} 變成 ${OWNER_AFTER}（${OWNER_CMD}），app 沒有 attach"
fi
[[ ${OWNER_CMD} == *omlx-server* && ${OWNER_PPID} == 1 ]] \
  || die "8000 的 owner 是 pid ${OWNER_AFTER}（${OWNER_CMD}，PPID ${OWNER_PPID}），不是 launchd 撐的 brew service"
HEALTH=$(curl -sS -m 5 "${HEALTH_URL}" 2>/dev/null || true)
[[ ${HEALTH} == *'"status":"healthy"'* ]] || die "/health 不是 healthy：${HEALTH:-無回應}"
say "完成：app pid $(pgrep -x oMLX | head -1)，8000 owner pid ${OWNER_AFTER}（${OWNER_CMD}，PPID ${OWNER_PPID}），/health healthy"
