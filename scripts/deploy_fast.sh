#!/usr/bin/env bash
# 快速部署：只改 omlx/ 的 Python 時，把指定 commit 的 omlx 套件重裝進 Homebrew venv，
# 不動依賴，再重啟 brew service。什麼時候能用、什麼時候要整套重裝，見 AGENTS.md「快速部署」。
# 一律連 custom kernel（ANE prefill、INT8 prefill 等的 native extension）一起編（Jason 2026-10-07 定）。
# brew service 和 Mac app 一律同步部署：成功後接著跑 scripts/deploy_app.sh。
#
#   scripts/deploy_fast.sh             部署 origin/main（再部署 Mac app）
#   scripts/deploy_fast.sh <commit>    部署指定 commit（回滾也用這個）
#   scripts/deploy_fast.sh --no-app    只部署 brew service（Jason 明說才用）
#   scripts/deploy_fast.sh --status    只印目前部署的 commit，以及 Mac app 是否同步
#   scripts/deploy_fast.sh --commit    只印目前部署的 commit（給 deploy_app.sh 用）
#
# macOS 內建 bash 3.2 在 C locale 會把緊貼在變數後面的全形字元當成變數名稱
# （"$SHA（" 會變成 unbound variable），所以變數一律寫成 ${VAR}。
set -euo pipefail

FORMULA=jason5545/omlx/omlx
OPT=/opt/homebrew/opt/omlx
REPO=$(cd "$(dirname "$0")/.." && pwd -P)
HEALTH_URL=http://127.0.0.1:8000/health
SERVER_LOG=${HOME}/.omlx/logs/server.log
WAIT_IDLE_MAX=900     # 有請求在跑時最多等 15 分鐘
HEALTH_MAX=600        # 重啟後最多等 10 分鐘 healthy

CELLAR=$(readlink -f "${OPT}")
PY=${CELLAR}/libexec/bin/python
# 用 venv 自己的 pip（shebang 是 Cellar 的 python3.11），重新產生的 bin/omlx 才跟 brew 裝的一樣。
PIP=${CELLAR}/libexec/bin/pip
RECEIPT=${CELLAR}/INSTALL_RECEIPT.json
# 跟 Formula/omlx.rb 的 CUSTOM_KERNELS 同一份清單。
CUSTOM_KERNELS="bonsai decode_fast glm_moe_dsa minimax_m3 qwen35_prefill"

T0=$(date +%s)
say() { printf '[deploy_fast +%ds] %s\n' "$(($(date +%s) - T0))" "$*"; }
die() { say "停止：$*" >&2; exit 1; }

# 目前部署的 commit：快速部署過的看 direct_url.json 的 vcs_info.commit_id；
# brew 整套安裝的 direct_url.json 只有暫存目錄，改看 INSTALL_RECEIPT.json 的 scm_revision。
deployed_commit() {
  "${PY}" -I - "${RECEIPT}" <<'PY'
import json, sys
from importlib.metadata import distribution
dist = distribution("omlx")
vcs = json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info") or {}
if vcs.get("commit_id"):
    print(vcs["commit_id"], "direct_url.json（快速部署）", dist.version, sep="\t")
else:
    receipt = json.load(open(sys.argv[1]))
    print(receipt["source"]["scm_revision"], "INSTALL_RECEIPT.json（brew 整套安裝）", dist.version, sep="\t")
PY
}

WITH_APP=1
ARGS=()
for arg in "$@"; do
  case ${arg} in
    --no-app) WITH_APP=0 ;;
    *) ARGS+=("${arg}") ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

if [[ ${1:-} == --commit ]]; then
  deployed_commit | cut -f1
  exit 0
fi

if [[ ${1:-} == --status ]]; then
  IFS=$'\t' read -r sha source version < <(deployed_commit)
  echo "部署中：${sha}  來源：${source}  版本：${version}"
  echo "venv 依賴基準（brew 整套安裝時的 commit）：$(jq -r .source.scm_revision "${RECEIPT}")"
  APP_SHA=$(cat /Applications/oMLX.app/Contents/Resources/omlx-source-commit 2>/dev/null || true)
  if [[ -z ${APP_SHA} ]]; then
    echo "Mac app：沒有來源 commit 記錄（不是 scripts/deploy_app.sh 部署的），無法確認是否同步"
  elif git -C "${REPO}" diff --quiet "${APP_SHA}" "${sha}" -- omlx 2>/dev/null; then
    echo "Mac app：${APP_SHA}，omlx/ 與 brew 同步"
  else
    echo "Mac app：${APP_SHA}，omlx/ 與 brew 不同步，跑 scripts/deploy_app.sh"
  fi
  exit 0
fi

# 印 server.log 最近 5 分鐘的摘要，再用 /api/status 判斷有沒有進行中的請求。
# server.log 只在請求結束時寫一行，看不到進行中的請求，所以以 active/waiting 為準；
# 最後一行 log 在 30 秒內也當作忙碌，避免切在連續請求中間。
# 回傳 0 = 閒置（或 server 沒在跑），1 = 忙碌，2 = 查不到狀態。
idle_check() {
  "${PY}" -I - "${SERVER_LOG}" <<'PY'
import datetime as dt, json, os, re, sys, urllib.error, urllib.request

now = dt.datetime.now()
recent = []
try:
    with open(sys.argv[1], "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4_000_000))
        lines = f.read().decode("utf-8", "replace").splitlines()
except FileNotFoundError:
    lines = []
for line in lines:
    try:
        ts = dt.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        continue
    if now - ts <= dt.timedelta(minutes=5):
        recent.append((ts, line))
finished = sum(bool(re.search(r"tokens in [0-9.]+s", line)) for _, line in recent)
last_age = (now - recent[-1][0]).total_seconds() if recent else None
print(f"server.log 最近 5 分鐘：{len(recent)} 行，完成的請求 {finished} 筆"
      + (f"，最後一行在 {last_age:.0f} 秒前" if recent else ""))
for _, line in recent[-3:]:
    print("  " + line[:200])

settings = os.path.expanduser("~/.omlx/settings.json")
key = json.load(open(settings)).get("auth", {}).get("api_key") if os.path.exists(settings) else None
req = urllib.request.Request(
    "http://127.0.0.1:8000/api/status",
    headers={"Authorization": f"Bearer {key}"} if key else {},
)
try:
    status = json.load(urllib.request.urlopen(req, timeout=5))
except urllib.error.HTTPError as exc:
    print(f"/api/status 回 {exc.code}")
    sys.exit(2)
except urllib.error.URLError as exc:
    if isinstance(exc.reason, ConnectionRefusedError):
        print("server 沒在跑，不用等")
        sys.exit(0)
    print(f"/api/status 連不上：{exc.reason}")
    sys.exit(2)
active = status.get("active_requests", 0)
waiting = status.get("waiting_requests", 0)
print(f"/api/status：active_requests={active} waiting_requests={waiting}")
sys.exit(1 if active or waiting or (last_age is not None and last_age < 30) else 0)
PY
}

TMP=$(mktemp -d)
STOPPED=0
STARTED=0
cleanup() {
  rm -rf "${TMP}"
  # 停了 service 之後任何一步失敗，都把 service 開回來（pip 失敗會自動還原舊版）。
  if ((STOPPED && !STARTED)); then
    say "中途失敗，把 service 開回來"
    brew services start "${FORMULA}" >/dev/null || true
  fi
}
trap cleanup EXIT

# 1. 決定要部署的 commit：必須已經 push 到 origin/main。
git -C "${REPO}" fetch -q origin
SHA=$(git -C "${REPO}" rev-parse --verify "${1:-origin/main}^{commit}")
git -C "${REPO}" merge-base --is-ancestor "${SHA}" origin/main \
  || die "${SHA} 不在 origin/main 上，先 commit 並 push"
IFS=$'\t' read -r PREV _ < <(deployed_commit)
say "目標 ${SHA}（目前部署 ${PREV}）"
if [[ -n $(git -C "${REPO}" status --porcelain -- omlx) ]]; then
  say "注意：working tree 的 omlx/ 有未 commit 的改動，這次不會帶上線"
fi

# 2. 只能在依賴沒變時走快速路徑：跟 venv 依賴基準（brew 整套安裝的 commit）比。
BASE=$(jq -r .source.scm_revision "${RECEIPT}")
CHANGED=$(git -C "${REPO}" diff --name-only "${BASE}" "${SHA}" -- pyproject.toml setup.py)
# Formula 只有 url／sha256（release tarball）變不算：HEAD 安裝不用它們，每次 upstream 發版都會動。
FORMULA_REAL=$(git -C "${REPO}" diff -U0 "${BASE}" "${SHA}" -- Formula \
  | { grep -E '^[+-][^+-]' || true; } | { grep -vE '^[+-][[:space:]]*(url|sha256)[[:space:]]' || true; })
[[ -z ${FORMULA_REAL} ]] || CHANGED=$(printf '%s\nFormula/omlx.rb' "${CHANGED}" | sed '/^$/d')
[[ -z ${CHANGED} ]] || die "跟 venv 依賴基準 ${BASE:0:8} 比，改到下列檔案，要整套重裝：
${CHANGED}"

# 3. custom kernel 一律一起編，跟 brew --with-custom-kernel 同一套環境變數（setup.py 看
#    OMLX_WITH_CUSTOM_KERNEL）。pip 的 wheel 快取每次換一個空目錄：舊的純 Python wheel 是
#    py3-none-any，同一個 commit 以前建過就會被直接拿來裝，kernel 就悄悄不見；下載快取
#    （http、http-v2）照樣共用，建置依賴不用重抓。帶 kernel 的 wheel 是 macosx_27_0 tag，
#    pip 不會從快取拿它，所以停 service 後的安裝還是會重編一次（2026-10-07 實測停機 70 秒）。
xcrun -f metal >/dev/null 2>&1 \
  || die "找不到 Metal 編譯器（xcrun -f metal），編不了 custom kernel。先跑 xcodebuild -downloadComponent MetalToolchain"
export OMLX_WITH_CUSTOM_KERNEL=1
USER_CMAKE_ARGS=${CMAKE_ARGS-}
export CMAKE_ARGS="${CMAKE_ARGS:+${CMAKE_ARGS} }-DPython_EXECUTABLE=${PY}"
REAL_PIP_CACHE=$("${PIP}" cache dir)
export PIP_CACHE_DIR=${TMP}/pip-cache
mkdir -p "${PIP_CACHE_DIR}"
for d in http http-v2; do
  [[ -d ${REAL_PIP_CACHE}/${d} ]] && ln -s "${REAL_PIP_CACHE}/${d}" "${PIP_CACHE_DIR}/${d}"
done

# 4. 先建一次 wheel：確認建得起來、kernel 都在，也把建置依賴（setuptools、cmake、nanobind、
#    mlx）抓進下載快取。第一次要下載，之後約 35～95 秒（多半是編 kernel）。這一步不動正式 venv。
say "預先建置 wheel（含 custom kernel）"
"${PIP}" wheel -q --no-deps -w "${TMP}/wheel" "git+file://${REPO}@${SHA}"
"${PY}" -I - "${TMP}"/wheel/omlx-*.whl ${CUSTOM_KERNELS} <<'PY' || die "預建的 wheel 缺 custom kernel，沒有重啟"
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
missing = [k for k in sys.argv[2:]
           if not any(n.startswith(f"omlx/custom_kernels/{k}/_ext") and n.endswith(".so") for n in names)]
if missing:
    print("缺：" + " ".join(missing))
    sys.exit(1)
PY

# 5. 等沒有進行中的請求。
say "檢查進行中的請求（最多等 $((WAIT_IDLE_MAX / 60)) 分鐘）"
deadline=$(($(date +%s) + WAIT_IDLE_MAX))
while :; do
  rc=0
  idle_check || rc=$?
  ((rc == 0)) && break
  ((rc == 1)) || die "查不到 server 狀態，沒有重啟"
  (($(date +%s) < deadline)) || die "等 $((WAIT_IDLE_MAX / 60)) 分鐘還有請求在跑，沒有重啟"
  sleep 15
done

# 6. 先停 service 再換檔案：避免舊程序在換檔中途 lazy import 到新舊混雜的模組。
T_STOP=$(date +%s)
PID=$(brew services info --json "${FORMULA}" | jq -r '.[0].pid // empty')
say "停止 service（pid ${PID:-無}）"
brew services stop "${FORMULA}" >/dev/null
STOPPED=1
if [[ -n ${PID} ]]; then
  for _ in $(seq 120); do
    kill -0 "${PID}" 2>/dev/null || break
    sleep 1
  done
  ! kill -0 "${PID}" 2>/dev/null || die "pid ${PID} 兩分鐘內沒結束"
fi

say "安裝 omlx @ ${SHA}（--no-deps --force-reinstall）"
T_INSTALL=$(date +%s)
"${PIP}" install -q --no-deps --force-reinstall "git+file://${REPO}@${SHA}"
say "安裝完成，花 $(($(date +%s) - T_INSTALL)) 秒"
# 跟 Formula/omlx.rb 的 verify_custom_kernels 同一套檢查：載得起來、ABI 對得上。
# 失敗時 cleanup 會把 service 開回來（跑的是新版但沒有 kernel），照訊息回滾。
"${PY}" -I - ${CUSTOM_KERNELS} <<'PY' || die "custom kernel 載入失敗，service 會以沒有 kernel 的狀態開回來；回滾：scripts/deploy_fast.sh ${PREV}"
import importlib, sys
failed = {}
for package in sys.argv[1:]:
    fast = importlib.import_module(f"omlx.custom_kernels.{package}.fast")
    if not fast.is_native_available():
        failed[package] = str(fast.import_error())
if failed:
    print(failed)
    sys.exit(1)
PY
say "custom kernel 驗證通過（${CUSTOM_KERNELS}）"

brew services start "${FORMULA}" >/dev/null
STARTED=1

# 7. 等 /health 回 healthy（載入中是 503 + "loading"）。
say "等 /health 回 healthy"
body=
while (($(date +%s) - T_STOP < HEALTH_MAX)); do
  body=$(curl -sS -m 5 "${HEALTH_URL}" 2>/dev/null || true)
  [[ ${body} == *'"status":"healthy"'* ]] && break
  sleep 2
done
[[ ${body} == *'"status":"healthy"'* ]] \
  || die "/health $((HEALTH_MAX / 60)) 分鐘內沒回 healthy：${body:-無回應}。改走 AGENTS.md 的整套重裝"
say "healthy：${body}"

IFS=$'\t' read -r NOW_SHA NOW_SOURCE _ < <(deployed_commit)
[[ ${NOW_SHA} == "${SHA}" ]] || die "部署後 commit 是 ${NOW_SHA}，不是 ${SHA}"
say "完成：部署 ${NOW_SHA}（${NOW_SOURCE}），停 service 到 healthy $(($(date +%s) - T_STOP)) 秒"
say "回滾：scripts/deploy_fast.sh ${PREV}"

# 8. Mac app 同步部署。app 從 working tree 建，所以 HEAD 的 omlx/ 要跟剛部署的 commit 相同。
if ((WITH_APP)); then
  say "接著部署 Mac app（scripts/deploy_app.sh）"
  # 步驟 3 給 brew venv 的 CMAKE_ARGS 不能漏給 app：裡面的 Python_EXECUTABLE 會蓋掉
  # deploy_app.sh 選的 Python，cmake 改去 brew venv 找 nanobind 而失敗（2026-10-08）。
  env -u PIP_CACHE_DIR -u OMLX_WITH_CUSTOM_KERNEL CMAKE_ARGS="${USER_CMAKE_ARGS}" \
    "${REPO}/scripts/deploy_app.sh" \
    || die "brew service 已部署 ${SHA:0:8}，但 Mac app 沒部署成功，兩邊不同步。照上面的原因處理後單獨跑 scripts/deploy_app.sh"
else
  say "--no-app：沒有部署 Mac app，兩邊可能不同步（scripts/deploy_fast.sh --status 可查）"
fi
