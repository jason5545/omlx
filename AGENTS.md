# AGENTS.md

這個 repo 是 Jason 的 oMLX 工作 fork。請把這份 clone 當成主要工作目錄，不要再把 Homebrew cache 裡的 checkout 當成 source of truth。

主要路徑：

- 工作 repo：`/Users/jianruicheng/GitHub/omlx`
- Homebrew tap：`jason5545/omlx`
- 已安裝 formula：`jason5545/omlx/omlx`
- upstream：`https://github.com/jundot/omlx`

Remote 應維持：

```bash
origin   https://github.com/jason5545/omlx.git
upstream https://github.com/jundot/omlx.git
```

`upstream` 的 push URL 應保持 disabled，避免誤推原作者 repo。

## 目前本地改動

自 0.6.3rc1 merge 起，這個 fork 盡量貼齊 upstream，只保留三個本地功能；其他一律 follow upstream（舊的 VLM/MTP、MTPLX、thinking-budget patch stack 已整批丟棄，存在 `backup/pre-upstream-merge` 分支僅供查閱，不要回移植）：

- API sub-key 可以套 request policy（`identify_api_key` → `DEFAULT_SUB_KEY_POLICIES`）。`voco` 預設是 `max_context_window<=16384` 且 `enable_thinking=false`。相關檔案：`omlx/server.py`、`omlx/admin/auth.py`、`omlx/api/openai_models.py`、`omlx/settings.py`。
- Mac app attach mode：`8000` 上已有健康 oMLX server 時 app 直接 attach，不顯示 port conflict；細節見下面 Mac app 章節。相關檔案：`apps/omlx-mac/Sources/Server/ServerProcess.swift` 等 6 個 Swift 檔。
- `Formula/omlx.rb` 的 homepage/HEAD 指向 `https://github.com/jason5545/omlx.git`（xgrammar macOS arm64 post-install patch 已被 upstream 吸收，不需再維護本地版）。
- qwen3_5_moe VLM 強制 sanitize（`omlx/engine/vlm.py` 的 `_force_qwen35_moe_sanitize_on_load`）：mlx-vlm 用第一個 glob 到的 shard metadata 判斷 `is_mlx_format`，mixed-metadata checkpoint（如 Ornith-1.5 MXFP8）會跳過 sanitize，per-expert MTP MoE 權重沒堆疊成 switch_mlp，strict load 失敗 → 退回純 LLM、視覺被靜默丟掉。追 upstream 時守住這段。
- pi / OpenCode integration 的 thinking 支援（`omlx/integrations/pi.py`、`omlx/integrations/opencode.py`）：`_is_reasoning_model` 加 `flash-next`；pi 重寫 `models.json` 時保留手動調過的 provider `compat`（`thinkingFormat: chat-template` + `chatTemplateKwargs`）與 per-model `thinkingLevelMap`；OpenCode 對 reasoning model 自動寫 `reasoning`/`interleaved`/`variants`（off/low/medium/xhigh 走 `chat_template_kwargs.enable_thinking`，Qwen3.8-Flash-Next 只能靠這個關思考），並保留手動調過的 model entry keys。
- thinking budget 診斷與 re-entry 修正（`omlx/scheduler.py`、`omlx/api/thinking.py`）：processor attach/skip/觸發/re-entry 各加 info log（排查「budget 沒生效」用）；scheduler 的 `think_start_id` 對沒有 mlx-lm 屬性的 tokenizer（如 Qwen3.8 的 Qwen2Tokenizer）加 `convert_tokens_to_ids("<think>")` fallback，否則模型自然關閉思考後再開 `<think>` 時 budget 不會重新生效。

追 upstream 時，conflict 只要守住上面幾塊，其餘一律取 upstream 版本。不要留下手動改 site-packages 的最終狀態。

## Homebrew 操作

只保留 Jason 的 tap，避免同名 formula ambiguous：

```bash
brew tap | rg 'omlx'
```

應只看到：

```text
jason5545/omlx
```

安裝或重裝這個 fork：

```bash
brew services stop jason5545/omlx/omlx
brew uninstall jason5545/omlx/omlx
brew install --HEAD --with-grammar jason5545/omlx/omlx
brew services start jason5545/omlx/omlx
```

Homebrew 5.1 的 `brew reinstall` 不接受 `--HEAD`，所以需要明確 uninstall/install 時，用上面的方式最穩。

確認安裝來源：

```bash
brew info --json=v2 jason5545/omlx/omlx | jq '.formulae[0] | {full_name,tap,tap_git_head,installed,urls}'
```

`urls.head.url` 應該是：

```text
https://github.com/jason5545/omlx.git
```

## Mac app 操作

主要路徑：

- Xcode project：`/Users/jianruicheng/GitHub/omlx/apps/omlx-mac/oMLX.xcodeproj`
- scheme：`oMLX`
- staged app：`/Users/jianruicheng/GitHub/omlx/apps/omlx-mac/build/Stage/oMLX.app`
- 已部署 app：`/Applications/oMLX.app`
- app server log：`~/Library/Application Support/oMLX/logs/server.log`

Mac app 跟 Homebrew service 都預設使用 `127.0.0.1:8000`。新版 app 的正確行為是：

- 如果 `8000` 上已經是健康的 oMLX server（`/health` 回 200），app 應 attach，而不是顯示 port conflict。
- attached mode 仍可控制 server。`Stop`/`Restart` 如果辨識到 owner 是 Homebrew oMLX，應先走 `brew services stop jason5545/omlx/omlx`，因為 formula service 是 `keep_alive true`，單純 kill PID 會被 launchd 拉回來。
- app 自己 Quit / relaunch 不應停掉 attached 的 Homebrew service。只有使用者明確按 Stop/Restart 時才控制外部 owner。
- 如果 `8000` 被非 oMLX process 佔用，才應顯示 port conflict。

重開 macOS app，但不要動 Homebrew service：

```bash
osascript -e 'tell application "oMLX" to quit'
pkill -x oMLX   # 只有卡住時才用
open /Applications/oMLX.app
```

build release app：

```bash
apps/omlx-mac/Scripts/build.sh release
```

如果 `packaging/_export` 不存在或 donor layers 不完整，才重建 donor：

```bash
apps/omlx-mac/Scripts/build.sh release --rebuild-donor
```

`build.sh` 會 ad-hoc sign。要部署給 Jason 用時，必須再用 Jason 的 Apple Development cert 重簽 staged app：

```text
Apple Development: Jui Chen Chien (4L22S63983)
TeamIdentifier=MW4GWYGX56
```

憑證通常只有 escalated shell 看得到；sandbox 內 `security find-identity` 可能會顯示 0 identities。

重簽原則：

- 先清掉 staged bundle 裡的 broken symlink（常見於 stripped dynlib links），否則 `codesign --strict` 可能回 `No such file or directory`。
- 先簽 `Contents/Resources/Python` 裡的 embedded Mach-O（`.so`/`.dylib`/`.bundle`/可執行檔）。
- 最後用 `--options runtime --entitlements apps/omlx-mac/Resources/oMLX.entitlements` 簽外層 `oMLX.app`。
- 用 `codesign --verify --deep --strict --verbose=4` 驗 staged app 與 `/Applications/oMLX.app`。
- Apple Development cert 未 notarize，`spctl --assess` 可能 rejected；這不等於 `codesign --verify` 失敗。

部署：

```bash
rm -rf /Applications/oMLX.app
ditto apps/omlx-mac/build/Stage/oMLX.app /Applications/oMLX.app
xattr -dr com.apple.quarantine /Applications/oMLX.app
codesign --verify --deep --strict --verbose=4 /Applications/oMLX.app
```

## 追 upstream

更新 upstream 時請先檢查差異，不要盲目覆蓋本地 patch：

```bash
cd /Users/jianruicheng/GitHub/omlx
git fetch upstream --prune
git log --oneline --left-right --graph main...upstream/main
git merge upstream/main
```

合併後要重新跑驗證，推回 fork：

```bash
git push origin main
brew update
brew uninstall jason5545/omlx/omlx
brew install --HEAD --with-grammar jason5545/omlx/omlx
brew services restart jason5545/omlx/omlx
```

如果衝突落在本地保留的三塊功能，特別檢查：

- `omlx/server.py`（sub-key policy hooks、`DEFAULT_SUB_KEY_POLICIES`）
- `omlx/admin/auth.py`、`omlx/api/openai_models.py`、`omlx/settings.py`
- `apps/omlx-mac/Sources/Server/ServerProcess.swift`（attach mode）
- `Formula/omlx.rb`（homepage/head 要維持 jason5545）

## 最小驗證

程式碼檢查：

```bash
git diff --check
ruby -c Formula/omlx.rb
brew style Formula/omlx.rb
brew audit --formula jason5545/omlx/omlx
/opt/homebrew/opt/omlx/libexec/bin/python -m py_compile \
  omlx/admin/auth.py \
  omlx/api/openai_models.py \
  omlx/server.py \
  omlx/settings.py
```

Homebrew venv 通常沒有 `pytest`。如果沒有安裝，不要說已經跑過 pytest；改說 pytest 不在 venv。要跑測試可用隔離 target：`pip install --target=/tmp/omlx-pytest-target pytest`，再用 venv python 跑 `PYTHONPATH=/tmp/omlx-pytest-target python -m pytest tests/test_admin_api_key.py tests/test_context_window.py tests/test_api_auth.py -q`，不要裝進 venv 的 site-packages。

安裝後確認：

```bash
/opt/homebrew/opt/omlx/libexec/bin/python - <<'PY'
from omlx.server import DEFAULT_SUB_KEY_POLICIES
import xgrammar
print(DEFAULT_SUB_KEY_POLICIES["voco"])
print("xgrammar ok")
PY

curl -sS http://127.0.0.1:8000/health
```

`voco` sub-key 的 log 應出現：

```text
Request policy active: client=voco source=api-sub-key ... max_context_window<=16384, enable_thinking=False
```

## 操作習慣

- 不要把 API key 印到 log 或回覆裡。
- 不要把 `jundot/omlx` tap 裝回來，除非 Jason 明確要求。
- 不要把 Homebrew cache 裡的 checkout 當主要 repo 修改。
- 做完實質變更後，commit 並 push 到 `origin/main`，再視需要重裝 tap。
- 回覆 Jason 時用自然、簡短的繁體中文，少模板感。
