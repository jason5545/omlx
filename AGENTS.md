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

## 本地 patch

這個 fork 保留的本地改動、每一條的行為底線、合併 upstream 時要守的檔案，都在 [`PATCHES.md`](PATCHES.md)。改到那些檔案、新增本地 patch、或 upstream 吸收了某條修法時，同步更新 `PATCHES.md`，不要寫回這裡。

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
scripts/deploy_app.sh      # Mac app 同步部署（brew 裝的是 origin/main，repo 要先在同一個 commit）
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

## 快速部署（只改 omlx/ 的 Python）

上面的整套重裝 2026-09-23 實測約 64 分鐘：主套件那步 pip 47 分鐘、mlx-audio 16 分鐘。Formula 帶 `--no-cache-dir`，每次都重建 venv、重新下載全部 wheel 和 git 依賴；Rust 套件（tokenizers、pydantic-core 等）的原始碼編譯只佔一兩分鐘。omlx 本身是純 Python wheel（`py3-none-any`；沒帶 `--with-custom-kernel` 時沒有 native extension），只改 omlx 程式時不必重建 venv，把 omlx 套件從 commit 重裝進現有 venv，再重啟 service 就好。

可以走快速路徑：

- 要上線的改動只在 `omlx/` 底下（Python，以及 admin templates、static、i18n 這類套件內的檔案）。同一段 commit 裡 `tests/`、文件、`apps/`、`packaging/` 的改動不進 server，可以一起帶。`packaging/` 只影響 DMG／Mac app 的建置，brew venv 不使用它。

一定要整套重裝：

- 改到 `pyproject.toml`（依賴、extras、版本 pin、build-system）、`setup.py`、`Formula/omlx.rb`。Formula 只動 `url`／`sha256`（upstream 發版換 release tarball）不算：`--HEAD` 安裝不用這兩行，腳本會略過它們。
- 要升級或新增依賴、有需要編譯的東西（custom kernel），或這份安裝帶 `--with-custom-kernel`：重裝純 Python wheel 會把編譯好的 kernel 蓋掉。
- 快速路徑失敗、`/health` 回不到 `healthy`，或模型載入失敗。

判斷依賴有沒有變，是跟「venv 依賴基準」比：brew 整套安裝時的 commit（`INSTALL_RECEIPT.json` 的 `source.scm_revision`），不是上一次快速部署的 commit。腳本會自動比對，上面列的檔案有變就拒絕。

brew service 和 Mac app 一律同步部署（Jason 2026-09-30 定）：`deploy_fast.sh` 部署完 brew service 後，預設接著跑 `scripts/deploy_app.sh`（見「Mac app 操作」）。只有 Jason 明說只部署一邊時才加 `--no-app`。app 的 build 加重簽要幾分鐘，但 brew service 的停機時間不變；app 只是介面，換 app 時 attach 的 server 照跑。

先 commit 並 push 到 `origin/main`，再執行：

```bash
scripts/deploy_fast.sh              # 部署 origin/main，再部署 Mac app
scripts/deploy_fast.sh <commit>     # 部署指定 commit（回滾也用這個）
scripts/deploy_fast.sh --no-app     # 只部署 brew service
scripts/deploy_fast.sh --status     # 看目前部署的 commit，以及 Mac app 是否同步
```

app 內建的 omlx 是從 working tree 複製的，所以 app 那一步要求 HEAD 的 `omlx/` 跟剛部署的 commit 相同、`omlx/`、`apps/`、`packaging/` 沒有未 commit 的改動；不符合時 brew service 照樣部署完成，腳本停下來說明兩邊不同步。回滾到舊 commit 時，先 checkout 那個 commit 再跑，app 才會一起退回。

腳本做的事，也就是手動操作的等價指令：

```bash
cd /Users/jianruicheng/GitHub/omlx
git fetch origin
SHA=$(git rev-parse origin/main)                    # 或要部署的 commit；必須在 origin/main 上
PIP=$(readlink -f /opt/homebrew/opt/omlx)/libexec/bin/pip
BASE=$(jq -r .source.scm_revision /opt/homebrew/opt/omlx/INSTALL_RECEIPT.json)
git diff --name-only "$BASE" "$SHA" -- pyproject.toml setup.py Formula             # 有輸出就改走整套重裝（Formula 只動 url／sha256 除外，要人工看 diff）
"$PIP" wheel --no-deps -w "$(mktemp -d)" "git+file://$PWD@$SHA"                   # 預先建置，不動正式 venv
# 確認沒有進行中的請求（見下面），再換檔案重啟
brew services stop jason5545/omlx/omlx
"$PIP" install --no-deps --force-reinstall "git+file://$PWD@$SHA"
brew services start jason5545/omlx/omlx
curl -sS http://127.0.0.1:8000/health                                              # 等到 "status":"healthy"
scripts/deploy_app.sh                                                              # Mac app 同步部署
```

細節：

- 用 venv 自己的 `pip`（shebang 是 Cellar 的 `python3.11`），重新產生的 `bin/omlx` 才會跟 brew 裝的一樣。
- 預先建置確認 wheel 建得起來，也把建置依賴（`pyproject.toml` build-system 的 mlx 0.32.2、cmake、nanobind）抓進 pip 快取。第一次要下載，這台實測 224 秒；之後約 6 秒。
- 2026-09-23 實測部署 0a34f284：整支腳本 18.6 秒，其中 install 7 秒，停 service 到 `/health` healthy 11 秒。healthy 時模型還沒載入（`loaded_count: 0`），第一個請求才載入 Ornith（`model_load_duration` 5.83 秒）。所以部署後要送一題短請求，確認模型真的載得起來。
- 先停 service 再換檔案：pip 會先移除舊檔再放新檔，舊程序在這段時間 lazy import 會拿到新舊混雜的模組，甚至找不到模組。停掉之後任何一步失敗，腳本都會把 service 開回來；pip 安裝失敗會自動還原舊版。
- 從 commit 安裝，不用 editable install（`pip install -e`）。`git+file` 會 clone repo 再建 wheel，只看 commit 內容：working tree 沒 commit 的改動和沒追蹤的檔案（例如本地建出的 `.so`，package-data 會收）都不會帶上線。editable install 會讓正式 server 跟著 working tree 變，查不到跑的是哪個 commit。

確認部署的是哪個 commit：以 omlx `direct_url.json` 的 `vcs_info.commit_id` 為準，`scripts/deploy_fast.sh --status` 會印出來。手動查：

```bash
$(readlink -f /opt/homebrew/opt/omlx)/libexec/bin/python -I -c \
  'from importlib.metadata import distribution as d; print(d("omlx").read_text("direct_url.json"))'
```

快速部署不經過 brew，所以 Cellar 目錄名稱（`HEAD-xxxxxxx`）、`brew info`、`brew list --versions`、`INSTALL_RECEIPT.json` 都還是上次整套安裝的 commit，只代表 venv 依賴基準，不代表正在跑的程式。`direct_url.json` 沒有 `vcs_info`、只有 `dir_info` 暫存路徑，表示最後一次是 brew 整套安裝，這時才看 `INSTALL_RECEIPT.json` 的 `source.scm_revision`。

回滾：用同一條路徑裝回前一個 commit。腳本開始時會印「目前部署」的 commit，結束時印回滾指令：

```bash
scripts/deploy_fast.sh <前一個 commit>
```

前一個 commit 跟依賴基準比有動到依賴檔的話，腳本會拒絕，這時只能整套重裝。

重啟前確認沒有進行中的請求：先看 `~/.omlx/logs/server.log` 最近 5 分鐘。server.log 只在請求結束時寫一行（`Chat completion: ... tokens in ...s`），看不到還在跑的請求，所以真正的判斷看 `/api/status` 的 `active_requests`、`waiting_requests`（要帶 `~/.omlx/settings.json` 的 API key，不要印出來）。腳本會印 log 摘要（行數、完成的請求數、最後三行），兩個數字都是 0、最後一行 log 也超過 30 秒才重啟；否則每 15 秒再查，15 分鐘還沒空就停下，不重啟。

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

donor 只在 `pyproject.toml`、`packaging/venvstacks.toml`、`uv.lock` 三個檔的指紋變了時才重建。合併 upstream 或走快速部署後，只要這三個沒動，`build.sh release` 會直接沿用 `packaging/_export`，只重貼 `framework-mlx-base`（約 1 GB）和 worktree 裡的 omlx，幾分鐘內完成。輸出看到 `Copying framework-mlx-base from donor` 就是走這條路。

`build.sh` 每次都會把 staged app 重新 ad-hoc sign，所以每次 build 之後都要重做下面的 dev 重簽。`build.sh swift` 只重編 Swift 外殼、不碰 Python layers，簽章仍在，可以跳過重簽。

部署 app 用 `scripts/deploy_app.sh`：build release、清 broken symlink（有才清）、重簽、驗證、關 app、替換 `/Applications/oMLX.app`、重開，最後確認 attach 成立，並在 `Contents/Resources/omlx-source-commit` 記下來源 commit（`deploy_fast.sh --status` 用它判斷兩邊是否同步）。`deploy_fast.sh` 會自動接著跑它；brew 整套重裝之後要手動跑。下面各段是它做的事，也是手動時的等價步驟。

要部署給 Jason 用時，必須再用 Jason 的 Apple Development cert 重簽 staged app：

```text
Apple Development: Jui Chen Chien (4L22S63983)
TeamIdentifier=MW4GWYGX56
SHA-1=F309AB3C905A91376F00359AEF88CE3650E8E18C
```

憑證能不能讀到取決於該工具的 sandbox，不要預設。先量：

```bash
security find-identity -v -p codesigning
```

看到 `1 valid identities found` 就直接簽，sandbox 不會擋。顯示 0 identities 只是這個 shell 看不到 Keychain（Codex sandbox 內的 `gh` 是同款問題），換成有憑證的 shell，不要據此判定憑證不見、也不要要求重新登入。SHA-1 固定不變，`--sign F309AB3C…` 可以當身分選擇器，避開引號。

重簽順序：

1. 先確認 broken symlink。`codesign --strict` 回 `No such file or directory` 就是它造成的（常見於 stripped dynlib links），2026-09-29 那次 build 是 0 個，所以不要無條件先清：

   ```bash
   cd apps/omlx-mac/build/Stage/oMLX.app
   find . -type l ! -exec test -e {} \; -print
   ```

2. 簽 `Contents/Resources/Python` 裡的 embedded Mach-O（`.so`/`.dylib`/可執行檔），2026-09-29 這批是 526 個、2026-09-30 是 493 個，數量會跟著 donor 內容變，看失敗數就好。簽完檢查有沒有真的失敗：

   ```bash
   for f in $(find Contents/Resources/Python -type f \( -name '*.so' -o -name '*.dylib' -o -perm -u+x \)); do
     codesign --force --sign "Apple Development: Jui Chen Chien (4L22S63983)" --timestamp=none --options runtime "$f"
   done
   ```

3. 最後簽外層 bundle，帶 entitlements：

   ```bash
   codesign --force --sign "Apple Development: Jui Chen Chien (4L22S63983)" --timestamp=none \
     --options runtime --entitlements ../../Resources/oMLX.entitlements oMLX.app
   codesign -dvvv oMLX.app   # Authority 要出現 Jui Chen Chien，TeamIdentifier=MW4GWYGX56
   ```

4. 驗 staged app：

   ```bash
   codesign --verify --deep --strict --verbose=4 apps/omlx-mac/build/Stage/oMLX.app
   ```

   看到 `valid on disk` 加 `satisfies its Designated Requirement` 就算過。Jason 講的「macho sign fault」指的就是過程中的 `No such file or directory`；那只是簽名器對某個檔案路徑的抱怨，verify 過了就不擋部署，但要在回報裡講明有沒有出現。Apple Development cert 未 notarize，`spctl --assess` 會 rejected，那不等於簽章失效。

部署（先關 app，否則 `rm -rf` 會留下跑著「已刪檔案」的程序；`osascript` 的 quit 可能被 app 的確認框擋下、回「使用者取消操作」，等不到就 `pkill`，attach 的 brew service 不受影響）：

```bash
osascript -e 'tell application "oMLX" to quit'
pkill -x oMLX   # 沒關掉的話
rm -rf /Applications/oMLX.app
ditto apps/omlx-mac/build/Stage/oMLX.app /Applications/oMLX.app
xattr -dr com.apple.quarantine /Applications/oMLX.app
codesign --verify --deep --strict --verbose=2 /Applications/oMLX.app
open /Applications/oMLX.app
```

部署後確認 attach 真的成立：8000 的 owner 必須還是 Homebrew 的 server（command 是 `omlx-server`、PPID 是 1 表示 launchd 撐的 service，不是 app 自己起的子程序）：

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
ps -o pid,ppid,command -p "$(lsof -tiTCP:8000 -sTCP:LISTEN | head -1)"
curl -sS http://127.0.0.1:8000/health | jq -c '{status, loaded: .engine_pool.loaded_count}'
```

`pgrep -x oMLX` 有 pid、`/health` 回 healthy、8000 owner 沒變，三個都對才算部署完成。

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
scripts/deploy_app.sh      # Mac app 同步部署
```

衝突落在本地保留的功能時，照 [`PATCHES.md`](PATCHES.md) 的「合併衝突檢查」逐項確認，其餘一律取 upstream 版本。

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

JANG 相關改動另外跑 `PYTHONPATH=/tmp/omlx-pytest-async python -m pytest tests/test_jang_engine.py tests/test_prism_jangq_compat.py tests/test_model_discovery.py tests/test_engine_pool.py -q`；`tests/test_engine_pool.py` 是 async，光裝 pytest 會整批報 'async def functions are not natively supported'，要一併裝 pytest-asyncio。

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
- 做完實質變更後，commit 並 push 到 `origin/main`。要上線時先看「快速部署」的判斷：只改 `omlx/` 的 Python 用 `scripts/deploy_fast.sh`；改到依賴、`Formula/omlx.rb` 或需要編譯的東西，才照「Homebrew 操作」整套重裝。
- brew service 和 Mac app 一律同步部署，同一份 `omlx/`：`deploy_fast.sh` 預設會接著部署 app，整套重裝之後手動跑 `scripts/deploy_app.sh`。只有 Jason 明說只部署一邊時才分開（Jason 2026-09-30 定）。部署完用 `scripts/deploy_fast.sh --status` 確認兩邊同步。
- 回覆 Jason 時用自然、簡短的繁體中文，少模板感。
