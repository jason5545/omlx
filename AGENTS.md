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

自 0.6.3rc1 merge 起，這個 fork 盡量貼齊 upstream，只保留九個本地功能；其他一律 follow upstream（舊的 VLM/MTP、MTPLX、thinking-budget patch stack 已整批丟棄，存在 `backup/pre-upstream-merge` 分支僅供查閱，不要回移植）：

- API sub-key 可以套 request policy（`identify_api_key` → `DEFAULT_SUB_KEY_POLICIES`）。`voco` 預設是 `max_context_window<=16384` 且 `enable_thinking=false`。相關檔案：`omlx/server.py`、`omlx/admin/auth.py`、`omlx/api/openai_models.py`、`omlx/settings.py`。
- Mac app attach mode：`8000` 上已有健康 oMLX server 時 app 直接 attach，不顯示 port conflict；細節見下面 Mac app 章節。相關檔案：`apps/omlx-mac/Sources/Server/ServerProcess.swift` 等 6 個 Swift 檔。
- `Formula/omlx.rb` 的 homepage/HEAD 指向 `https://github.com/jason5545/omlx.git`（xgrammar macOS arm64 post-install patch 已被 upstream 吸收，不需再維護本地版）。
- qwen3_5_moe VLM 強制 sanitize（`omlx/engine/vlm.py` 的 `_force_qwen35_moe_sanitize_on_load`）：mlx-vlm 用第一個 glob 到的 shard metadata 判斷 `is_mlx_format`，mixed-metadata checkpoint（如 Ornith-1.5 MXFP8）會跳過 sanitize，per-expert MTP MoE 權重沒堆疊成 switch_mlp，strict load 失敗 → 退回純 LLM、視覺被靜默丟掉。追 upstream 時守住這段。
- EnginePool 只允許一個 resident model：harness 或 API request 從 model A 切到 model B 時，先 unload 其他閒置 engine，再載入 B；如果 A 有 active request 或 lease，回 `ModelBusyError`，不要硬拆進行中的 request。相關檔案：`omlx/engine_pool.py`、`omlx/admin/routes.py`、`tests/test_engine_pool.py`。
- `packaging/build.py` 下載一律走 `_urlopen`／`_ssl_context`，不直接用 `urllib.request.urlretrieve`：python.org 的 macOS 直譯器（build driver 預設用 PATH 上的 `python3`）附的 OpenSSL 沒有預設信任庫，spacy 模型下載會以 `CERTIFICATE_VERIFY_FAILED` 失敗，donor 重建中斷在 `_install_spacy_model`。helper 依序找 `SSL_CERT_FILE`、certifi、`/etc/ssl/cert.pem` 等系統 bundle。相關檔案：`packaging/build.py`。
- JANGQ affine-ternary prism 轉接：dealignai 的 Bonsai-2-27B-*-Ternary-JANG 沿用 PrismML 的 ternary 權重，但把 `model_type` 改成 `qwen3_5`、又加 `storage_bits`，所以走不到 upstream #3782 已支援的 `prism_hadamard_qwen35`；載入時把 config 正規化成 schema-2、重建 modules manifest、拿掉 `storage_bits`、放寬 prism 的量化檢查、補 161 個 zero-centered norm 的 +1.0，全部在一次載入內 patch 並還原。相關檔案：`omlx/patches/prism_jangq_compat.py`、`omlx/utils/model_loading.py` 的 `maybe_load_jangq_prism`、`omlx/engine/vlm.py` 與 `omlx/engine/batched.py` 的呼叫點（batched 端只負責擋純文字載入）、`tests/test_prism_jangq_compat.py`。
- JANG mixed-precision bundle 轉接：逐張量 bit width 記在 sidecar（`jang_config.json` 等），stock mlx-lm／mlx-vlm 讀不到，所以交給 `jang_tools.loader` 載入後再進正常的 BatchedEngine／VLMBatchedEngine——不要改成新增 engine 類別，server 有一批 `isinstance(engine, VLMBatchedEngine)` 的圖片、prefix cache、tool calling 判定會斷。閘門要求 sidecar 的 `format` 是 `jang`／`jjqf`／`mxq`：只有 vMLX sidecar、`format` 未設的 MXFP8 包（如 Ornith-1.5 MXFP8）要留給原路徑，不要搶過來。`omlx/patches/jang_load.py` 另外補 jang runtime 兩個洞：Nemotron-H gate 的後綴比對（上游 PR #364 那段是死碼，從沒解量化過任何 gate）與 6-bit + `--hadamard` 的 sign 寬度（它用 `packed_cols * (32 // bits)`，6-bit 會算成 60）。相關檔案：`omlx/patches/jang_load.py`、`omlx/utils/model_loading.py` 的 `maybe_load_jang`、`omlx/engine/batched.py` 與 `omlx/engine/vlm.py` 的載入插入點、`omlx/model_discovery.py` 的 `JANG_CONFIG_FILES`／`_jang_has_vision`、`omlx/exceptions.py` 的 `JANGDependencyError`／`JANGLoadError`、`tests/test_jang_engine.py`。依賴是 `pyproject.toml` 的 `jang` extra 加 `Formula/omlx.rb` 一行 `system(*pip_install, "jang[mlx]>=2.5.47")`（必須共用 `pip_install` flags，`tests/test_homebrew_formula.py` 會數裸 pip 呼叫）；上游 PR #364 合併後可整批換成 upstream 版。
- GDN MTP verify prework 的 upstream kernel 擴充為同時支援 fp16／bf16：輸入、conv state、conv1d 權重 dtype 必須相同，scale 也用該 dtype；fp16 和既有 bf16 組合路徑要求逐位元一致。追 upstream 時守住。相關檔案：`omlx/patches/qwen35_gdn_prework.py`、`tests/test_qwen35_gdn_prework.py`。

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

## 快速部署（只改 omlx/ 的 Python）

上面的整套重裝 2026-09-23 實測約 64 分鐘：主套件那步 pip 47 分鐘、mlx-audio 16 分鐘。Formula 帶 `--no-cache-dir`，每次都重建 venv、重新下載全部 wheel 和 git 依賴；Rust 套件（tokenizers、pydantic-core 等）的原始碼編譯只佔一兩分鐘。omlx 本身是純 Python wheel（`py3-none-any`；沒帶 `--with-custom-kernel` 時沒有 native extension），只改 omlx 程式時不必重建 venv，把 omlx 套件從 commit 重裝進現有 venv，再重啟 service 就好。

可以走快速路徑：

- 要上線的改動只在 `omlx/` 底下（Python，以及 admin templates、static、i18n 這類套件內的檔案）。同一段 commit 裡 `tests/`、文件、`apps/`、`packaging/` 的改動不進 server，可以一起帶。`packaging/` 只影響 DMG／Mac app 的建置，brew venv 不使用它。

一定要整套重裝：

- 改到 `pyproject.toml`（依賴、extras、版本 pin、build-system）、`setup.py`、`Formula/omlx.rb`。
- 要升級或新增依賴、有需要編譯的東西（custom kernel），或這份安裝帶 `--with-custom-kernel`：重裝純 Python wheel 會把編譯好的 kernel 蓋掉。
- 快速路徑失敗、`/health` 回不到 `healthy`，或模型載入失敗。

判斷依賴有沒有變，是跟「venv 依賴基準」比：brew 整套安裝時的 commit（`INSTALL_RECEIPT.json` 的 `source.scm_revision`），不是上一次快速部署的 commit。腳本會自動比對，上面列的檔案有變就拒絕。

先 commit 並 push 到 `origin/main`，再執行：

```bash
scripts/deploy_fast.sh              # 部署 origin/main
scripts/deploy_fast.sh <commit>     # 部署指定 commit（回滾也用這個）
scripts/deploy_fast.sh --status     # 只看目前部署的 commit
```

腳本做的事，也就是手動操作的等價指令：

```bash
cd /Users/jianruicheng/GitHub/omlx
git fetch origin
SHA=$(git rev-parse origin/main)                    # 或要部署的 commit；必須在 origin/main 上
PIP=$(readlink -f /opt/homebrew/opt/omlx)/libexec/bin/pip
BASE=$(jq -r .source.scm_revision /opt/homebrew/opt/omlx/INSTALL_RECEIPT.json)
git diff --name-only "$BASE" "$SHA" -- pyproject.toml setup.py Formula             # 有輸出就改走整套重裝
"$PIP" wheel --no-deps -w "$(mktemp -d)" "git+file://$PWD@$SHA"                   # 預先建置，不動正式 venv
# 確認沒有進行中的請求（見下面），再換檔案重啟
brew services stop jason5545/omlx/omlx
"$PIP" install --no-deps --force-reinstall "git+file://$PWD@$SHA"
brew services start jason5545/omlx/omlx
curl -sS http://127.0.0.1:8000/health                                              # 等到 "status":"healthy"
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

如果衝突落在本地保留的功能，特別檢查：

- `omlx/server.py`（sub-key policy hooks、`DEFAULT_SUB_KEY_POLICIES`）
- `omlx/admin/auth.py`、`omlx/api/openai_models.py`、`omlx/settings.py`
- `omlx/engine_pool.py`（single-model residency；不要恢復成可同時 resident 多個 model 的 admission 行為）
- `apps/omlx-mac/Sources/Server/ServerProcess.swift`（attach mode）
- `Formula/omlx.rb`（homepage/head 要維持 jason5545）
- `packaging/build.py`（`_ssl_context`／`_urlopen`；不要退回裸 `urllib.request.urlretrieve`）
- `omlx/patches/prism_jangq_compat.py`、`omlx/utils/model_loading.py`（`maybe_load_jangq_prism`、`maybe_load_jang`）、`omlx/engine/vlm.py` 與 `omlx/engine/batched.py` 的載入插入點（兩個 JANG 轉接；插入點在 custom quantization 之前，prism 要先於 JANG，順序不要顛倒）
- `omlx/model_discovery.py`（`JANG_CONFIG_FILES`／`_jang_has_vision`；JANG 包的 modality 判定）
- `pyproject.toml`（`jang` extra 留在 optional-dependencies，不要搬進 `dependencies` 或 `bundle`——它晚於 packaging/venvstacks.toml 的 exclude-newer cutoff，搬進去 DMG layer 解析不到）

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
- 回覆 Jason 時用自然、簡短的繁體中文，少模板感。
