# PATCHES.md

這份是 Jason 的 oMLX fork 相對 upstream（`jundot/omlx`）保留的本地 patch 清單。操作流程（Homebrew、快速部署、Mac app、追 upstream 的步驟）在 [`AGENTS.md`](AGENTS.md)。

維護規則：

- 新增本地 patch：在「保留的本地改動」加一條（為什麼要改、相關檔案、測試、行為底線），在「合併衝突檢查」列出要守的檔案與函式，並更新條數。
- upstream 修了同一件事：確認該條列的測試仍通過或等價改寫，再整段換成 upstream 版，兩處一起拿掉。
- 修的是 upstream bug 時，寫明來源 commit／PR 與對應的上游 issue，方便判斷什麼時候可以拿掉。

## 保留的本地改動

自 0.6.3rc1 merge 起，這個 fork 盡量貼齊 upstream，只保留十二個本地功能；其他一律 follow upstream（舊的 VLM/MTP、MTPLX、thinking-budget patch stack 已整批丟棄，存在 `backup/pre-upstream-merge` 分支僅供查閱，不要回移植）：

- API sub-key 可以套 request policy（`identify_api_key` → `DEFAULT_SUB_KEY_POLICIES`）。`voco` 預設是 `max_context_window<=16384` 且 `enable_thinking=false`。相關檔案：`omlx/server.py`、`omlx/admin/auth.py`、`omlx/api/openai_models.py`、`omlx/settings.py`。
- Mac app attach mode：`8000` 上已有健康 oMLX server 時 app 直接 attach，不顯示 port conflict；細節見下面 Mac app 章節。相關檔案：`apps/omlx-mac/Sources/Server/ServerProcess.swift` 等 6 個 Swift 檔。
- `Formula/omlx.rb` 的 homepage/HEAD 指向 `https://github.com/jason5545/omlx.git`（xgrammar macOS arm64 post-install patch 已被 upstream 吸收，不需再維護本地版）。
- qwen3_5_moe VLM 強制 sanitize（`omlx/engine/vlm.py` 的 `_force_qwen35_moe_sanitize_on_load`）：mlx-vlm 用第一個 glob 到的 shard metadata 判斷 `is_mlx_format`，mixed-metadata checkpoint（如 Ornith-1.5 MXFP8）會跳過 sanitize，per-expert MTP MoE 權重沒堆疊成 switch_mlp，strict load 失敗 → 退回純 LLM、視覺被靜默丟掉。追 upstream 時守住這段。
- EnginePool 只允許一個 resident model：harness 或 API request 從 model A 切到 model B 時，先 unload 其他閒置 engine，再載入 B；如果 A 有 active request 或 lease，回 `ModelBusyError`，不要硬拆進行中的 request。相關檔案：`omlx/engine_pool.py`、`omlx/admin/routes.py`、`tests/test_engine_pool.py`。
- `packaging/build.py` 下載一律走 `_urlopen`／`_ssl_context`，不直接用 `urllib.request.urlretrieve`：python.org 的 macOS 直譯器（build driver 預設用 PATH 上的 `python3`）附的 OpenSSL 沒有預設信任庫，spacy 模型下載會以 `CERTIFICATE_VERIFY_FAILED` 失敗，donor 重建中斷在 `_install_spacy_model`。helper 依序找 `SSL_CERT_FILE`、certifi、`/etc/ssl/cert.pem` 等系統 bundle。相關檔案：`packaging/build.py`。
- JANGQ affine-ternary prism 轉接：dealignai 的 Bonsai-2-27B-*-Ternary-JANG 沿用 PrismML 的 ternary 權重，但把 `model_type` 改成 `qwen3_5`、又加 `storage_bits`，所以走不到 upstream #3782 已支援的 `prism_hadamard_qwen35`；載入時把 config 正規化成 schema-2、重建 modules manifest、拿掉 `storage_bits`、放寬 prism 的量化檢查、補 161 個 zero-centered norm 的 +1.0，全部在一次載入內 patch 並還原。相關檔案：`omlx/patches/prism_jangq_compat.py`、`omlx/utils/model_loading.py` 的 `maybe_load_jangq_prism`、`omlx/engine/vlm.py` 與 `omlx/engine/batched.py` 的呼叫點（batched 端只負責擋純文字載入）、`tests/test_prism_jangq_compat.py`。
- JANG mixed-precision bundle 轉接：逐張量 bit width 記在 sidecar（`jang_config.json` 等），stock mlx-lm／mlx-vlm 讀不到，所以交給 `jang_tools.loader` 載入後再進正常的 BatchedEngine／VLMBatchedEngine——不要改成新增 engine 類別，server 有一批 `isinstance(engine, VLMBatchedEngine)` 的圖片、prefix cache、tool calling 判定會斷。閘門要求 sidecar 的 `format` 是 `jang`／`jjqf`／`mxq`：只有 vMLX sidecar、`format` 未設的 MXFP8 包（如 Ornith-1.5 MXFP8）要留給原路徑，不要搶過來。`omlx/patches/jang_load.py` 另外補 jang runtime 兩個洞：Nemotron-H gate 的後綴比對（上游 PR #364 那段是死碼，從沒解量化過任何 gate）與 6-bit + `--hadamard` 的 sign 寬度（它用 `packed_cols * (32 // bits)`，6-bit 會算成 60）。相關檔案：`omlx/patches/jang_load.py`、`omlx/utils/model_loading.py` 的 `maybe_load_jang`、`omlx/engine/batched.py` 與 `omlx/engine/vlm.py` 的載入插入點、`omlx/model_discovery.py` 的 `JANG_CONFIG_FILES`／`_jang_has_vision`、`omlx/exceptions.py` 的 `JANGDependencyError`／`JANGLoadError`、`tests/test_jang_engine.py`。依賴是 `pyproject.toml` 的 `jang` extra 加 `Formula/omlx.rb` 一行 `system(*pip_install, "jang[mlx]>=2.5.47")`（必須共用 `pip_install` flags，`tests/test_homebrew_formula.py` 會數裸 pip 呼叫）；上游 PR #364 合併後可整批換成 upstream 版。
- GDN MTP verify prework 的 upstream kernel 擴充為同時支援 fp16／bf16：輸入、conv state、conv1d 權重 dtype 必須相同，scale 也用該 dtype；fp16 和既有 bf16 組合路徑要求逐位元一致。追 upstream 時守住。相關檔案：`omlx/patches/qwen35_gdn_prework.py`、`tests/test_qwen35_gdn_prework.py`。
- MTP depth controller 的校準階段（修長 context park/probe 震盪，2026-09-30）：warmup sweep 量完各深度成本後，acceptance EMA（ALPHA=0.08）只有 ≤max_depth 次更新，而 seed 必然來自剛 park 的 controller（p 鎖在 dip 值）——4 個 warmup 投機 cycle 爬不回 16-cycle 的 streak budget，probe 於是鎖死 depth 0、23 cycle 再 park（server.log 的 `finish=parked cycles=23 d0=18`），冷卻還倍增。修法是 sweep 後進入 `CAL_LEN=24` 校準：鎖定當時最佳投機深度、exit gate 與 staleness probe 暫停，EMA 收斂後才裁決；`_maybe_finish_mtp_reentry_probe` 也必須等校準結束才算成功（提早成功會洗掉該倍增的冷卻）。相關檔案：`omlx/patches/mlx_lm_mtp/batch_generator.py`（`CAL_LEN`、`observe()` 校準分支、`_speculation_losing` 的 cal guard、`_best_speculative()`、probe 成功的 cal 條件）、`tests/test_mtp_depth_controller.py`、`tests/test_mlx_lm_mtp_patch.py` 的 `test_calibrating_reentry_probe_is_not_a_win_yet`。追 upstream 時守住；如果 upstream 對同一震盪出了更好的修法（不同的 probe 成功條件、seed 策略或估計器），換 upstream 版前先確認 dipped-seed 測試（`test_probe_with_dipped_seed_recovers_at_long_context`、`test_genuine_regression_still_parks_after_calibration`）仍通過或等價改寫，行為底線是「校準完成前 exit gate 不裁決、真衰退仍會 park」。
- prefill transient tracker 不讓殘留 pool 的樣本抬高估值（修 admission 永久誤擋，2026-09-30）：decode fairness 把 chunk 壓到 256～320 token 時，`_should_clear_after_chunk` 刻意不清 MLX pool，但量測是「chunk 後 active+pool」減「chunk 前 active」，前面每個 chunk 留下的 pool 都被算成這個 chunk 的成本，每 token 從 3 MB 一路爬到 68 MB（EWMA 跟著爬，8 倍離群過濾擋不到；last-delta 又是原值直用），speed priority 再乘 2048×1.3，下一個 14.9k token prompt 被估成 174 GB 直接 400。之後被 throttle 縮小的 chunk 被當 speed partial 丟掉、大 prompt 在 preflight 就被拒，沒有任何樣本能蓋掉，直到模型重載。修法：兩個 prefill loop 把 chunk 開始時的 pool 傳進 `_record_chunk_transient(pre_pool_bytes=)`，超過 `max(64MB, delta×10%)` 就視為上界；上界樣本與 speed partial 一律走 `PrefillTransientTracker.tighten()`，只能把 EWMA 和 last-delta 的每 token 成本往下修、不算 sample。相關檔案：`omlx/scheduler.py`（`_RETAINED_POOL_TOLERANCE`／`_RETAINED_POOL_FLOOR_BYTES`、`_record_chunk_transient`、兩處 `_chunk_memory_sample` 呼叫點）、`omlx/prefill_transient_tracker.py` 的 `tighten`、`tests/test_prefill_oom_graceful.py` 的 `test_contended_chunks_with_retained_pool_cannot_inflate_admission`（用實際 log 數字重播）與 `test_speed_partial_recovers_a_poisoned_estimate`、`tests/test_prefill_transient_tracker.py` 的 `TestTighten`。這是 upstream 的 bug，不是 fork 造成的：不清 pool 來自 #2633（decode fairness），active 基準與 speed priority 收 2048 來自 #3933，丟掉 speed partial 來自 #2434；截至 2026-09-30 upstream/main（807ed7e8）仍未修，上游 issue [jundot/omlx#3997](https://github.com/jundot/omlx/issues/3997)（同架構 Qwen3.8-27B hd256、10.7k token 估 58.5 GB、重載才好）很可能是同一件，尚未證實也無人回覆。追 upstream 時守住；upstream 修了（#3997 關閉，或改了量測基準，例如 chunk 前改用 active+pool、改量 peak memory），先確認上面兩個重播測試仍通過或等價改寫，再整段換 upstream 版並從這份清單拿掉，行為底線是「並行 decode 下的 prefill 不會讓之後的 prompt 永久被 preflight 擋下」。
- Mac app 下載速度文字的字級（修 upstream 編譯錯誤，2026-09-30）：upstream e4c8d76b（#3875，下載即時速度）在 `DownloadsScreen.swift` 用了 `.omlxMono(DesignTokens.FontSize.aux)`，但整個 upstream 沒有定義 `DesignTokens`，`build.sh release` 的 xcodebuild 直接失敗。改成跟同一列進度文字一樣的 `.omlxMono(11)`。`DesignTokens` 定義在同作者尚未合併的 [jundot/omlx#4082](https://github.com/jundot/omlx/pull/4082)（`DesignTokens.swift`，`FontSize.aux = 12`）；#3875 描述說它可以單獨合併，實際上依賴 #4082。相關檔案：`apps/omlx-mac/Sources/AppView/Screens/DownloadsScreen.swift`（`ActiveDownloadsSection` 的 `task.speedText`）。#4082 合併後，upstream 若沒再改這一行，git 會保留我們的 `.omlxMono(11)`，不會自動換回，所以合併時看到 #4082 進來要手動改回 `DesignTokens.FontSize.aux` 並拿掉這條。

追 upstream 時，conflict 只要守住上面幾塊，其餘一律取 upstream 版本。不要留下手動改 site-packages 的最終狀態。

## 合併衝突檢查

合併 upstream 後衝突落在下列檔案時逐項確認：

- `omlx/server.py`（sub-key policy hooks、`DEFAULT_SUB_KEY_POLICIES`）
- `omlx/admin/auth.py`、`omlx/api/openai_models.py`、`omlx/settings.py`
- `omlx/engine_pool.py`（single-model residency；不要恢復成可同時 resident 多個 model 的 admission 行為）
- `apps/omlx-mac/Sources/Server/ServerProcess.swift`（attach mode）
- `Formula/omlx.rb`（homepage/head 要維持 jason5545）
- `packaging/build.py`（`_ssl_context`／`_urlopen`；不要退回裸 `urllib.request.urlretrieve`）
- `omlx/patches/prism_jangq_compat.py`、`omlx/utils/model_loading.py`（`maybe_load_jangq_prism`、`maybe_load_jang`）、`omlx/engine/vlm.py` 與 `omlx/engine/batched.py` 的載入插入點（兩個 JANG 轉接；插入點在 custom quantization 之前，prism 要先於 JANG，順序不要顛倒）
- `omlx/model_discovery.py`（`JANG_CONFIG_FILES`／`_jang_has_vision`；JANG 包的 modality 判定）
- `omlx/patches/mlx_lm_mtp/batch_generator.py`（depth controller 校準階段：`CAL_LEN`、`observe()` 校準分支、`_speculation_losing` cal guard、`_best_speculative()`、`_maybe_finish_mtp_reentry_probe` 的 cal 條件；行為底線見「保留的本地改動」該條）
- `omlx/scheduler.py` 的 `_record_chunk_transient`（`pre_pool_bytes`、`pool_retained`、speed partial 改走 `tighten` 而不是直接 return）、`_RETAINED_POOL_TOLERANCE`／`_RETAINED_POOL_FLOOR_BYTES`、兩個 prefill loop 的 `_pre_total` 取樣與傳參；`omlx/prefill_transient_tracker.py` 的 `tighten`（prefill 估算防汙染；upstream 改這幾段時最容易被整段蓋掉，合併後跑 `tests/test_prefill_oom_graceful.py`、`tests/test_prefill_transient_tracker.py`）
- `apps/omlx-mac/Sources/AppView/Screens/DownloadsScreen.swift`（下載速度文字的字級；upstream 已定義 `DesignTokens`（#4082 合併）就改回 `DesignTokens.FontSize.aux`，否則維持 `.omlxMono(11)`，合併後要能跑過 `build.sh release`）
- `pyproject.toml`（`jang` extra 留在 optional-dependencies，不要搬進 `dependencies` 或 `bundle`——它晚於 packaging/venvstacks.toml 的 exclude-newer cutoff，搬進去 DMG layer 解析不到）
