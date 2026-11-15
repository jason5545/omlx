# 生圖 API（`/v1/images/*`）

Jason fork 的本地 patch（upstream 不做生圖，見 jundot/omlx #477、#1221）。端點、欄位與回應格式跟 mlx-vlm 自己的 server 相同（`mlx_vlm/server/schemas.py`），oMLX 只多了串流、data URL 和幾條限制。給 Easel 的約定在 `~/GitHub/easel/docs/omlx-images-api.md`。

## 哪些模型算生圖模型

由 mlx-vlm 判斷：`mlx_vlm.generate.image.is_image_generation_model`、`mlx_vlm.generate.edit_image.is_image_edit_model`。oMLX 不維護 model_type 清單。

oMLX 只做一道前置過濾：資料夾裡沒有 `model_index.json`、`manifest.json`、`transformer/*.index.json`，`config.json` 也沒有 `_class_name` 或 `dit`，就不呼叫 mlx-vlm 的判斷。原因是 mlx-vlm 的判斷會對每個候選型別 import `mlx_vlm.models.<type>.model`，對一般 LLM 資料夾做這件事，會在 oMLX 的載入 patch 之前把模型程式（甚至 oMLX 自己 vendor 的套件的 stock 版）先 import 進來。過了前置過濾，還要候選型別在 mlx-vlm 裡有 `models/<type>/model.py`（只有生圖 pipeline 有這個檔），才交給 mlx-vlm。

判成生圖模型的資料夾：

- `/v1/models` 的 `model_type` 是 `"image_generation"`，`max_model_len` 是 `null`。
- 元件子資料夾（`transformer/`、`text_encoder/`、`vae/`）不會再被當成獨立的 LLM 註冊。
- 下載沒完成（HF 的 `*.incomplete` 還在、元件缺 shard、元件只有 index 沒有權重）就先不列出，log 寫 `Skipping incomplete image model`。

## 記憶體與並行

生圖模型是 EnginePool 的一種引擎（`engine_type: image_generation`，`omlx/engine/image_generation.py`），跟其他模型一樣受 single-model residency 管：

- 記憶體估算（`omlx/image_generation.py` 的 `estimate_image_memory`）：認得的家族用元件大小推「一次請求的峰值」，其他家族照舊是整個 checkpoint 的 safetensors 加 5%。載入前照既有規則先卸載其他閒置模型。
- MiniMax-H3-Image 的規則：mlx-vlm 的 pipeline 啟動時什麼都不載（lazy）。每次請求先載 text encoder、編碼完就丟掉（`evict_text_encoder` 預設 true，所以每次都重載）；transformer 和 VAE 第一次用到才載，之後常駐；第一次請求建完 AdaLN cache 後會丟掉 AdaLN 投影權重（8-bit 約 26 GB）。所以第二次請求起的峰值是「transformer − AdaLN 權重 + VAE + text encoder」，再加 5%。AdaLN 權重的大小從 transformer shard 的 safetensors header 讀（只讀 header，不讀權重）。MiniMax-H3-Image-8bit：(47.48 − 26.08 + 2.34 + 27.49) GB × 1.05 = **53.79 GB**（pool log 用 GiB，印成 `50.09GB`）。讀不到 header、找不到 AdaLN 權重或元件不在本機時，退回檔案大小。
- 11/15 的實測對得上這個規則：第一次 2048² 的 peak 26.9 GB（只有 text encoder 約 26.7 GB），第二次 49.9 GB（常駐 transformer 21.4 + VAE decoder 1.6 + text encoder 26.7），4096² edit 51.4 GB（多了 VAE encoder 和視覺塔）。舊估算 75.60 GiB 是三個元件同時常駐的量，實際上不會發生。
- 每張圖之後，引擎量自己的峰值：請求開始前的常駐量，加上這次請求在其他模型之上多用的最大量（`mx.get_peak_memory()` 減請求開始時的 `mx.get_active_memory()`；所有 MLX 工作都在同一條 MLX 執行緒上，請求期間其他模型不會配置記憶體）。log 是 `Image generate done: ... peak=49.9GB (own 49.9GB, resident 23.0GB)`，`peak` 是整個行程的數字，`own` 是這個模型自己的。失敗的請求也回報峰值。
- pool 的 charge 是「估算」和「量過的最大峰值 × 1.05」取大，只升不降：量到更大的峰值時當場調高 charge 和 pool 總量（log：`Image model ... peaked at ...; charge raised ... -> ...`），卸載重載、重新掃描模型後也保留。不認得的家族從檔案大小起跳，所以不會低估。
- 卸載時的 settle 期望值改用引擎實測的常駐量（H3-8bit 生過圖後約 23 GB），不再用 charge，所以不會每次 `stalled`。沒生過圖就卸載時常駐量是 0，直接 settled。
- 每個請求對引擎取 lease，生圖期間不會被淘汰。這時別的模型的請求會拿到 **409**（`ModelBusyError`：`Model '...' is busy; cannot ...`），不排隊等。生圖結束、lease 放掉後才能切換。
- 同一個生圖模型的多個請求依序執行：所有 MLX 運算都在全域 MLX 執行緒上（避免 Metal command buffer 競爭，見 `engine_core.get_mlx_executor`），後到的請求排在前一個後面。
- event loop 不會被卡住：生圖在 MLX 執行緒上跑，`/health`、`/v1/models`、admin 照常回應。
- client 斷線時，已經在跑的那張會跑完（mlx-vlm 沒有中斷點），`n` 裡還沒開始的就不跑；lease 等 MLX 工作結束才放掉，模型不會在運算中途被卸載。

## Prompt 擴寫（PE）

mlx-vlm 用兩個欄位控制：`upsample_prompt`（`null` 表示照模型預設）和 `prompt_expansion_model`（PE 模型的路徑或 HF repo）。MiniMax-H3-Image 和 Flux3 預設會擴寫（`upsample_prompt` 預設 true，不分 prompt 長短），PE 模型分別是 `MiniMaxAI/MiniMax-H3-Image-PE`（約 18 GB）和 `black-forest-labs/flux-3-image-klein-pup`，repo 名稱在執行時從 mlx-vlm 讀（`<family>.upsampler.DEFAULT_UPSAMPLER`）。mlx-vlm 第一次用會自動下載。

oMLX 在請求處理中不下載任何東西，所以先在本機找 PE 模型：路徑本身、oMLX 模型資料夾的 `<org>/<name>` 或 `<name>`、HF cache（`local_files_only`）。找到就把本機路徑交給 mlx-vlm。

| 請求 | PE 在本機 | PE 不在本機 |
|---|---|---|
| `upsample_prompt` 沒給 | 擴寫（mlx-vlm 預設） | **不擴寫**，直接用原 prompt 出圖（log 一行說明） |
| `upsample_prompt: true` | 擴寫 | **400**，訊息寫出要下載的 HF repo |
| `upsample_prompt: false` | 不擴寫 | 不擴寫 |
| 給了 `prompt_expansion_model` | 用它 | **400**（除非同時 `upsample_prompt: false`） |

有擴寫時，回應的 `revised_prompt` 是擴寫後送進模型的 prompt；沒擴寫是 `null`。

要讓 H3 預設擴寫，先下載 PE：`hf download MiniMaxAI/MiniMax-H3-Image-PE --local-dir ~/.lmstudio/models/MiniMaxAI/MiniMax-H3-Image-PE`。

## 串流

`"stream": true` 回 `text/event-stream`：

```
data: {"type": "started", "model": "MiniMax-H3-Image-8bit", "n": 1}

: keepalive

data: {"type": "result", "created": 1794700000, "output_format": "png", "size": "2048x2048", "data": [{"b64_json": "...", ...}]}

data: [DONE]
```

- **不送 `progress`**：mlx-vlm 的生圖 pipeline 在內部跑完整個去噪迴圈，沒有每一步的回呼。H3-Image 走的是 `mlx_vlm/models/minimax_h3_image/`：`generate_array` 固定跑 6 步、沒有 callback 參數，`model._run` 也會拒絕不認得的 `extra`。有 `progress_callback` 的是影音版 `mlx_vlm/models/minimax_h3/`，生圖用不到。所以只有 `started` 和 `result`；要送 `progress` 得先改 mlx-vlm。
- 生圖期間每 15 秒送一行 SSE 註解 `: keepalive`，讓連線不會因為閒置被斷。SSE 規範要求 client 忽略 `:` 開頭的行。
- 失敗時送 `data: {"type": "error", "message": "..."}` 再送 `[DONE]`。
- 請求本身不合法（模型不存在、PE 缺檔、模型忙）時不開串流，直接回一般的 HTTP 錯誤。

## 和 mlx-vlm server 的差異

- 只回 `b64_json`。`response_format: "path"`、`output_path`、`output_dir` 會回 400：這幾個欄位讓 mlx-vlm 把檔案寫到 server 上任意路徑，sub-key 也能用，所以不開。
- `/v1/images/edits` 的 `image` 除了本機絕對路徑，也收 `data:image/...;base64,...`。server 解成暫存檔（`omlx-image-edit-*`），請求結束就刪。相對路徑回 400。
- `model` 是 `/v1/models` 的 id（或 alias），不收任意路徑或 HF repo，所以載入模型也不會連網。
- 模型自己的參數檢查錯誤（例如 H3 只接受 `steps: 6`、`guidance: 1`）回 400，其他失敗回 500。

## 認證與 sub-key policy

三個端點都走 `verify_inference_api_key`（跟 chat、audio 一樣）。

sub-key policy 一條都不套用：`max_context_window`、`enable_thinking`、`thinking_budget_tokens` 管的是 LLM 的對話長度和思考，生圖請求沒有對話 context，也沒有思考可以關。H3 的 PE 雖然是一個會思考的 Qwen3.5，但它的思考模式和取樣參數是 MiniMax 發布的固定設定，不是使用者的對話，關掉只會讓擴寫品質變差。帶 policy 的 sub-key 打生圖時，log 會記一行 `Request policy not applied to image generation`。

## 端點與範例

先取得 API key（不要印出來）：

```bash
KEY=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.omlx/settings.json")))["auth"]["api_key"])')
```

### GET /v1/images/defaults

只讀 metadata，不載入權重。回 mlx-vlm 的 `ImageSamplingDefaults`（`steps`、`guidance`）。模型不支援回 501。

```bash
curl -s -H "Authorization: Bearer $KEY" \
  'http://127.0.0.1:8000/v1/images/defaults?model=MiniMax-H3-Image-8bit&task=generate'
# {"steps":6,"guidance":1.0}
```

### POST /v1/images/generations

```bash
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://127.0.0.1:8000/v1/images/generations \
  -d '{"model": "MiniMax-H3-Image-8bit", "prompt": "A red ceramic teapot by a sunny window",
       "size": "2048x2048", "seed": 42, "response_format": "b64_json"}' \
  | python3 -c 'import sys,json,base64;d=json.load(sys.stdin);open("out.png","wb").write(base64.b64decode(d["data"][0]["b64_json"]))'
```

串流：

```bash
curl -sN -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://127.0.0.1:8000/v1/images/generations \
  -d '{"model": "MiniMax-H3-Image-8bit", "prompt": "A red ceramic teapot", "size": "1024x1024", "stream": true}'
```

### POST /v1/images/edits

```bash
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://127.0.0.1:8000/v1/images/edits \
  -d '{"model": "MiniMax-H3-Image-8bit", "prompt": "Make the teapot in <Picture 1> blue",
       "image": ["/Users/me/Pictures/teapot.png"], "seed": 7}'
```

`image` 也可以是 `"data:image/png;base64,iVBOR..."`，或混用兩種，最多看模型（H3 是 16 張）。`size` 可以不給，交給模型決定。

## 相關檔案

- `omlx/image_generation.py`：判定、缺檔檢查、PE 解析
- `omlx/engine/image_generation.py`：`ImageGenerationEngine`
- `omlx/api/image_routes.py`：三個端點、串流、data URL
- `omlx/model_discovery.py`、`omlx/engine_pool.py`、`omlx/server.py`：註冊與路由
- `tests/test_image_generation.py`
