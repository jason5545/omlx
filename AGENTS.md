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

這個 fork 不是乾淨 upstream。它包含幾個 Jason 目前需要的本地功能：

- VLM + Native MTP：VLM 模型啟用 `mtp_enabled=true` 時仍走 `VLMBatchedEngine`，不能退回 LM-only dispatch。
- `VLMModelAdapter` 要保留 `mtp_forward`、`make_mtp_cache`、`return_hidden=True` passthrough。
- mlx-vlm Qwen3.5/Qwen3.6 runtime 有 Native MTP hook。
- 多 token verify 時要保留正確的 mRoPE / position_ids / rope_deltas 行為。
- 只維持一個非 pinned 模型常駐；切換 request model 時會 unload 前一個模型。
- API sub-key 可以套 request policy。`voco` 預設是 `max_context_window<=16384` 且 `enable_thinking=false`。
- `Formula/omlx.rb` 的 HEAD 指向 `https://github.com/jason5545/omlx.git`，並保留 xgrammar macOS arm64 post-install patch。

不要用「關掉 MTP」當 workaround。不要留下手動改 site-packages 的最終狀態。

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

如果衝突落在 VLM/MTP 相關檔案，特別檢查：

- `omlx/engine_pool.py`
- `omlx/engine/vlm.py`
- `omlx/models/vlm.py`
- `omlx/patches/mlx_lm_mtp/batch_generator.py`
- `omlx/patches/mlx_vlm_mtp/`
- `omlx/server.py`
- `omlx/settings.py`

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
  omlx/engine/vlm.py \
  omlx/engine_pool.py \
  omlx/models/vlm.py \
  omlx/patches/gated_delta_advance.py \
  omlx/patches/mlx_lm_mtp/batch_generator.py \
  omlx/patches/mlx_vlm_mtp/__init__.py \
  omlx/patches/mlx_vlm_mtp/qwen35_vlm_runtime.py \
  omlx/server.py \
  omlx/settings.py
```

Homebrew venv 通常沒有 `pytest`。如果沒有安裝，不要說已經跑過 pytest；改說 pytest 不在 venv。

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

VLM + MTP 兼容模型的 log 不應再出現：

```text
forcing LM-only dispatch, vision components ignored
```

## 操作習慣

- 不要把 API key 印到 log 或回覆裡。
- 不要把 `jundot/omlx` tap 裝回來，除非 Jason 明確要求。
- 不要把 Homebrew cache 裡的 checkout 當主要 repo 修改。
- 做完實質變更後，commit 並 push 到 `origin/main`，再視需要重裝 tap。
- 回覆 Jason 時用自然、簡短的繁體中文，少模板感。
