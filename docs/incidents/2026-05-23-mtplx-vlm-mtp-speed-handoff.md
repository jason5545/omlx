# 2026-05-23 MTPLX VLM MTP speed handoff

This note records the current working state and the remaining questions for the
Qwen3.6 MTPLX VLM native-MTP speed regression. It is written as a handoff for
another reviewer, not as a final root-cause report.

## Reviewer boundary

The next reviewer's role is read-only review and advice only.

They should inspect this note, code, git history, and logs as needed, then give
their analysis, hypotheses, and suggested next steps. They should not modify
files, change settings, reinstall packages, restart services, clear logs, or run
commands that mutate the repo or machine state.

## Current installed state

- Repo: `/Users/jianruicheng/GitHub/omlx`
- Installed Homebrew formula: `jason5545/omlx/omlx`
- Installed version after the latest repair: `HEAD-14b59c0`
- Important commits now on `origin/main`:
  - `1ce1d62 fix: use stock MLX wheel in Homebrew formula`
  - `1211026 fix: restore May 19 MTP adaptive depth`
  - `68a8b00 fix: restore May 19 adaptive fallback policy`
  - `14b59c0 Revert "perf: avoid profiling sync in MTP verify loop"`

The service is healthy on `127.0.0.1:8000`.

## What is fixed

The garbage-output issue is fixed by keeping the Homebrew formula on the stock
MLX wheel.

The bad path was introduced by:

- `3453c23 Patch MLX small-M qmv fast path`

That commit made Homebrew build/patched MLX from source and correlated with
corrupted quantized output. Do not reintroduce that MLX qmv patch while trying
to regain speed.

The current install has normal text output in the direct API test. No garbage
was observed in the latest tests.

## What partially recovered speed

The May 19 fast behavior depended on a more aggressive adaptive MTP policy.
Later code made depth fallback too conservative, so the runtime spent too much
time at D1/D2 even when `mtp_draft_depth=3`.

Two pieces were restored:

- Start adaptive depth at max depth.
- Only penalize very early rejects. At depth 3, accepting one drafted token and
  rejecting the second does not immediately count as an early-depth failure.

Before restoring the May 19 policy, one direct API run on `1ce1d62` looked like:

```text
Chat completion: 140 tokens in 6.66s (21.0 tok/s), prompt: 52
MTP[...] cycles=72 adaptive<=3 draft_accept=67/118
accept_by_depth=[51, 16, 0, 0]
draft_by_depth=[72, 28, 0, 0]
fb_replay=0
```

After restoring the May 19 policy, warmed runs reached:

```text
run=1 tokens=140 total=6.12 tokps=22.9
run=2 tokens=140 total=5.83 tokps=24.0
```

The corresponding best log line:

```text
Chat completion: 140 tokens in 5.83s (24.0 tok/s), prompt: 52
MTP[...] cycles=57 adaptive<=3 draft_accept=83/167 (49.7%)
accept_by_depth=[44, 27, 12, 0]
draft_by_depth=[57, 44, 25, 0]
fb_replay=0 spec_rb_count=30
```

This means the native-MTP path is active, rollback is the native speculative
rollback path, and the runtime is no longer stuck almost entirely at D1/D2.

## What is still not fully recovered

Jason previously saw about `25.2 tok/s` from a direct glue/script path. The
current direct API test reaches about `23-24 tok/s` warmed. That is better than
the broken state, but not fully back to the best observed number.

The user-facing client path is still the important unresolved case. Jason saw
Voco / PI-agent style calls drop to very low single-digit speeds earlier, while
the direct script could be much faster. The latest tests in this note were
direct API calls with the `none` API key, not the full Voco or PI-agent client
workflow.

## Direct API repro used

The API key used was the literal test key `none`; do not print real keys.

Request shape:

```json
{
  "model": "Qwen3.6-27B-MTPLX-Optimized-Quality",
  "messages": [
    {
      "role": "system",
      "content": "You are a concise technical assistant."
    },
    {
      "role": "user",
      "content": "Write a numbered checklist for debugging ML inference speed. Use one short sentence per item and keep going until about 120 output tokens."
    }
  ],
  "max_tokens": 140,
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "seed": 0,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

## Important negative result

I tried removing the per-cycle `mx.eval(logits)` profiling barrier in the MTP
verify loop, behind an env flag. It did not improve real wall-clock throughput.
The measured time only moved from `backbone` to `sample/tdist eval`, and tok/s
did not improve. That experiment was reverted by `14b59c0`.

Do not spend time on that exact change unless there is a new measurement that
shows a different result.

## Current log expectations

Good current behavior:

```text
VLM+MTP enabled and active ...
MTP path activated ... (model has mtp_forward, batch=1)
MTP[...] adaptive<=3 ... fb_replay=0 ... spec_rb_count>0
Chat completion: ... 23-24 tok/s ...
```

Bad lines that should not appear:

```text
forcing LM-only dispatch, vision components ignored
MTP prefill handoff
committed-history
fb_replay>0
```

## Remaining places to investigate

1. Compare the May 19 post-init MTP activation path against the current lazy
   activation path.

   Current code lazily activates MTP in `GenerationBatch.next()` to avoid cache
   corruption when singleton donor batches are later merged into continuous
   batches. The May 19 fast path activated earlier in `GenerationBatch.__init__`.
   There may be room for a safe singleton-only fast path, but it must preserve
   the later batch-reshape reconciliation safety.

2. Reproduce with the actual Voco / PI-agent request shape.

   The direct API path is now okay-ish. The customer-facing pain was client
   calls becoming single-digit tok/s. Need capture the exact request body,
   headers/sub-key policy, streaming mode, prompt length, and whether grammar or
   tool processors are enabled.

3. Check whether client defaults are accidentally disabling the MTPLX sampler
   contract.

   The direct test explicitly used `temperature=0.6`, `top_p=0.95`, `top_k=20`,
   and `enable_thinking=false`. MTPLX speed/acceptance is sensitive to sampler
   shape. If a client sends `top_k=0`, different temperature, thinking enabled,
   grammar constraints, or a much longer prompt, the MTP path can look active
   but fail to deliver the same speed.

4. Keep stock MLX while testing speed.

   Speed experiments should not resurrect the local MLX qmv patch, because that
   path explains the no-garbage regression.

5. Do not use "disable MTP" as the answer.

   It is fine to run no-MTP as a baseline measurement, but the target fix is
   VLM + Native MTP with normal text output and acceptable throughput.
