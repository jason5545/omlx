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

The previous client-path collapse into single-digit tok/s is no longer the
active problem. A later Voco sub-key request on the current repaired code hit
the expected speed range:

```text
Request policy active: client=voco source=api-sub-key ...
Chat completion: 49 tokens in 1.92s (25.6 tok/s), prompt: 195
MTP[...] cycles=14 adaptive<=3 draft_accept=34/42 (81.0%)
accept_by_depth=[13, 12, 9, 0]
draft_by_depth=[14, 13, 12, 0]
fb_replay=0 spec_rb_count=4
```

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

The only remaining goal is to get the repaired path consistently back to
`25.2 tok/s` or higher, without garbage output and without disabling native MTP.

Jason previously saw about `25.2 tok/s` from a direct glue/script path. The
current 140-token direct API test reaches about `23-24 tok/s` warmed. A shorter
Voco sub-key request reached `25.6 tok/s`, so the client single-digit issue
should not be treated as the active regression anymore.

The open question is narrower now: why the representative direct/API benchmark
does not reliably stay above `25.2 tok/s`, and whether the remaining gap is
benchmark-shape variance, TTFT amortization, lazy MTP activation cost, or another
small per-cycle cost.

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

Note: the OpenAI chat request schema does not currently expose `top_k` as a
request-level field. In this setup the effective MTPLX sampler contract is
coming from the model artifact's `mtplx_runtime.json`, which currently contains:

```json
{
  "temperature": 0.6,
  "top_k": 20,
  "top_p": 0.95
}
```

So `top_k=0` is a valid general failure mode to guard against, but it is not the
current explanation for this model's latest direct/Voco measurements.

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
Chat completion: ... target >=25.2 tok/s ...
```

Bad lines that should not appear:

```text
forcing LM-only dispatch, vision components ignored
MTP prefill handoff
committed-history
fb_replay>0
```

## Remaining places to investigate

1. Standardize the benchmark shape before chasing code.

   The current evidence mixes a 140-token direct API benchmark at `23-24 tok/s`
   and a shorter Voco sub-key request at `25.6 tok/s`. Use one representative
   prompt length, output length, sampler contract, streaming mode, and warm/cold
   state before declaring the remaining delta real.

2. Compare the May 19 post-init MTP activation path against the current lazy
   activation path.

   Current code lazily activates MTP in `GenerationBatch.next()` to avoid cache
   corruption when singleton donor batches are later merged into continuous
   batches. The May 19 fast path activated earlier in `GenerationBatch.__init__`.
   There may be room for a safe singleton-only fast path, but it must preserve
   the later batch-reshape reconciliation safety.

3. Quantify `_post_init_mtp` cost and how much it affects short benchmarks.

   Lazy activation may mostly affect TTFT, not steady-state tok/s. On a short
   49-140 token completion, that one-time cost can still move the reported
   average. Measure it before changing the activation boundary.

4. Keep checking the MTPLX sampler contract, but do not assume it is currently
   broken.

   This model's `mtplx_runtime.json` already provides `temperature=0.6`,
   `top_p=0.95`, and `top_k=20`, so current direct/Voco measurements should be
   on the sparse acceptance path. A future cleanup may still wire request-level
   `top_k` into OpenAI chat handling, but that is not the main path to recovering
   the last `1-2 tok/s` here.

5. Treat grammar/tools as a guardrail, not the current target.

   If a future slow log lacks `MTP path activated` or says grammar-constrained
   decoding is active, then inspect `tools`, `response_format`, and
   `structured_outputs`. That was a plausible explanation for old client-path
   slowdowns, but the single-digit client issue is already fixed and should not
   drive this pass.

6. Keep stock MLX while testing speed.

   Speed experiments should not resurrect the local MLX qmv patch, because that
   path explains the no-garbage regression.

7. Do not use "disable MTP" as the answer.

   It is fine to run no-MTP as a baseline measurement, but the target fix is
   VLM + Native MTP with normal text output and `25.2 tok/s+` throughput.
