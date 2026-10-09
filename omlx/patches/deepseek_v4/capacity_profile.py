# SPDX-License-Identifier: Apache-2.0
"""Per-layer expert cache capacity for the glm5_next / DeepSeek V4 offload.

The offload gives every layer the same number of slots. Layers do not miss
alike: replaying route traces of GLM-5.3 oQ3.5e at 54% residency, the first
decoder layers keep missing until about 230 slots while the middle ones barely
gain past 120, so the same expert bytes moved between layers read 9-14% fewer
bytes in decode. A *profile* holds, per layer, the decode misses per forward
at a grid of capacities, replayed from route traces (route_trace.py) through a
copy of the live cache's rules; at load, :func:`plan_capacities` spends the
uniform split's expert bytes on the layers where a slot saves the most reads.

Profiles live in ``~/.omlx/moe_offload_profiles/<model directory name>.json``;
``OMLX_MOE_OFFLOAD_CAPACITY_PROFILE`` names another file, or ``0`` turns them
off. Build one from traces of the model::

    python -m omlx.patches.deepseek_v4.capacity_profile TRACE.bin ... --out PATH

No profile, or one that does not cover exactly the wrapped layers, keeps the
uniform split.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FORMAT = 1
_ROUTES, _STATE = 0, 2
_DECODE_MAX_ROWS = 8


def profile_path(model_dir) -> Path | None:
    """The profile to use for ``model_dir``, or ``None`` when turned off."""
    raw = os.environ.get("OMLX_MOE_OFFLOAD_CAPACITY_PROFILE")
    if raw == "0":
        return None
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".omlx" / "moe_offload_profiles" / f"{Path(model_dir).name}.json"


def load_profile(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    try:
        profile = json.loads(path.read_text())
        if profile.get("format") != FORMAT:
            raise ValueError(f"format {profile.get('format')!r}")
        grid = [int(c) for c in profile["capacities"]]
        layers = {
            int(layer): [float(v) for v in misses]
            for layer, misses in profile["layers"].items()
        }
        if grid != sorted(set(grid)) or any(len(v) != len(grid) for v in layers.values()):
            raise ValueError("misses do not match the capacity grid")
        return {"n_experts": int(profile["n_experts"]), "capacities": grid, "layers": layers}
    except Exception as exc:
        logger.warning("moe offload capacity profile %s ignored: %s", path, exc)
        return None


def plan_capacities(profile, expert_bytes: dict, uniform: dict, n_experts: int,
                    minimum: int) -> dict | None:
    """Per-layer capacities spending at most the uniform split's bytes.

    ``expert_bytes`` and ``uniform`` map each wrapped layer to its bytes per
    expert and its uniform capacity. Every layer starts at the smallest grid
    capacity it may hold, then the next grid step goes to the layer whose
    step saves the most misses per slot (bytes read per byte of memory) while
    the total stays within budget. ``None`` keeps the uniform split: the
    profile covers other layers or another expert count, or its smallest
    capacities already overrun the budget.
    """
    if profile is None or profile["n_experts"] != n_experts:
        return None
    if set(profile["layers"]) != set(expert_bytes):
        return None
    grid = [c for c in profile["capacities"] if minimum <= c <= n_experts]
    if len(grid) < 2:
        return None
    cols = [profile["capacities"].index(c) for c in grid]
    misses = {L: [profile["layers"][L][i] for i in cols] for L in expert_bytes}
    budget = sum(uniform[L] * expert_bytes[L] for L in expert_bytes)
    step = {L: 0 for L in expert_bytes}
    used = sum(grid[0] * expert_bytes[L] for L in expert_bytes)
    if used > budget:
        return None
    while True:
        best = None
        for L in sorted(expert_bytes):
            i = step[L]
            if i + 1 >= len(grid):
                continue
            cost = (grid[i + 1] - grid[i]) * expert_bytes[L]
            if used + cost > budget:
                continue
            gain = (misses[L][i] - misses[L][i + 1]) / (grid[i + 1] - grid[i])
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, L, cost)
        if best is None:
            break
        _, L, cost = best
        step[L] += 1
        used += cost
    return {L: grid[step[L]] for L in expert_bytes}


# ---------------------------------------------------------------------------
# building a profile from route traces


class _Cache:
    """The live slot cache's bookkeeping without the bytes: decayed route
    counts (+1 per route, x``decay`` every ``every`` calls), a miss takes a
    free slot (LIFO) or evicts the lowest count outside the call; an
    over-capacity call streams (free slots take its most-routed new experts,
    nothing is evicted)."""

    def __init__(self, n, capacity, decay, every, state):
        score = np.asarray(state["score"], dtype=np.float32)
        slots = [int(e) for e in state["slots"]]
        if capacity > len(slots):
            slots += [-1] * (capacity - len(slots))
        elif capacity < len(slots):  # keep the most counted residents
            res = sorted((e for e in slots if e >= 0), key=lambda e: -score[e])[:capacity]
            slots = res + [-1] * (capacity - len(res))
        self.n, self.capacity, self.decay, self.every = n, capacity, decay, every
        self.slot_expert = np.array(slots, dtype=np.int64)
        self.slot_of = {int(e): s for s, e in enumerate(self.slot_expert) if e >= 0}
        self.free = [s for s in range(capacity) if self.slot_expert[s] < 0]
        self.score = score.copy()
        self.calls = int(state["calls"])

    def _count(self, ids):
        np.add.at(self.score, np.asarray(ids, dtype=np.int64), np.float32(1.0))
        self.calls += 1
        if self.calls % self.every == 0:
            self.score *= np.float32(self.decay)

    def _install(self, ids) -> int:
        needed = list(dict.fromkeys(int(e) for e in ids))
        self._count(ids)
        misses = [e for e in needed if e not in self.slot_of]
        protected = frozenset(needed)
        for e in misses:
            if self.free:
                slot = self.free.pop()
            else:
                scores = self.score[self.slot_expert]
                for p in protected:
                    s = self.slot_of.get(p)
                    if s is not None:
                        scores[s] = np.inf
                slot = int(np.argmin(scores))
                del self.slot_of[int(self.slot_expert[slot])]
            self.slot_of[e] = slot
            self.slot_expert[slot] = e
        return len(misses)

    def call(self, rec) -> int:
        """Misses of one recorded call at this capacity: a call whose distinct
        experts fit installs them (decode, verify, in-capacity prefill),
        otherwise it streams — the wrapper's choice, made per capacity."""
        ids = np.asarray(rec["ids"]).reshape(-1)
        if len(set(ids.tolist())) <= self.capacity:
            if len(self.slot_of) == self.n:
                return 0
            return self._install(ids.tolist())
        used, counts = np.unique(ids.astype(np.int64), return_counts=True)
        fill, misses = [], 0
        if self.free:
            new = used[[int(e) not in self.slot_of for e in used]]
            if len(new):
                top = np.argsort(-counts[np.searchsorted(used, new)], kind="stable")
                fill = [int(e) for e in new[top[: len(self.free)]]]
                misses += self._install(fill)
        filled = set(fill)
        misses += sum(1 for e in used if int(e) not in self.slot_of)
        self._count([int(e) for e in used if int(e) not in filled])
        return misses


def replay_decode_misses(records, capacities, decay, every, n_experts=None):
    """``{layer: [decode misses per decode forward at each capacity]}`` of one
    trace's records (``route_trace.read``), each layer replayed from its first
    recorded state."""
    by_layer, states = collections.defaultdict(list), {}
    for r in records:
        if r["kind"] == _STATE:
            states.setdefault(r["layer"], r)
        elif r["layer"] in states:
            by_layer[r["layer"]].append(r)
    if not states:
        return {}, 0
    n = n_experts or len(next(iter(states.values()))["score"])
    first = min(states)
    forwards = sum(
        1 for r in by_layer[first] if r["kind"] == _ROUTES and r["rows"] <= _DECODE_MAX_ROWS
    )
    out = {}
    for L, st in states.items():
        row = []
        for C in capacities:
            cache = _Cache(n, C, decay, every, st)
            m = 0
            for r in by_layer[L]:
                k = cache.call(r)
                if r["kind"] == _ROUTES and r["rows"] <= _DECODE_MAX_ROWS:
                    m += k
            row.append(m / max(forwards, 1))
        out[L] = row
    return out, forwards


def build_profile(paths, capacities, decay, every) -> dict:
    from .route_trace import read

    total = None
    sources = []
    n_experts = None
    for p in paths:
        _, records = read(p)
        states = [r for r in records if r["kind"] == _STATE]
        if not states:
            continue
        n = len(states[0]["score"])
        if n_experts not in (None, n):
            raise ValueError(f"{p}: {n} experts, earlier traces {n_experts}")
        n_experts = n
        misses, forwards = replay_decode_misses(records, capacities, decay, every, n)
        if not forwards:
            continue
        sources.append({"trace": Path(p).name, "decode_forwards": forwards})
        if total is None:
            total = {L: np.array(v) for L, v in misses.items()}
        else:
            if set(total) != set(misses):
                raise ValueError(f"{p}: layers differ from the earlier traces")
            for L, v in misses.items():
                total[L] += np.array(v)
    if total is None:
        raise ValueError("no decode forwards in the traces")
    k = len(sources)
    return {
        "format": FORMAT,
        "n_experts": n_experts,
        "decay": decay,
        "decay_every": every,
        "capacities": list(capacities),
        "sources": sources,
        "layers": {str(L): [round(x / k, 4) for x in v] for L, v in sorted(total.items())},
    }


def main(argv=None) -> int:
    from . import moe_offload

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="96:288:8", help="min:max:step capacities")
    a = ap.parse_args(argv)
    lo, hi, st = (int(x) for x in a.grid.split(":"))
    grid = list(range(lo, hi + 1, st))
    profile = build_profile(
        a.traces, grid, moe_offload._SLOT_SCORE_DECAY, moe_offload._SCORE_DECAY_EVERY
    )
    out = Path(a.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=1))
    print(f"wrote {out}: {len(profile['layers'])} layers, {len(grid)} capacities, "
          f"{sum(s['decode_forwards'] for s in profile['sources'])} decode forwards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
