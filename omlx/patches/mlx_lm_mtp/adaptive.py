# SPDX-License-Identifier: Apache-2.0
"""Adaptive-depth policy for native MTP draft/verify cycles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class AdaptiveDepthPolicy:
    max_depth: int = 3
    min_depth: int = 1
    start_depth: int = 2
    increase_after: int = 1
    deep_increase_after: int = 3
    decrease_after: int = 2
    deep_min_accept_rate: float = 0.70
    window_size: int = 16

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.min_depth < 1:
            raise ValueError("min_depth must be >= 1")
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth must be <= max_depth")
        if self.increase_after < 1:
            raise ValueError("increase_after must be >= 1")
        if self.deep_increase_after < 1:
            raise ValueError("deep_increase_after must be >= 1")
        if self.decrease_after < 1:
            raise ValueError("decrease_after must be >= 1")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.current_depth = min(max(self.start_depth, self.min_depth), self.max_depth)
        self._full_accept_streak = 0
        self._early_reject_streak = 0
        self._recent: Deque[tuple[int, int]] = deque(maxlen=self.window_size)

    def _record(self, attempted_depth: int, accepted_depths: int) -> None:
        self._recent.append((attempted_depth, accepted_depths))

    def _recent_accept_rate(self) -> float:
        attempted = sum(item[0] for item in self._recent)
        if attempted <= 0:
            return 0.0
        accepted = sum(item[1] for item in self._recent)
        return accepted / attempted

    def observe(
        self, *, attempted_depth: int, accepted_depths: int
    ) -> dict[str, int | str]:
        attempted_depth = max(1, min(int(attempted_depth), self.max_depth))
        accepted_depths = max(0, min(int(accepted_depths), attempted_depth))
        previous_depth = self.current_depth
        action = "hold"
        self._record(attempted_depth, accepted_depths)

        if accepted_depths == attempted_depth:
            self._full_accept_streak += 1
            self._early_reject_streak = 0
            required_full_accepts = (
                self.deep_increase_after
                if self.current_depth >= 2
                else self.increase_after
            )
            high_recent_accept = self._recent_accept_rate() >= self.deep_min_accept_rate
            if (
                self._full_accept_streak >= required_full_accepts
                and self.current_depth < self.max_depth
                and (self.current_depth < 2 or high_recent_accept)
            ):
                self.current_depth += 1
                self._full_accept_streak = 0
                action = "increase"
        else:
            self._full_accept_streak = 0
            late_tail_reject = accepted_depths == attempted_depth - 1
            if self.current_depth >= 3 and not late_tail_reject:
                self.current_depth = max(self.min_depth, self.current_depth - 1)
                self._early_reject_streak = 0
                action = "decrease"
            else:
                if late_tail_reject:
                    self._early_reject_streak = 0
                else:
                    self._early_reject_streak += 1

                if (
                    self._early_reject_streak >= self.decrease_after
                    and self.current_depth > self.min_depth
                ):
                    self.current_depth -= 1
                    self._early_reject_streak = 0
                    action = "decrease"

        return {
            "previous_depth": previous_depth,
            "attempted_depth": attempted_depth,
            "accepted_depths": accepted_depths,
            "next_depth": self.current_depth,
            "action": action,
        }
