# SPDX-License-Identifier: Apache-2.0
"""Adaptive-depth policy for native MTP draft/verify cycles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveDepthPolicy:
    max_depth: int = 3
    min_depth: int = 1
    start_depth: int = 3
    increase_after: int = 1
    decrease_after: int = 2

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if self.min_depth < 1:
            raise ValueError("min_depth must be >= 1")
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth must be <= max_depth")
        self.current_depth = min(max(self.start_depth, self.min_depth), self.max_depth)
        self._full_accept_streak = 0
        self._early_reject_streak = 0

    def observe(
        self, *, attempted_depth: int, accepted_depths: int
    ) -> dict[str, int | str]:
        attempted_depth = max(1, min(int(attempted_depth), self.max_depth))
        accepted_depths = max(0, min(int(accepted_depths), attempted_depth))
        previous_depth = self.current_depth
        action = "hold"

        if accepted_depths == attempted_depth:
            self._full_accept_streak += 1
            self._early_reject_streak = 0
            if (
                self._full_accept_streak >= self.increase_after
                and self.current_depth < self.max_depth
            ):
                self.current_depth += 1
                self._full_accept_streak = 0
                action = "increase"
        else:
            self._full_accept_streak = 0
            rejected_at = accepted_depths + 1
            if rejected_at <= max(1, previous_depth // 2):
                self._early_reject_streak += 1
            else:
                self._early_reject_streak = 0

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
