# SPDX-License-Identifier: Apache-2.0

from omlx.patches.mlx_lm_mtp.adaptive import AdaptiveDepthPolicy


def test_mid_depth_reject_holds_depth() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=3)

    result = policy.observe(attempted_depth=3, accepted_depths=1)

    assert result["previous_depth"] == 3
    assert result["next_depth"] == 3
    assert result["action"] == "hold"


def test_late_tail_reject_holds_depth() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=3)

    result = policy.observe(attempted_depth=3, accepted_depths=2)

    assert result["next_depth"] == 3
    assert result["action"] == "hold"


def test_single_full_accept_restores_depth() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=3)
    policy.observe(attempted_depth=3, accepted_depths=0)
    policy.observe(attempted_depth=3, accepted_depths=0)

    first_full = policy.observe(attempted_depth=2, accepted_depths=2)

    assert first_full["next_depth"] == 3
    assert first_full["action"] == "increase"


def test_repeated_first_token_reject_reaches_min_depth() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=3)

    policy.observe(attempted_depth=3, accepted_depths=0)
    second_reject = policy.observe(attempted_depth=3, accepted_depths=0)
    policy.observe(attempted_depth=2, accepted_depths=0)
    min_reject = policy.observe(attempted_depth=1, accepted_depths=0)

    assert second_reject["next_depth"] == 2
    assert second_reject["action"] == "decrease"
    assert min_reject["next_depth"] == 1
    assert min_reject["action"] == "decrease"
