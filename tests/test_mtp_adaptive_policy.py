# SPDX-License-Identifier: Apache-2.0

from omlx.patches.mlx_lm_mtp.adaptive import AdaptiveDepthPolicy


def test_default_policy_starts_at_depth_two() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1)

    assert policy.current_depth == 2


def test_depth_three_mid_reject_drops_to_depth_two() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=3)

    result = policy.observe(attempted_depth=3, accepted_depths=1)

    assert result["previous_depth"] == 3
    assert result["next_depth"] == 2
    assert result["action"] == "decrease"


def test_depth_three_late_tail_reject_holds_depth() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=3)

    result = policy.observe(attempted_depth=3, accepted_depths=2)

    assert result["next_depth"] == 3
    assert result["action"] == "hold"


def test_full_accept_immediately_increases_depth() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=1)

    result = policy.observe(attempted_depth=1, accepted_depths=1)

    assert result["next_depth"] == 2
    assert result["action"] == "increase"


def test_depth_three_requires_multiple_full_accepts() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=2)

    first = policy.observe(attempted_depth=2, accepted_depths=2)
    second = policy.observe(attempted_depth=2, accepted_depths=2)
    third = policy.observe(attempted_depth=2, accepted_depths=2)

    assert first["next_depth"] == 2
    assert first["action"] == "hold"
    assert second["next_depth"] == 2
    assert second["action"] == "hold"
    assert third["next_depth"] == 3
    assert third["action"] == "increase"


def test_depth_two_needs_repeated_first_token_reject_to_decrease() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=2)

    first_reject = policy.observe(attempted_depth=2, accepted_depths=0)
    second_reject = policy.observe(attempted_depth=2, accepted_depths=0)
    min_reject = policy.observe(attempted_depth=1, accepted_depths=0)

    assert first_reject["next_depth"] == 2
    assert first_reject["action"] == "hold"
    assert second_reject["next_depth"] == 1
    assert second_reject["action"] == "decrease"
    assert min_reject["next_depth"] == 1
    assert min_reject["action"] == "hold"


def test_depth_two_late_tail_reject_holds_depth() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=2)

    result = policy.observe(attempted_depth=2, accepted_depths=1)

    assert result["next_depth"] == 2
    assert result["action"] == "hold"
