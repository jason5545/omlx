# SPDX-License-Identifier: Apache-2.0

from omlx.patches.mlx_lm_mtp.adaptive import AdaptiveDepthPolicy


def test_default_policy_starts_at_max_depth() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1)

    assert policy.current_depth == 3


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


def test_full_accept_immediately_increases_depth() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=1)

    result = policy.observe(attempted_depth=1, accepted_depths=1)

    assert result["next_depth"] == 2
    assert result["action"] == "increase"


def test_depth_three_requires_one_full_accept() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=2)

    result = policy.observe(attempted_depth=2, accepted_depths=2)

    assert result["next_depth"] == 3
    assert result["action"] == "increase"


def test_repeated_first_token_reject_reaches_min_depth() -> None:
    policy = AdaptiveDepthPolicy(max_depth=3, min_depth=1, start_depth=3)

    first_reject = policy.observe(attempted_depth=3, accepted_depths=0)
    second_reject = policy.observe(attempted_depth=3, accepted_depths=0)
    third_reject = policy.observe(attempted_depth=2, accepted_depths=0)
    fourth_reject = policy.observe(attempted_depth=2, accepted_depths=0)
    min_reject = policy.observe(attempted_depth=1, accepted_depths=0)

    assert first_reject["next_depth"] == 3
    assert first_reject["action"] == "hold"
    assert second_reject["next_depth"] == 2
    assert second_reject["action"] == "decrease"
    assert third_reject["next_depth"] == 2
    assert third_reject["action"] == "hold"
    assert fourth_reject["next_depth"] == 1
    assert fourth_reject["action"] == "decrease"
    assert min_reject["next_depth"] == 1
    assert min_reject["action"] == "hold"
