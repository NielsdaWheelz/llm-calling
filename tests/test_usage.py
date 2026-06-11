import pytest

from provider_runtime import Pricing, TokenUsage
from provider_runtime.usage import estimate_cost


def test_estimate_cost_uses_known_components_and_excludes_cached_input_from_full_price() -> None:
    cost = estimate_cost(
        TokenUsage(
            input_tokens=1000,
            output_tokens=2000,
            total_tokens=3000,
            reasoning_tokens=100,
            cached_tokens=400,
        ),
        Pricing(
            input_per_million=1.0,
            output_per_million=2.0,
            cached_input_per_million=0.25,
            reasoning_per_million=3.0,
        ),
    )

    assert cost.input_cost == 0.0006
    assert cost.output_cost == 0.004
    assert cost.cached_input_cost == 0.0001
    assert cost.reasoning_cost == pytest.approx(0.0003)
    assert cost.total_cost == pytest.approx(0.005)


def test_estimate_cost_does_not_synthesize_zero_for_unknown_prices() -> None:
    cost = estimate_cost(
        TokenUsage(input_tokens=1000, output_tokens=1000, total_tokens=2000),
        Pricing(),
    )

    assert cost.input_cost is None
    assert cost.output_cost is None
    assert cost.total_cost is None
