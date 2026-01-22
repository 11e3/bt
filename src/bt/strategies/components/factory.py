"""Factory functions for creating strategy components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bt.strategies.components.allocations import (
    AllInAllocation,
    BaseAllocation,
    EqualWeightAllocation,
    MomentumAllocation,
    VBOPortfolioAllocation,
    VolatilityBreakoutAllocation,
)
from bt.strategies.components.conditions import (
    BaseCondition,
    NoOpenPositionCondition,
    PriceAboveSMACondition,
    VBOPortfolioBuyCondition,
    VBOPortfolioSellCondition,
    VolatilityBreakoutCondition,
)
from bt.strategies.components.indicators import (
    BaseIndicator,
    EMAIndicator,
    MomentumIndicator,
    RSIIndicator,
    SMAIndicator,
)
from bt.strategies.components.pricing import (
    BasePricing,
    CurrentClosePricing,
    CurrentOpenPricing,
    VBOPortfolioPricing,
    VolatilityBreakoutPricing,
)
from bt.strategies.components.regime import (
    VBORegimeBuyCondition,
    VBORegimeSellCondition,
)

if TYPE_CHECKING:
    from bt.interfaces.strategy_types import IAllocation, ICondition, IPricing


def create_allocation(allocation_type: str, **config) -> IAllocation:
    """Factory function for allocation strategies."""

    allocations: dict[str, type[BaseAllocation]] = {
        "all_in": AllInAllocation,
        "equal_weight": EqualWeightAllocation,
        "equal_weight_momentum": MomentumAllocation,
        "volatility_breakout": VolatilityBreakoutAllocation,
        "vbo_portfolio": VBOPortfolioAllocation,
    }

    if allocation_type not in allocations:
        raise ValueError(f"Unknown allocation type: {allocation_type}")

    return allocations[allocation_type](**config)


def create_condition(condition_type: str, **config: Any) -> ICondition:
    """Factory function for condition strategies."""

    conditions: dict[str, type[BaseCondition]] = {
        "no_open_position": NoOpenPositionCondition,
        "price_above_sma": PriceAboveSMACondition,
        "volatility_breakout": VolatilityBreakoutCondition,
        "vbo_portfolio_buy": VBOPortfolioBuyCondition,
        "vbo_portfolio_sell": VBOPortfolioSellCondition,
        "vbo_regime_buy": VBORegimeBuyCondition,
        "vbo_regime_sell": VBORegimeSellCondition,
    }

    if condition_type not in conditions:
        raise ValueError(f"Unknown condition type: {condition_type}")

    return conditions[condition_type](**config)


def create_pricing(pricing_type: str, **config) -> IPricing:
    """Factory function for pricing strategies."""

    pricing_strategies: dict[str, type[BasePricing]] = {
        "current_close": CurrentClosePricing,
        "current_open": CurrentOpenPricing,
        "volatility_breakout": VolatilityBreakoutPricing,
        "vbo_portfolio": VBOPortfolioPricing,
    }

    if pricing_type not in pricing_strategies:
        raise ValueError(f"Unknown pricing type: {pricing_type}")

    return pricing_strategies[pricing_type](**config)


def create_indicator(indicator_type: str, **config) -> BaseIndicator:
    """Factory function for technical indicators."""

    indicators: dict[str, type[BaseIndicator]] = {
        "sma": SMAIndicator,
        "ema": EMAIndicator,
        "rsi": RSIIndicator,
        "momentum": MomentumIndicator,
    }

    if indicator_type not in indicators:
        raise ValueError(f"Unknown indicator type: {indicator_type}")

    return indicators[indicator_type](**config)
