"""Trading strategies implementations."""

from .allocation import (
    all_in_allocation,
    cash_partition_allocation,
    create_cash_partition_allocator,
    create_momentum_allocator,
    create_target_weight_allocator,
    equal_weight_allocation,
)
from .components import (
    AllInAllocation,
    BaseAllocation,
    BaseCondition,
    BaseIndicator,
    BasePricing,
    CurrentClosePricing,
    EMAIndicator,
    EqualWeightAllocation,
    MomentumIndicator,
    NoOpenPositionCondition,
    PriceAboveSMACondition,
    RSIIndicator,
    SMAIndicator,
    VolatilityBreakoutAllocation,
    VolatilityBreakoutCondition,
    VolatilityBreakoutPricing,
    create_allocation,
    create_condition,
    create_indicator,
    create_pricing,
)
from .implementations import (
    BaseStrategy,
    BuyAndHoldStrategy,
    MomentumStrategy,
    StrategyFactory,
    VolatilityBreakoutStrategy,
)

__all__ = [
    # Allocation functions
    "all_in_allocation",
    "equal_weight_allocation",
    "cash_partition_allocation",
    "create_cash_partition_allocator",
    "create_target_weight_allocator",
    "create_momentum_allocator",
    # Components - Allocations
    "BaseAllocation",
    "AllInAllocation",
    "EqualWeightAllocation",
    "VolatilityBreakoutAllocation",
    # Components - Conditions
    "BaseCondition",
    "NoOpenPositionCondition",
    "PriceAboveSMACondition",
    "VolatilityBreakoutCondition",
    # Components - Pricing
    "BasePricing",
    "CurrentClosePricing",
    "VolatilityBreakoutPricing",
    # Components - Indicators
    "BaseIndicator",
    "SMAIndicator",
    "EMAIndicator",
    "RSIIndicator",
    "MomentumIndicator",
    # Component factories
    "create_allocation",
    "create_condition",
    "create_pricing",
    "create_indicator",
    # Strategy implementations
    "BaseStrategy",
    "VolatilityBreakoutStrategy",
    "MomentumStrategy",
    "BuyAndHoldStrategy",
    "StrategyFactory",
]
