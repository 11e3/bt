"""Consolidated strategy implementations.

Unified strategy definitions using the new component system.
Replaces scattered VBO and other strategy files.
"""

from bt.strategies.implementations.factory import StrategyFactory
from bt.strategies.implementations.strategies import (
    BaseStrategy,
    BuyAndHoldStrategy,
    MomentumStrategy,
    VBOPortfolioStrategy,
    VBORegimeStrategy,
    VolatilityBreakoutStrategy,
)

__all__ = [
    # Strategies
    "BaseStrategy",
    "VolatilityBreakoutStrategy",
    "MomentumStrategy",
    "BuyAndHoldStrategy",
    "VBOPortfolioStrategy",
    "VBORegimeStrategy",
    # Factory
    "StrategyFactory",
]
