"""Strategy building blocks - reusable components.

Consolidated allocation, conditions, pricing, and indicators
from scattered strategy files into organized components.
"""

# Allocations
from bt.strategies.components.allocations import (
    AllInAllocation,
    BaseAllocation,
    EqualWeightAllocation,
    MomentumAllocation,
    VBOPortfolioAllocation,
    VolatilityBreakoutAllocation,
)

# Conditions
from bt.strategies.components.conditions import (
    BaseCondition,
    NoOpenPositionCondition,
    PriceAboveSMACondition,
    VBOPortfolioBuyCondition,
    VBOPortfolioSellCondition,
    VolatilityBreakoutCondition,
)

# Factory functions
from bt.strategies.components.factory import (
    create_allocation,
    create_condition,
    create_indicator,
    create_pricing,
)

# Indicators
from bt.strategies.components.indicators import (
    BaseIndicator,
    EMAIndicator,
    MomentumIndicator,
    RSIIndicator,
    SMAIndicator,
)

# Pricing
from bt.strategies.components.pricing import (
    BasePricing,
    CurrentClosePricing,
    CurrentOpenPricing,
    VBOPortfolioPricing,
    VolatilityBreakoutPricing,
)

# Regime
from bt.strategies.components.regime import (
    RegimeModelLoader,
    VBORegimeBuyCondition,
    VBORegimeSellCondition,
    calculate_regime_features,
    get_regime_model_loader,
    predict_regime,
)

__all__ = [
    # Allocations
    "BaseAllocation",
    "AllInAllocation",
    "EqualWeightAllocation",
    "MomentumAllocation",
    "VolatilityBreakoutAllocation",
    "VBOPortfolioAllocation",
    # Conditions
    "BaseCondition",
    "NoOpenPositionCondition",
    "PriceAboveSMACondition",
    "VolatilityBreakoutCondition",
    "VBOPortfolioBuyCondition",
    "VBOPortfolioSellCondition",
    "VBORegimeBuyCondition",
    "VBORegimeSellCondition",
    # Pricing
    "BasePricing",
    "CurrentClosePricing",
    "CurrentOpenPricing",
    "VolatilityBreakoutPricing",
    "VBOPortfolioPricing",
    # Indicators
    "BaseIndicator",
    "SMAIndicator",
    "EMAIndicator",
    "RSIIndicator",
    "MomentumIndicator",
    # Regime
    "RegimeModelLoader",
    "get_regime_model_loader",
    "calculate_regime_features",
    "predict_regime",
    # Factory
    "create_allocation",
    "create_condition",
    "create_pricing",
    "create_indicator",
]
