# Core interfaces for the backtesting framework

from .core import (
    BacktestEngine,
    BacktestError,
    Configuration,
    ConfigurationError,
    DataError,
    DataProvider,
    InsufficientDataError,
    PerformanceMetrics,
    Plugin,
    Portfolio,
    Strategy,
    ValidationError,
)
from .protocols import (
    IBacktestEngine,
    IChartGenerator,
    IDataProvider,
    ILogger,
    IMetricsGenerator,
    IPortfolio,
)
from .strategy_types import (
    AllocationFunc,
    ConditionDict,
    ConditionFunc,
    IAllocation,
    ICondition,
    IPricing,
    IStrategy,
    IStrategyComponent,
    PriceFunc,
    StrategyConfig,
)

__all__ = [
    # Core abstract classes
    "DataProvider",
    "Portfolio",
    "BacktestEngine",
    "Strategy",
    "PerformanceMetrics",
    "Configuration",
    "Plugin",
    "BacktestError",
    "DataError",
    "ConfigurationError",
    "InsufficientDataError",
    "ValidationError",
    # Protocol interfaces
    "IDataProvider",
    "IPortfolio",
    "IBacktestEngine",
    "IStrategy",
    "IStrategyComponent",
    "ICondition",
    "IAllocation",
    "IPricing",
    "ILogger",
    "IMetricsGenerator",
    "IChartGenerator",
    # Type aliases
    "ConditionFunc",
    "PriceFunc",
    "AllocationFunc",
    "ConditionDict",
    "StrategyConfig",
]
