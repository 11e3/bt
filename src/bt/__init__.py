"""
Compo BT - Cryptocurrency Backtesting Framework

A modern, event-driven backtesting engine for cryptocurrency trading strategies.
"""

__version__ = "0.1.0"

# Core components
# Strategy components
# Reporting and visualization
from bt import reporting, strategies
from bt.core.base import (
    BaseBacktestEngine,
    BaseConfiguration,
    BaseDataProvider,
    BasePerformanceMetrics,
    BasePortfolio,
    BaseStrategy,
)

# Data abstraction layer
from bt.data.storage import DataManager, get_data_manager, retrieve_data, store_data

# Domain models and types
from bt.domain.models import BacktestConfig, PerformanceMetrics, Position, Trade
from bt.domain.types import Amount, Fee, Percentage, Price, Quantity
from bt.engine.backtest import BacktestEngine
from bt.framework import BacktestFramework
from bt.interfaces.core import (
    BacktestEngine as IBacktestEngine,
)

# Exceptions
from bt.interfaces.core import (
    BacktestError,
    ConfigurationError,
    DataError,
    DataProvider,
    InsufficientDataError,
    Portfolio,
    ValidationError,
)
from bt.interfaces.core import (
    Configuration as IConfiguration,
)

# Interfaces and base classes
from bt.interfaces.core import (
    DataProvider as IDataProvider,
)
from bt.interfaces.core import (
    PerformanceMetrics as IPerformanceMetrics,
)
from bt.interfaces.core import (
    Portfolio as IPortfolio,
)
from bt.interfaces.core import (
    Strategy as IStrategy,
)

# Profiling and code quality
from bt.profiling import (
    PerformanceProfiler,
    get_performance_stats,
    profile_context,
    profile_function,
    run_quality_analysis,
)
from bt.reporting import (
    calculate_performance_metrics,
    plot_equity_curve,
    plot_market_regime_analysis,
    plot_wfa_results,
    plot_yearly_returns,
    print_performance_report,
    print_sample_trades,
    save_all_charts,
)

# Security
from bt.security import SecurityManager, scan_security, validate_input
from bt.strategies import (
    BuyAndHoldStrategy,
    MomentumStrategy,
    StrategyFactory,
    VolatilityBreakoutStrategy,
    create_allocation,
    create_condition,
    create_pricing,
)

# Utilities
from bt.utils.logging import get_logger

__all__ = [
    # Core
    "BacktestFramework",
    "BacktestEngine",
    "DataProvider",
    "Portfolio",
    # Domain
    "BacktestConfig",
    "Position",
    "Trade",
    "PerformanceMetrics",
    "Price",
    "Quantity",
    "Amount",
    "Percentage",
    "Fee",
    # Interfaces
    "IDataProvider",
    "IPortfolio",
    "IStrategy",
    "IBacktestEngine",
    "IPerformanceMetrics",
    "IConfiguration",
    # Base classes
    "BaseDataProvider",
    "BasePortfolio",
    "BaseStrategy",
    "BaseBacktestEngine",
    "BasePerformanceMetrics",
    "BaseConfiguration",
    # Strategy components
    "VolatilityBreakoutStrategy",
    "MomentumStrategy",
    "BuyAndHoldStrategy",
    "StrategyFactory",
    "create_allocation",
    "create_condition",
    "create_pricing",
    # Reporting
    "reporting",
    "calculate_performance_metrics",
    "print_performance_report",
    "print_sample_trades",
    "plot_equity_curve",
    "plot_yearly_returns",
    "plot_wfa_results",
    "plot_market_regime_analysis",
    "save_all_charts",
    # Security
    "SecurityManager",
    "validate_input",
    "scan_security",
    # Data abstraction layer
    "DataManager",
    "get_data_manager",
    "store_data",
    "retrieve_data",
    # Profiling and code quality
    "PerformanceProfiler",
    "profile_function",
    "profile_context",
    "run_quality_analysis",
    "get_performance_stats",
    # Utilities
    "get_logger",
    # Exceptions
    "BacktestError",
    "DataError",
    "ConfigurationError",
    "InsufficientDataError",
    "ValidationError",
]
