"""Simplified public API facade for the backtesting framework.

Provides a single entry point for common backtesting operations
with streamlined imports and configuration.
"""

from typing import Any, Optional

from bt.config.settings import BacktestConfig, create_production_config, get_config_manager
from bt.core.container import Container, get_default_container, set_default_container
from bt.core.orchestrator import BacktestOrchestrator
from bt.core.registry import StrategyFactory, get_strategy_registry
from bt.domain.models import PerformanceMetrics
from bt.interfaces.protocols import ILogger
from bt.security import SecurityManager
from bt.utils.logging import get_logger, get_logger_adapter


class BacktestFramework:
    """High-level facade for backtesting operations.

    Provides simplified interface for:
    - Running backtests with any strategy
    - Loading and validating configurations
    - Accessing performance results
    - Managing strategy registry
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        container: Container | None = None,
        logger: Any | None = None,
    ):
        """Initialize backtest framework.

        Args:
            config: Configuration overrides
            container: Custom dependency injection container
            logger: Custom logger instance
        """
        # Set up container with default services
        if container is not None:
            set_default_container(container)

        if logger is not None:
            container.logger = logger

        # Set up configuration manager
        config_manager = get_config_manager()
        if config is not None:
            config_manager.set_config(config)

        self.container = container or get_default_container()
        self.config_manager = config_manager
        # Use provided logger or create default
        self.logger = logger or get_logger_adapter("bt.framework")
        self._config = config or {}

        # Register core services
        try:
            self.container.register_singleton(SecurityManager, SecurityManager)
        except:
            pass  # Ignore registration errors for now

        # Initialize orchestrator with container
        self.orchestrator = BacktestOrchestrator(self.container, config, self.logger)
        self._registry = get_strategy_registry()
        self._factory = StrategyFactory()

        self.logger.info("BacktestFramework initialized")

    def run_backtest(
        self,
        strategy: str,
        symbols: list[str],
        data: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run a backtest with simplified interface.

        Args:
            strategy: Strategy name to use
            symbols: Symbols to trade
            data: Market data (symbol -> DataFrame)
            config: Additional configuration
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with backtest results
        """
        # Get security manager for input validation
        security_manager = self.container.get(SecurityManager)

        # Validate inputs
        symbols = security_manager.validator.validate(symbols, "list")
        data = security_manager.validator.validate(data, "dict")

        # Validate market data
        for symbol, df in data.items():
            security_manager.validator.validate(symbol, "symbol")
            security_manager.validator.validate(df, "dataframe")

        # Merge configurations
        final_config = {**self._config, **(config or {})}

        # Validate configuration if provided
        if config is not None:
            config = security_manager.validator.validate(config, "dict")
            final_config = security_manager.validator.validate(final_config, "dict")

            config_manager = self.config_manager
            errors = config_manager.validate_backtest_parameters(final_config)
            if errors:
                raise ValueError(f"Invalid backtest configuration: {', '.join(errors)}")

        # Get strategy
        try:
            strategy = self._registry.get_strategy(strategy, **final_config)
        except Exception as e:
            self.logger.error(f"Error getting strategy '{strategy}': {e}")
            raise

        # Run backtest
        results = self.orchestrator.run_backtest(strategy, symbols, data)

        # Add framework configuration to results
        results["configuration"] = final_config

        self.logger.info(
            f"Backtest completed for strategy: {strategy}",
            extra={
                "symbols": symbols,
                "trades": len(results.get("trades", [])),
                "total_return": results.get("performance", {}).get("total_return", 0),
            },
        )

        return results

    def list_available_strategies(self, category: str | None = None) -> list[str]:
        """List all available strategies."""
        return self._registry.list_strategies(category)

    def get_strategy_info(self, strategy: str) -> dict[str, Any] | None:
        """Get information about a strategy."""
        return self._registry.get_strategy_info(strategy)

    def create_strategy(self, strategy: str, config: dict[str, Any] | None = None, **kwargs) -> Any:
        """Create strategy instance with configuration."""
        return self._registry.get_strategy(strategy, **(config or {}), **kwargs)

    def validate_strategy_config(self, strategy: str, config: dict[str, Any]) -> list[str]:
        """Validate strategy configuration."""
        return self._registry.validate_strategy_config(strategy, config)

    def load_market_data(
        self, data_directory: str = "data", symbols: list[str] | None = None
    ) -> dict[str, Any]:
        """Load market data for symbols.

        Args:
            data_directory: Directory containing market data files
            symbols: Symbols to load (defaults to config default)

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        symbols = symbols or self.config_manager.get_base_config().default_symbols

        data = {}

        for symbol in symbols:
            try:
                import pandas as pd

                file_path = f"{data_directory}/{symbol.lower()}.parquet"
                df = pd.read_parquet(file_path)
                if df is not None and not df.empty:
                    data[symbol] = df
                    self.logger.info(f"Loaded {len(df)} bars for {symbol}")
                else:
                    self.logger.warning(f"No data found for {symbol}")
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")

        return data

    def run_simple_backtest(
        self, strategy: str, symbols: list[str] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Run simple backtest with default parameters."""

        # Use default configuration
        config = self.config_manager.get_backtest_config()

        return self.run_backtest(strategy, symbols, config=config, **kwargs)

    def run_production_backtest(
        self, strategy: str, symbols: list[str], **kwargs
    ) -> dict[str, Any]:
        """Run production backtest with optimal settings."""

        # Use production configuration
        config = self.config_manager.get_backtest_config()

        return self.run_backtest(strategy, symbols, config=config, **kwargs)

    def create_performance_report(self, results: dict[str, Any]) -> None:
        """Generate and save performance report."""
        from bt.reporting.charts import generate_all_charts

        performance_data = results.get("performance", {})

        # Generate charts
        charts_dir = self.config_manager.get_reporting_config().report_directory
        generate_all_charts(results.get("equity_curve", {}), performance_data, charts_dir)

        self.logger.info(f"Performance report generated in {charts_dir}")

    def get_framework_info(self) -> dict[str, Any]:
        """Get information about the backtesting framework."""
        return {
            "version": "1.0.0",
            "strategies_count": len(self._registry._strategies),
            "categories": self._registry.get_available_categories(),
            "container_type": self.container.__class__.__name__,
            "orchestrator": self.orchestrator.__class__.__name__,
            "configuration": self.config_manager.__class__.__name__,
        }


# Convenience functions for common operations


def quick_backtest(
    strategy: str = "volatility_breakout", symbols: list[str] | None = None, **kwargs
) -> dict[str, Any]:
    """Quick backtest with VBO strategy and defaults."""
    framework = BacktestFramework()
    return framework.run_backtest(strategy, symbols, **kwargs)


def momentum_backtest(
    strategy: str = "momentum", symbols: list[str] | None = None, **kwargs
) -> dict[str, Any]:
    """Quick backtest with momentum strategy and defaults."""
    framework = BacktestFramework()
    return framework.run_backtest(strategy, symbols, **kwargs)


def buy_and_hold_backtest(symbols: list[str] | None = None, **kwargs) -> dict[str, Any]:
    """Quick buy and hold backtest with defaults."""
    framework = BacktestFramework()
    return framework.run_backtest("buy_and_hold", symbols, **kwargs)
