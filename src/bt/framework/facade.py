"""Refactored BacktestFramework following SOLID principles.

This is the new facade that delegates responsibilities to specialized classes.
Follows Single Responsibility Principle by only coordinating components.
"""

import contextlib
from typing import Any

from bt.config.settings import get_config_manager
from bt.core.bootstrap import register_core_services
from bt.core.container import Container, get_default_container, set_default_container
from bt.core.orchestrator import BacktestOrchestrator
from bt.framework.data_loader import DataLoader
from bt.framework.report_generator import ReportGenerator
from bt.framework.runner import BacktestRunner
from bt.framework.strategy_manager import StrategyManager
from bt.interfaces.protocols import ILogger
from bt.security import SecurityManager
from bt.utils.logging import get_logger_adapter


class BacktestFacade:
    """High-level facade for backtesting operations (SRP Compliant).

    Responsibilities (ONLY coordination):
    - Initialize components
    - Delegate to specialized managers
    - Provide unified interface

    Delegates:
    - Strategy management -> StrategyManager
    - Data loading -> DataLoader
    - Backtest execution -> BacktestRunner
    - Report generation -> ReportGenerator
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        container: Container | None = None,
        logger: ILogger | None = None,
    ):
        """Initialize backtest facade with dependency injection.

        Args:
            config: Configuration overrides
            container: Custom dependency injection container
            logger: Custom logger instance
        """
        # Set up container with default services
        if container is not None:
            set_default_container(container)

        if logger is not None and container is not None:
            container.logger = logger

        # Set up configuration manager
        config_manager = get_config_manager()
        if config is not None:
            config_manager.set_config(config)

        self.container = container or get_default_container()
        self.config_manager = config_manager
        self.logger = logger or get_logger_adapter("bt.framework.facade")
        self._config = config or {}

        # Register core services (Dependency Inversion Principle)
        register_core_services(self.container)
        with contextlib.suppress(BaseException):
            self.container.register_singleton(SecurityManager, SecurityManager)

        # Initialize specialized components (Single Responsibility)
        security_manager = self.container.get(SecurityManager)
        orchestrator = BacktestOrchestrator(self.container, config, self.logger)

        # Delegate responsibilities to specialized classes
        self.strategy_manager = StrategyManager(logger=self.logger)
        self.data_loader = DataLoader(logger=self.logger)
        self.runner = BacktestRunner(orchestrator, security_manager, logger=self.logger)

        # Report generator uses config for directory
        report_dir = self.config_manager.get_reporting_config().report_directory
        self.report_generator = ReportGenerator(report_directory=report_dir, logger=self.logger)

        self.logger.info("BacktestFacade initialized with SOLID principles")

    # ============================================================
    # BACKTEST EXECUTION (delegated to BacktestRunner)
    # ============================================================

    def run_backtest(
        self,
        strategy: str,
        symbols: list[str],
        data: dict[str, Any],
        config: dict[str, Any] | None = None,
        **_kwargs,
    ) -> dict[str, Any]:
        """Run a backtest with simplified interface.

        Args:
            strategy: Strategy name to use
            symbols: Symbols to trade
            data: Market data (symbol -> DataFrame)
            config: Additional configuration

        Returns:
            Dictionary with backtest results
        """
        # Merge configurations
        final_config = {**self._config, **(config or {})}

        # Validate configuration if provided
        if config is not None:
            security_manager = self.container.get(SecurityManager)
            config = security_manager.validator.validate(config, "dict")
            final_config = security_manager.validator.validate(final_config, "dict")

            errors = self.config_manager.validate_backtest_parameters(final_config)
            if errors:
                raise ValueError(f"Invalid backtest configuration: {', '.join(errors)}")

        # Create strategy using StrategyManager
        strategy_instance = self.strategy_manager.create_strategy(strategy, final_config)

        # Execute backtest using BacktestRunner
        return self.runner.run(strategy_instance, symbols, data, final_config)

    def run_simple_backtest(
        self,
        strategy: str,
        symbols: list[str] | None = None,
        data: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run simple backtest with default parameters.

        Args:
            strategy: Strategy name
            symbols: Symbols to trade
            data: Market data (if None, will try to load from default directory)
            **kwargs: Additional arguments

        Returns:
            Backtest results
        """
        # Use default configuration
        config = self.config_manager.get_backtest_config()

        # Load data if not provided
        if data is None:
            symbols = symbols or self.config_manager.get_base_config().default_symbols
            data = self.data_loader.load_from_directory("data", symbols)

        return self.run_backtest(strategy, symbols, data, config=config, **kwargs)

    def run_production_backtest(
        self, strategy: str, symbols: list[str], data: dict[str, Any], **kwargs
    ) -> dict[str, Any]:
        """Run production backtest with optimal settings.

        Args:
            strategy: Strategy name
            symbols: Symbols to trade
            data: Market data
            **kwargs: Additional arguments

        Returns:
            Backtest results
        """
        # Use production configuration
        config = self.config_manager.get_backtest_config()

        return self.run_backtest(strategy, symbols, data, config=config, **kwargs)

    # ============================================================
    # STRATEGY MANAGEMENT (delegated to StrategyManager)
    # ============================================================

    def list_available_strategies(self, category: str | None = None) -> list[str]:
        """List all available strategies.

        Args:
            category: Optional category filter

        Returns:
            List of strategy names
        """
        return self.strategy_manager.list_strategies(category)

    def get_strategy_info(self, strategy: str) -> dict[str, Any] | None:
        """Get information about a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Strategy information dictionary
        """
        return self.strategy_manager.get_strategy_info(strategy)

    def create_strategy(self, strategy: str, config: dict[str, Any] | None = None, **kwargs) -> Any:
        """Create strategy instance with configuration.

        Args:
            strategy: Strategy name
            config: Strategy configuration
            **kwargs: Additional arguments

        Returns:
            Strategy instance
        """
        return self.strategy_manager.create_strategy(strategy, config, **kwargs)

    def validate_strategy_config(self, strategy: str, config: dict[str, Any]) -> list[str]:
        """Validate strategy configuration.

        Args:
            strategy: Strategy name
            config: Configuration to validate

        Returns:
            List of validation errors
        """
        return self.strategy_manager.validate_config(strategy, config)

    # ============================================================
    # DATA LOADING (delegated to DataLoader)
    # ============================================================

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
        return self.data_loader.load_from_directory(data_directory, symbols)

    # ============================================================
    # REPORT GENERATION (delegated to ReportGenerator)
    # ============================================================

    def create_performance_report(self, results: dict[str, Any]) -> None:
        """Generate and save performance report.

        Args:
            results: Backtest results dictionary
        """
        self.report_generator.generate_full_report(results)

    def generate_charts(self, results: dict[str, Any]) -> None:
        """Generate performance charts.

        Args:
            results: Backtest results dictionary
        """
        self.report_generator.generate_charts(results)

    def print_summary(self, results: dict[str, Any]) -> None:
        """Print performance summary to console.

        Args:
            results: Backtest results dictionary
        """
        self.report_generator.print_summary(results)

    # ============================================================
    # FRAMEWORK INFORMATION
    # ============================================================

    def get_framework_info(self) -> dict[str, Any]:
        """Get information about the backtesting framework.

        Returns:
            Framework information dictionary
        """
        return {
            "version": "2.0.0-SOLID",
            "strategies_count": len(self.strategy_manager.list_strategies()),
            "categories": self.strategy_manager.get_categories(),
            "container_type": self.container.__class__.__name__,
            "configuration": self.config_manager.__class__.__name__,
            "solid_principles": [
                "Single Responsibility: Separated into specialized managers",
                "Open/Closed: Extensible through strategies and plugins",
                "Liskov Substitution: Protocol-based abstractions",
                "Interface Segregation: Small, focused interfaces",
                "Dependency Inversion: Container-based DI",
            ],
        }


# Convenience functions for common operations


def quick_backtest(
    strategy: str = "volatility_breakout", symbols: list[str] | None = None, **kwargs
) -> dict[str, Any]:
    """Quick backtest with VBO strategy and defaults."""
    facade = BacktestFacade()
    return facade.run_simple_backtest(strategy, symbols, **kwargs)


def momentum_backtest(
    strategy: str = "momentum", symbols: list[str] | None = None, **kwargs
) -> dict[str, Any]:
    """Quick backtest with momentum strategy and defaults."""
    facade = BacktestFacade()
    return facade.run_simple_backtest(strategy, symbols, **kwargs)


def buy_and_hold_backtest(symbols: list[str] | None = None, **kwargs) -> dict[str, Any]:
    """Quick buy and hold backtest with defaults."""
    facade = BacktestFacade()
    return facade.run_simple_backtest("buy_and_hold", symbols, **kwargs)
