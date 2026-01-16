"""Backtest execution runner.

Responsible for executing backtests with given configuration and strategy.
Follows Single Responsibility Principle - only handles backtest execution.
"""

from typing import Any

from bt.interfaces.protocols import ILogger, IStrategy
from bt.security import SecurityManager
from bt.utils.logging import get_logger


class BacktestRunner:
    """Executes backtest with strategy and configuration.

    Responsibilities:
    - Validate inputs
    - Execute backtest via orchestrator
    - Return results

    Does NOT handle:
    - Strategy creation (StrategyManager)
    - Data loading (DataLoader)
    - Report generation (ReportGenerator)
    """

    def __init__(
        self,
        orchestrator: Any,
        security_manager: SecurityManager,
        logger: ILogger | None = None,
    ):
        """Initialize backtest runner.

        Args:
            orchestrator: Backtest orchestrator
            security_manager: Security validation manager
            logger: Logger instance
        """
        self.orchestrator = orchestrator
        self.security_manager = security_manager
        self.logger = logger or get_logger(__name__)

    def run(
        self,
        strategy: IStrategy,
        symbols: list[str],
        data: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute backtest with given parameters.

        Args:
            strategy: Strategy to execute
            symbols: Symbols to trade
            data: Market data (symbol -> DataFrame)
            config: Additional configuration

        Returns:
            Dictionary with backtest results

        Raises:
            ValueError: If validation fails
        """
        # Validate inputs
        symbols = self.security_manager.validator.validate(symbols, "list")
        data = self.security_manager.validator.validate(data, "dict")

        # Validate market data
        for symbol, df in data.items():
            self.security_manager.validator.validate(symbol, "symbol")
            self.security_manager.validator.validate(df, "dataframe")

        self.logger.info(
            "Starting backtest execution",
            extra={
                "strategy": strategy.get_name() if hasattr(strategy, "get_name") else "unknown",
                "symbols": symbols,
            },
        )

        # Run backtest via orchestrator
        results = self.orchestrator.run_backtest(strategy, symbols, data)

        # Add configuration to results
        if config:
            results["configuration"] = config

        # Extract total_return from performance (handle both dict and Pydantic model)
        performance = results.get("performance")
        if hasattr(performance, "total_return"):
            total_return = float(performance.total_return)
        elif isinstance(performance, dict):
            total_return = performance.get("total_return", 0)
        else:
            total_return = 0

        self.logger.info(
            "Backtest completed",
            extra={
                "trades": len(results.get("trades", [])),
                "total_return": total_return,
            },
        )

        return results
