"""Bootstrap module for dependency injection container setup.

Registers core services into the DI container.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from bt.core.container import Container
from bt.domain.types import Amount, Fee, Percentage
from bt.interfaces.protocols import (
    IDataProvider,
    ILogger,
    IMetricsGenerator,
    IPortfolio,
)

# Default configuration values
DEFAULT_INITIAL_CASH = Amount(Decimal("10000000"))
DEFAULT_FEE = Fee(Decimal("0.0005"))
DEFAULT_SLIPPAGE = Percentage(Decimal("0.001"))


class MetricsGeneratorAdapter:
    """Adapter to make calculate_performance_metrics work as IMetricsGenerator."""

    def calculate_metrics(
        self,
        equity_curve: list[Decimal],
        dates: list[datetime],
        trades: list[Any],
        initial_cash: Amount,
    ) -> Any:
        """Calculate performance metrics."""
        from bt.reporting.metrics import calculate_performance_metrics

        return calculate_performance_metrics(equity_curve, dates, trades, initial_cash)


def register_core_services(container: Container) -> None:
    """Register core services into the container.

    Args:
        container: DI container to register services into
    """
    from bt.core.simple_implementations import SimpleDataProvider, SimplePortfolio
    from bt.utils.logging import get_logger_adapter

    # Register data provider
    if not container.is_registered(IDataProvider):
        container.register_factory(IDataProvider, SimpleDataProvider)

    # Register portfolio with default configuration
    if not container.is_registered(IPortfolio):
        container.register_factory(
            IPortfolio,
            lambda: SimplePortfolio(DEFAULT_INITIAL_CASH, DEFAULT_FEE, DEFAULT_SLIPPAGE),
        )

    # Register logger
    if not container.is_registered(ILogger):
        container.register_factory(ILogger, lambda: get_logger_adapter("bt"))

    # Register metrics generator
    if not container.is_registered(IMetricsGenerator):
        container.register_factory(IMetricsGenerator, MetricsGeneratorAdapter)
