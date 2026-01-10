"""Portfolio allocation strategies."""

from decimal import Decimal
from typing import TYPE_CHECKING

from bt.domain.types import Price, Quantity
from bt.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from bt.engine.backtest import BacktestEngine

logger = get_logger(__name__)


def all_in_allocation(engine: BacktestEngine, symbol: str, price: Price) -> Quantity:
    """Buy with all available cash accounting for costs.

    Used for Buy & Hold or Single Asset strategies.
    """
    cash = Decimal(str(engine.portfolio.cash))
    current_price = Decimal(str(price))

    if cash <= 0 or current_price <= 0:
        return Quantity(Decimal("0"))

    # Calculate cost multiplier (1 + fee + slippage)
    cost_multiplier = Decimal("1") + Decimal(engine.config.fee) + Decimal(engine.config.slippage)

    # Add a tiny safety buffer (0.1%) to prevent precision issues
    safety_buffer = Decimal("0.999")

    available_cash = cash * safety_buffer
    quantity = available_cash / (current_price * cost_multiplier)

    return Quantity(quantity)


def equal_weight_allocation(engine: BacktestEngine, symbol: str, price: Price) -> Quantity:
    """Equal weight allocation across all symbols."""
    num_symbols = len(engine.data_provider.symbols)
    if num_symbols == 0:
        return Quantity(Decimal("0"))

    target_allocation = Decimal(engine.portfolio.cash) / Decimal(num_symbols)

    cost_multiplier = Decimal("1") + Decimal(engine.config.fee) + Decimal(engine.config.slippage)
    quantity = Quantity(target_allocation / (Decimal(price) * cost_multiplier))

    return Quantity(quantity)


def cash_partition_allocation(
    engine: BacktestEngine,
    symbol: str,
    price: Price,
    pool: list[str],
) -> Quantity:
    """Divides remaining cash by the number of remaining assets."""
    remaining_assets = sum(1 for s in pool if not engine.portfolio.get_position(s).is_open)

    if remaining_assets == 0:
        return Quantity(Decimal("0"))

    target_allocation = Decimal(engine.portfolio.cash) / Decimal(remaining_assets)

    cost_multiplier = Decimal("1") + Decimal(engine.config.fee) + Decimal(engine.config.slippage)
    quantity = Quantity(target_allocation / (Decimal(price) * cost_multiplier))

    return Quantity(quantity)


def create_cash_partition_allocator(
    pool: list[str],
) -> Callable[[BacktestEngine, str, Price], Quantity]:
    """Factory for cash partition allocator."""

    def allocator(engine: BacktestEngine, symbol: str, price: Price) -> Quantity:
        return cash_partition_allocation(engine, symbol, price, pool)

    return allocator
