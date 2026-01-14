"""Simplified portfolio allocation strategies.

Eliminates factory pattern overhead with direct functions.
"""

from collections.abc import Callable
from decimal import Decimal
from typing import TYPE_CHECKING

from bt.domain.types import Price, Quantity
from bt.utils.calculations import calculate_cost_multiplier, calculate_max_affordable_quantity
from bt.utils.constants import SAFETY_BUFFER, ZERO
from bt.utils.logging import get_logger

if TYPE_CHECKING:
    from bt.engine.backtest import BacktestEngine

logger = get_logger(__name__)


def all_in_allocation(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
    """Buy with all available cash (simplified)."""
    if engine.portfolio is None:
        return Quantity(ZERO)
    cash = Decimal(str(engine.portfolio.cash))
    current_price = Decimal(str(price))

    if cash <= 0 or current_price <= 0:
        return Quantity(ZERO)

    cost_multiplier = calculate_cost_multiplier(engine.config)
    quantity = calculate_max_affordable_quantity(cash, current_price, cost_multiplier)
    return Quantity(quantity)


def equal_weight_allocation(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
    """Equal weight allocation (simplified)."""
    if engine.data_provider is None or engine.portfolio is None:
        return Quantity(ZERO)
    num_symbols = len(engine.data_provider.symbols)
    if num_symbols == 0:
        return Quantity(ZERO)

    target_allocation = Decimal(engine.portfolio.cash) / Decimal(num_symbols)
    current_price = Decimal(str(price))

    if target_allocation <= 0 or current_price <= 0:
        return Quantity(ZERO)

    cost_multiplier = calculate_cost_multiplier(engine.config)
    quantity = calculate_max_affordable_quantity(target_allocation, current_price, cost_multiplier)
    return Quantity(quantity)


def cash_partition_allocation(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
    """Cash partition allocation (simplified)."""
    # Count open positions
    if engine.data_provider is None or engine.portfolio is None:
        return Quantity(ZERO)
    remaining_assets = sum(
        1 for s in engine.data_provider.symbols if not engine.portfolio.get_position(s).is_open
    )

    if remaining_assets == 0:
        return Quantity(ZERO)

    target_allocation = Decimal(engine.portfolio.cash) / Decimal(remaining_assets)
    current_price = Decimal(str(price))

    if target_allocation <= 0 or current_price <= 0:
        return Quantity(ZERO)

    cost_multiplier = calculate_cost_multiplier(engine.config)
    quantity = calculate_max_affordable_quantity(target_allocation, current_price, cost_multiplier)
    return Quantity(quantity)


def momentum_allocation(
    top_n: int = 3,
    mom_lookback: int = 20,
) -> Callable[["BacktestEngine", str, Price], Quantity]:
    """Momentum allocation (simplified - no factory)."""

    def allocator(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
        """Momentum allocation implementation."""
        # Calculate momentum scores for all symbols
        momentum_scores = {}
        if engine.data_provider is None:
            return Quantity(ZERO)
        for s in engine.data_provider.symbols:
            bars = engine.get_bars(s, mom_lookback + 2)
            if bars is not None and len(bars) >= mom_lookback + 2:
                prev_close = float(bars.iloc[-2]["close"])
                old_close = float(bars.iloc[-(mom_lookback + 2)]["close"])
                score = prev_close / old_close - 1 if old_close > 0 else -999.0
                momentum_scores[s] = score
            else:
                momentum_scores[s] = -999.0

        # Sort by momentum and get top N
        sorted_symbols = sorted(momentum_scores, key=lambda x: momentum_scores[x], reverse=True)
        top_symbols = sorted_symbols[:top_n]

        # Check if current symbol is in top N
        if symbol not in top_symbols:
            return Quantity(ZERO)

        # Count trading symbols (no position)
        if engine.portfolio is None:
            return Quantity(ZERO)
        trading_symbols = sum(
            1 for s in top_symbols if not engine.portfolio.get_position(s).is_open
        )

        if trading_symbols == 0:
            trading_symbols = 1

        total_equity = Decimal(str(engine.portfolio.value))
        target_amount = total_equity / Decimal(trading_symbols)

        # Apply safety buffer
        cash = Decimal(str(engine.portfolio.cash))
        buy_amount = min(target_amount, cash * SAFETY_BUFFER)

        if buy_amount <= 0:
            return Quantity(ZERO)

        # Calculate quantity
        current_price = Decimal(str(price))
        cost_multiplier = calculate_cost_multiplier(engine.config)
        quantity = calculate_max_affordable_quantity(buy_amount, current_price, cost_multiplier)

        return Quantity(quantity)

    return allocator


def aggressive_allocation() -> Callable[["BacktestEngine", str, Price], Quantity]:
    """Aggressive allocation for higher performance."""

    def allocator(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
        """Aggressive allocation implementation."""
        if engine.portfolio is None:
            return Quantity(ZERO)
        # Simple momentum-based allocation with higher targets
        total_equity = Decimal(str(engine.portfolio.value))

        # Use higher allocation percentage for more aggressive performance
        target_amount = total_equity * Decimal("0.20")  # 20% instead of 15%

        cash = Decimal(str(engine.portfolio.cash))
        buy_amount = min(target_amount, cash * Decimal("0.999"))

        if buy_amount <= 0:
            return Quantity(ZERO)

        current_price = Decimal(str(price))
        cost_multiplier = calculate_cost_multiplier(engine.config)
        quantity = calculate_max_affordable_quantity(buy_amount, current_price, cost_multiplier)

        return Quantity(quantity)

    return allocator


def vbo_momentum_allocation(
    top_n: int = 5,
    mom_lookback: int = 15,
) -> Callable[["BacktestEngine", str, Price], Quantity]:
    """VBO + Momentum allocation (simplified)."""

    def allocator(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
        """VBO + Momentum allocation implementation."""
        return momentum_allocation(top_n, mom_lookback)(engine, symbol, price)

    return allocator
