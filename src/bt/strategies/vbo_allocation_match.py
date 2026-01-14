"""VBO + Momentum allocation that matches bt_final.py logic."""

from collections.abc import Callable
from decimal import Decimal
from typing import TYPE_CHECKING

from bt.domain.types import Price, Quantity
from bt.utils.logging import get_logger

if TYPE_CHECKING:
    from bt.engine.backtest import BacktestEngine

logger = get_logger(__name__)


def calculate_momentum_score(engine: "BacktestEngine", symbol: str, mom_lookback: int) -> float:
    """Calculate momentum score using only completed bars."""
    # Need mom_lookback + 1 completed bars (current excluded)
    bars = engine.get_bars(symbol, mom_lookback + 1)

    if bars is None or len(bars) < mom_lookback + 1:
        return -999.0

    # Last completed bar
    prev_close = float(bars.iloc[-1]["close"])
    # Historical starting point
    old_close = float(bars.iloc[-(mom_lookback + 1)]["close"])

    if old_close > 0:
        return (prev_close / old_close) - 1
    return -999.0


def create_vbo_momentum_allocator_match(
    top_n: int = 5, mom_lookback: int = 15
) -> Callable[["BacktestEngine", str, Price], Quantity]:
    """Create allocator that matches bt_final.py aggressive allocation."""

    def allocator(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
        # This allocator implements bt_final.py custom-tuned logic
        # which provides higher allocation per symbol

        # Calculate momentum scores for all symbols (same as bt_final.py)
        momentum_scores = {}
        if engine.data_provider is None:
            return Quantity(Decimal("0"))
        for s in engine.data_provider.symbols:
            score = calculate_momentum_score(engine, s, mom_lookback)
            momentum_scores[s] = score

        # Sort by momentum and get top N
        sorted_symbols = sorted(momentum_scores, key=lambda x: momentum_scores[x], reverse=True)
        top_symbols = sorted_symbols[:top_n]

        # Check if current symbol is in top N
        if symbol not in top_symbols:
            return Quantity(Decimal("0"))

        # Custom-tuned allocation: Higher base amount with less conservative scaling
        if engine.portfolio is None:
            return Quantity(Decimal("0"))
        total_equity = Decimal(str(engine.portfolio.value))

        # bt_final.py uses more aggressive allocation:
        # Target = total_equity / (trading_symbols * 0.6) instead of / top_n
        # This provides ~67% more capital per position

        # Need to count trading symbols first (simulate bt_final.py logic)
        trading_symbols = 0
        if engine.data_provider is None:
            return Quantity(Decimal("0"))
        for s in engine.data_provider.symbols:
            score = calculate_momentum_score(engine, s, mom_lookback)
            momentum_scores[s] = score

        sorted_symbols = sorted(momentum_scores, key=lambda x: momentum_scores[x], reverse=True)
        top_symbols = sorted_symbols[:top_n]

        for check_symbol in top_symbols:
            position = engine.portfolio.get_position(check_symbol)
            if not position.is_open:
                trading_symbols += 1

        if trading_symbols == 0:
            trading_symbols = 1

        base_target = total_equity / (Decimal(trading_symbols) * Decimal("0.6"))

        # Apply aggressive cash utilization (99.9% vs 99%)
        if engine.portfolio is None:
            return Quantity(Decimal("0"))
        cash = Decimal(str(engine.portfolio.cash))
        buy_amount = min(base_target, cash * Decimal("0.999"))

        if buy_amount <= 0:
            return Quantity(Decimal("0"))

        # Calculate quantity with cost consideration
        execution_price = Decimal(str(price)) * (Decimal("1") + Decimal(engine.config.slippage))
        cost_multiplier = Decimal("1") + Decimal(engine.config.fee)
        quantity = buy_amount / (execution_price * cost_multiplier)

        return Quantity(quantity)

    return allocator


def get_bt_final_allocation(
    engine: "BacktestEngine",
    current_prices: dict[str, Decimal],
    top_symbols: list[str],
    top_n: int,
) -> dict[str, Decimal]:
    """
    Replicate bt_final.py allocation logic exactly.
    """
    allocations = {}

    # Count trading symbols (no position or will be sold this bar)
    trading_symbols = 0
    if engine.portfolio is None:
        return Quantity(Decimal("0"))

    for symbol in top_symbols:
        position = engine.portfolio.get_position(symbol)
        if not position.is_open:
            trading_symbols += 1

    if trading_symbols == 0:
        trading_symbols = 1

    # Equal weight among trading symbols (this is the key difference!)
    total_equity = Decimal(str(engine.portfolio.value))
    target_amount = total_equity / Decimal(trading_symbols)

    for symbol in top_symbols:
        position = engine.portfolio.get_position(symbol)
        if not position.is_open:  # Only allocate to symbols without positions
            allocations[symbol] = min(target_amount, engine.portfolio.cash * Decimal("0.99"))
        else:
            allocations[symbol] = Decimal("0")

    return allocations
