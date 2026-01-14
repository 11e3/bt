"""Fixed VBO + Momentum allocation strategy."""

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


def create_vbo_momentum_allocator(
    top_n: int = 5, mom_lookback: int = 15
) -> Callable[["BacktestEngine", str, Price], Quantity]:
    """Create VBO + Momentum allocator (fixed look-ahead bias)."""

    def allocator(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
        # 1. Calculate momentum scores for all symbols
        momentum_scores: dict[str, float] = {}
        if engine.data_provider is None:
            return False
        for s in engine.data_provider.symbols:
            score = calculate_momentum_score(engine, s, mom_lookback)
            momentum_scores[s] = score

        # 2. Sort by momentum and get top N
        sorted_symbols = sorted(momentum_scores, key=lambda x: momentum_scores[x], reverse=True)
        top_symbols = sorted_symbols[:top_n]

        # 3. Check if current symbol is in top N
        if symbol not in top_symbols:
            return Quantity(Decimal("0"))

        # 4. Count trading symbols (no position or will be sold this bar)
        trading_symbols = 0
        if engine.portfolio is None:
            return False
        for s in top_symbols:
            pos = engine.portfolio.get_position(s)
            # Check if this symbol would be sold (simplified - assume no scheduled sells)
            if not pos.is_open:
                trading_symbols += 1

        if trading_symbols == 0:
            trading_symbols = 1

        # 5. Equal weight among trading symbols
        total_equity = Decimal(str(engine.portfolio.value))
        target_amount = total_equity / Decimal(trading_symbols)

        # 6. Apply cash constraint
        cash = Decimal(str(engine.portfolio.cash))
        buy_amount = min(target_amount, cash * Decimal("0.99"))

        if buy_amount <= 0:
            return Quantity(Decimal("0"))

        # 7. Calculate quantity with cost consideration
        execution_price = Decimal(str(price)) * (Decimal("1") + Decimal(engine.config.slippage))
        cost_multiplier = Decimal("1") + Decimal(engine.config.fee)
        quantity = buy_amount / (execution_price * cost_multiplier)

        return Quantity(quantity)

    return allocator
