"""Vectorized allocation strategies for performance optimization."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from bt.domain.types import Price, Quantity
from bt.utils.calculations import calculate_cost_multiplier
from bt.utils.constants import SAFETY_BUFFER, ZERO
from bt.utils.decimal_cache import get_decimal
from bt.utils.logging import get_logger

if TYPE_CHECKING:
    from bt.engine.backtest import BacktestEngine

logger = get_logger(__name__)


def create_momentum_allocator_optimized(
    top_n: int = 3, mom_lookback: int = 20
) -> Callable[["BacktestEngine", str, Price], Quantity]:
    """
    Optimized momentum allocator with vectorized operations.

    Vectorized momentum calculation for all symbols simultaneously.
    Uses numpy arrays for better performance on large symbol sets.
    """

    def allocator(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
        # Vectorized momentum calculation for all symbols
        momentum_data = {}

        # Get all symbols data at once for efficiency
        all_symbols = engine.data_provider.symbols
        required_bars = mom_lookback + 2  # Current + lookback + 1 for previous day

        for s in all_symbols:
            bars = engine.get_bars(s, required_bars)
            if bars is not None and len(bars) >= required_bars:
                # Vectorized calculation using numpy
                close_prices = bars["close"].values
                if len(close_prices) >= required_bars:
                    # Use previous day's close (look-ahead bias prevention)
                    prev_close = close_prices[-2]
                    old_close = close_prices[-(mom_lookback + 2)]

                    momentum = prev_close / old_close - 1 if old_close > 0 else -999.0
                    momentum_data[s] = momentum if not np.isnan(momentum) else -999.0
                else:
                    momentum_data[s] = -999.0
            else:
                momentum_data[s] = -999.0

        # Vectorized sorting using numpy
        symbols_array = np.array(list(momentum_data.keys()))
        momentums_array = np.array(list(momentum_data.values()))

        # Get top N indices
        top_indices = np.argpartition(-momentums_array, top_n)[:top_n]
        top_symbols = set(symbols_array[top_indices])

        if symbol not in top_symbols:
            return Quantity(ZERO)

        # Calculate allocation
        total_equity = get_decimal(engine.portfolio.value)
        target_amount = total_equity / get_decimal(top_n)

        cash = get_decimal(engine.portfolio.cash)
        buy_amount = min(target_amount, cash * get_decimal(SAFETY_BUFFER))

        if buy_amount <= 0:
            return Quantity(ZERO)

        current_price = get_decimal(price)
        cost_multiplier = calculate_cost_multiplier(engine.config)
        return Quantity(buy_amount / (current_price * cost_multiplier))

    return allocator


def create_equal_weight_momentum_allocator(
    top_n: int = 3, mom_lookback: int = 20
) -> Callable[["BacktestEngine", str, Price], Quantity]:
    """
    Equal weight momentum allocator with vectorized operations.

    Allocates equally among top momentum symbols.
    """

    def allocator(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
        all_symbols = engine.data_provider.symbols
        if not all_symbols:
            return Quantity(ZERO)

        # Vectorized momentum calculation
        momentum_scores = {}

        for s in all_symbols:
            bars = engine.get_bars(s, mom_lookback + 2)
            if bars is not None and len(bars) >= mom_lookback + 2:
                close_prices = bars["close"].values
                prev_close = close_prices[-2]
                old_close = close_prices[-(mom_lookback + 2)]

                momentum = prev_close / old_close - 1 if old_close > 0 else -999.0
                momentum_scores[s] = momentum
            else:
                momentum_scores[s] = -999.0

        # Get top symbols
        sorted_items = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [item[0] for item in sorted_items[:top_n]]

        if symbol not in top_symbols:
            return Quantity(ZERO)

        # Equal weight among top symbols
        cash = get_decimal(engine.portfolio.cash)
        target_allocation = cash / get_decimal(len(top_symbols))

        current_price = get_decimal(price)
        cost_multiplier = calculate_cost_multiplier(engine.config)
        return Quantity(target_allocation / (current_price * cost_multiplier))

    return allocator


def adaptive_allocation(engine: "BacktestEngine", symbol: str, price: Price) -> Quantity:
    """
    Adaptive allocation based on portfolio size and volatility.

    Allocates more to less volatile symbols, dynamically adjusting
    allocation based on current portfolio composition.
    """
    # Calculate portfolio volatility (simplified)
    symbols = engine.data_provider.symbols
    if not symbols:
        return Quantity(ZERO)

    # Get recent returns for all symbols
    returns_data = {}
    for s in symbols:
        bars = engine.get_bars(s, 21)  # 20 returns
        if bars is not None and len(bars) >= 21:
            close_prices = bars["close"].values
            returns = np.diff(close_prices) / close_prices[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 1.0
            returns_data[s] = volatility
        else:
            returns_data[s] = 1.0  # Default high volatility

    # Calculate inverse volatility weights
    if symbol not in returns_data:
        return Quantity(ZERO)

    symbol_volatility = returns_data[symbol]
    inv_volatility_sum = sum(1.0 / v for v in returns_data.values() if v > 0)

    if inv_volatility_sum == 0:
        return Quantity(ZERO)

    weight = (1.0 / symbol_volatility) / inv_volatility_sum

    # Allocate based on inverse volatility
    cash = get_decimal(engine.portfolio.cash)
    target_amount = cash * get_decimal(weight)

    current_price = get_decimal(price)
    cost_multiplier = calculate_cost_multiplier(engine.config)
    return Quantity(target_amount / (current_price * cost_multiplier))
