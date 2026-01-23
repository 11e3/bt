"""Allocation components for portfolio sizing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bt.interfaces.strategy_types import IStrategyComponent

if TYPE_CHECKING:
    from bt.interfaces.protocols import IBacktestEngine


class BaseAllocation(IStrategyComponent):
    """Base class for allocation strategies."""

    def __init__(self, **config):
        self.config = config
        self.validate()

    def validate(self) -> bool:
        """Validate allocation configuration."""
        return True

    def get_description(self) -> str:
        """Get allocation description."""
        return f"{self.__class__.__name__}({self.config})"

    def calculate_quantity(self, engine: IBacktestEngine, symbol: str, price: float) -> float:
        """IAllocation protocol method - delegates to __call__."""
        return self(engine, symbol, price)


class AllInAllocation(BaseAllocation):
    """Buy with all available cash accounting for costs.

    Matches standalone/Upbit-style fee model:
    - Fee is deducted from the investment amount (not from transaction)
    - qty = cash * (1 - fee) / (price * (1 + slippage))
    """

    def __call__(self, engine: IBacktestEngine, _symbol: str, price: float) -> float:
        if engine.portfolio.cash <= 0 or price <= 0:
            return 0.0

        # Standalone/Upbit fee model:
        #   buy_value = cash
        #   buy_fee = buy_value * fee (fee from investment amount)
        #   qty = (buy_value - buy_fee) / exec_price
        #       = cash * (1 - fee) / (price * (1 + slippage))
        #   cash = 0 after buy
        #
        # portfolio.buy() then calculates:
        #   total_cost = exec_price * qty * (1 + fee)
        #              = cash * (1 - fee) * (1 + fee)
        #              = cash * (1 - fee²) < cash  ✓
        fee = float(engine.config.fee)
        slippage = float(engine.config.slippage)
        exec_price = price * (1 + slippage)
        cash = float(engine.portfolio.cash)

        return cash * (1 - fee) / exec_price


class EqualWeightAllocation(BaseAllocation):
    """Equal weight allocation across all symbols."""

    def __call__(self, engine: IBacktestEngine, _symbol: str, price: float) -> float:
        num_symbols = len(engine.data_provider.symbols)
        if num_symbols == 0:
            return 0.0

        target_allocation = float(engine.portfolio.cash) / num_symbols
        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        return target_allocation / (price * cost_multiplier)


class MomentumAllocation(BaseAllocation):
    """Momentum allocation - equal weight allocation with momentum filter."""

    def __call__(self, engine: IBacktestEngine, symbol: str, price: float) -> float:
        mom_lookback = self.config.get("mom_lookback", 20)

        # Check momentum for this symbol
        bars = engine.get_bars(symbol, mom_lookback + 2)
        if bars is None or len(bars) < mom_lookback + 2:
            return 0.0

        close_prices = bars["close"].values
        prev_close = close_prices[-2]
        old_close = close_prices[-(mom_lookback + 2)]

        momentum = prev_close / old_close - 1 if old_close > 0 else -999.0
        if np.isnan(momentum) or momentum <= 0:
            return 0.0

        # Equal allocation among symbols
        num_symbols = len(engine.data_provider.symbols)
        if num_symbols == 0:
            return 0.0

        target_allocation = float(engine.portfolio.cash) / num_symbols
        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        return target_allocation / (price * cost_multiplier)


class VolatilityBreakoutAllocation(BaseAllocation):
    """VBO momentum allocation - allocate to top N momentum assets equally."""

    def __call__(self, engine: IBacktestEngine, symbol: str, price: float) -> float:
        top_n = self.config.get("top_n", 3)
        mom_lookback = self.config.get("mom_lookback", 20)

        # Vectorized momentum calculation
        momentum_data = {}
        all_symbols = engine.data_provider.symbols

        for s in all_symbols:
            bars = engine.get_bars(s, mom_lookback + 2)
            if bars is not None and len(bars) >= mom_lookback + 2:
                close_prices = bars["close"].values
                prev_close = close_prices[-2]
                old_close = close_prices[-(mom_lookback + 2)]

                momentum = prev_close / old_close - 1 if old_close > 0 else -999.0
                momentum_data[s] = momentum if not np.isnan(momentum) else -999.0
            else:
                momentum_data[s] = -999.0

        # Get top symbols
        sorted_items = sorted(momentum_data.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [item[0] for item in sorted_items[:top_n]]

        if symbol not in top_symbols:
            return 0.0

        # Equal allocation among top symbols
        total_equity = float(engine.portfolio.value)
        target_amount = total_equity / top_n

        cash = float(engine.portfolio.cash)
        buy_amount = min(target_amount, cash * 0.999)  # Safety buffer

        if buy_amount <= 0:
            return 0.0

        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        return buy_amount / (price * cost_multiplier)


class VBOPortfolioAllocation(BaseAllocation):
    """VBO Portfolio allocation - equal weight among all portfolio coins (1/N).

    Allocates total_equity / n_strategies to each coin.
    Like Upbit, supports fractional (decimal) quantities.

    This returns the target buy_value, not quantity.
    The backtest engine will calculate the actual quantity based on execution price.

    To match standalone logic exactly:
    - buy_value = min(target_alloc, cash * 0.99)
    - Quantity calculation is done by backtest engine using:
      qty = (buy_value - buy_fee) / buy_price
    """

    def __call__(self, engine: IBacktestEngine, _symbol: str, price: float) -> float:
        if price <= 0:
            return 0.0

        # Get number of symbols (n_strategies)
        symbols = engine.data_provider.symbols
        n_strategies = len(symbols)
        if n_strategies == 0:
            return 0.0

        # Calculate total equity = cash + position values at open prices
        cash = float(engine.portfolio.cash)
        position_value = 0.0
        for sym in symbols:
            pos = engine.portfolio.get_position(sym)
            if pos.is_open:
                bar = engine.get_bar(sym)
                if bar is not None:
                    open_price = float(bar["open"])
                    position_value += float(pos.quantity) * open_price

        total_equity = cash + position_value
        target_alloc = total_equity / n_strategies

        # Match standalone logic: buy_value limited to 99% of cash
        buy_value = min(target_alloc, cash * 0.99)

        if buy_value <= 0:
            return 0.0

        # Fee model difference between standalone and framework:
        #
        # Standalone (Upbit-style):
        #   buy_fee = buy_value * fee (fee from investment amount)
        #   qty = (buy_value - buy_fee) / exec_price
        #   cash -= buy_value
        #
        # Framework (standard):
        #   cost = exec_price * qty * (1 + fee) (fee on transaction)
        #   cash -= cost
        #
        # To match standalone exactly, we compute qty that results in
        # portfolio.buy() spending exactly buy_value:
        #   cost = exec_price * qty * (1 + fee) = buy_value
        #   qty = buy_value / (exec_price * (1 + fee))
        #       = buy_value / (price * (1 + slippage) * (1 + fee))
        fee = float(engine.config.fee)
        slippage = float(engine.config.slippage)
        return buy_value / (price * (1 + slippage) * (1 + fee))
