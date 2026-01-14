"""Event-driven backtesting engine.

Main orchestrator that coordinates data, portfolio, and strategy execution.
"""

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from bt.domain.models import BacktestConfig
from bt.domain.types import Price, Quantity
from bt.engine.data_provider import DataProvider
from bt.engine.portfolio import Portfolio
from bt.utils.constants import PRECISION_BUFFER
from bt.utils.decimal_cache import get_decimal
from bt.utils.logging import get_logger

if TYPE_CHECKING:
    from bt.domain.types import (
        AllocationFunc,
        ConditionFunc,
        PriceFunc,
    )

logger = get_logger(__name__)


class BacktestEngine:
    """Event-driven backtesting engine.

    Coordinates:
    - Market data iteration
    - Strategy signal evaluation
    - Order execution via portfolio
    - Performance tracking

    Uses dependency injection for flexibility and testability.
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_provider: DataProvider | None = None,
        portfolio: Portfolio | None = None,
    ) -> None:
        """Initialize backtest engine.

        Args:
            config: Backtest configuration
            data_provider: Optional custom data provider (creates default if None)
            portfolio: Optional custom portfolio (creates default if None)
        """
        self.config = config

        # Dependency injection with defaults
        self.data_provider = data_provider
        self.portfolio = portfolio
        if self.data_provider is None:
            from bt.core.simple_implementations import SimpleDataProvider

            self.data_provider = SimpleDataProvider()
        if self.portfolio is None:
            from bt.core.simple_implementations import SimplePortfolio

            self.portfolio = SimplePortfolio(
                initial_cash=config.initial_cash,
                fee=config.fee,
                slippage=config.slippage,
            )

        logger.info(
            "BacktestEngine initialized",
            extra={
                "config": config.model_dump(),
            },
        )

    def load_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Load market data for a symbol.

        Args:
            symbol: Trading symbol
            df: DataFrame with OHLCV data
        """
        if self.data_provider is None:
            raise ValueError("Data provider not initialized")
        self.data_provider.load_data(symbol, df)

    def get_bar(self, symbol: str, offset: int = 0) -> pd.Series | None:
        """Get bar data (delegates to data provider)."""
        if self.data_provider is None:
            return None
        return self.data_provider.get_bar(symbol, offset)

    def get_bars(self, symbol: str, count: int) -> pd.DataFrame | None:
        """Get multiple bars (delegates to data provider)."""
        if self.data_provider is None:
            return None
        return self.data_provider.get_bars(symbol, count)

    def run(
        self,
        symbols: list[str],
        buy_conditions: dict[str, "ConditionFunc"],
        sell_conditions: dict[str, "ConditionFunc"],
        buy_price_func: "PriceFunc",
        sell_price_func: "PriceFunc",
        allocation_func: "AllocationFunc",
    ) -> None:
        """Run backtest with given strategy.

        Args:
            symbols: List of symbols to trade
            buy_conditions: Dictionary of buy condition functions
            sell_conditions: Dictionary of sell condition functions
            buy_price_func: Function to calculate buy price
            sell_price_func: Function to calculate sell price
            allocation_func: Optional position sizing function
        """
        # Initialize starting position for all symbols
        start_idx = self.config.lookback * self.config.multiplier
        if self.data_provider is None:
            raise ValueError("Data provider not initialized")
        for symbol in symbols:
            self.data_provider.set_current_bar(symbol, start_idx)

        logger.info(
            "Starting backtest",
            extra={"symbols": symbols, "start_idx": start_idx},
        )

        bar_count = 0

        # Main event loop
        while self.data_provider.has_more_data():
            # Optimized batch data access
            if hasattr(self.data_provider, "get_prices_batch"):
                # Use optimized batch methods if available
                raw_prices = self.data_provider.get_prices_batch(symbols)
                current_prices = {
                    symbol: Price(get_decimal(price)) for symbol, price in raw_prices.items()
                }
                current_date = self.data_provider.get_current_datetime_batch(symbols)
            else:
                # Fallback to individual lookups
                current_date: datetime | None = None
                current_prices: dict[str, Price] = {}

                # Collect current prices and date
                for symbol in symbols:
                    bar = self.data_provider.get_bar(symbol)
                    if bar is not None:
                        current_prices[symbol] = Price(get_decimal(bar["close"]))
                        if current_date is None:
                            dt = bar["datetime"]
                            if isinstance(dt, datetime):
                                current_date = dt
                            elif hasattr(dt, "to_pydatetime"):
                                current_date = dt.to_pydatetime()
                            else:
                                current_date = pd.to_datetime(dt).to_pydatetime()

            # Update equity curve
            if current_date:
                # Convert Price to Decimal for portfolio
                decimal_prices = {
                    symbol: get_decimal(price) for symbol, price in current_prices.items()
                }
                self.portfolio.update_equity(current_date, decimal_prices)

            # Process each symbol
            for symbol in symbols:
                bar = self.data_provider.get_bar(symbol)
                if bar is None:
                    continue

                position = self.portfolio.get_position(symbol)

                # Check sell conditions first
                if position.is_open:
                    sell_signals = self._evaluate_conditions(sell_conditions, symbol, "sell")

                    if all(sell_signals):
                        sell_price = sell_price_func(self, symbol)
                        dt = bar["datetime"]
                        if isinstance(dt, datetime):
                            sell_date = dt
                        elif hasattr(dt, "to_pydatetime"):
                            sell_date = dt.to_pydatetime()
                        else:
                            sell_date = pd.to_datetime(dt).to_pydatetime()
                        self.portfolio.sell(
                            symbol, sell_price, Quantity(position.quantity), sell_date
                        )

                # Check buy conditions
                elif not position.is_open:
                    buy_signals = self._evaluate_conditions(buy_conditions, symbol, "buy")

                    if all(buy_signals):
                        buy_price = buy_price_func(self, symbol)

                        # --- Precise Quantity Calculation with Safety Clamp ---

                        # 1. Calculate actual unit cost (matches Portfolio.buy logic exactly)
                        execution_price = get_decimal(buy_price) * (
                            get_decimal("1") + get_decimal(self.config.slippage)
                        )
                        unit_cost = execution_price * (
                            get_decimal("1") + get_decimal(self.config.fee)
                        )

                        quantity = Quantity(get_decimal(0))

                        if unit_cost > 0:
                            # 2. Max quantity affordable with current cash
                            max_qty = get_decimal(self.portfolio.cash) / unit_cost

                            # 3. Apply epsilon buffer (0.99999) to prevent float precision rejection
                            # This ensures we don't try to spend 100.0000001% of cash due to rounding
                            max_affordable_qty = Quantity(max_qty * get_decimal(PRECISION_BUFFER))

                            # 4. Determine requested quantity
                            if allocation_func is not None:
                                requested_qty = allocation_func(self, symbol, buy_price)
                            else:
                                # Default: Buy as much as possible
                                requested_qty = max_affordable_qty

                            # 5. Clamp quantity to max affordable (Safety Check)
                            if requested_qty > max_affordable_qty:
                                quantity = max_affordable_qty
                            else:
                                quantity = requested_qty

                        if quantity > 0:
                            dt = bar["datetime"]
                            if isinstance(dt, datetime):
                                buy_date = dt
                            elif hasattr(dt, "to_pydatetime"):
                                buy_date = dt.to_pydatetime()
                            else:
                                buy_date = pd.to_datetime(dt).to_pydatetime()
                            self.portfolio.buy(symbol, buy_price, quantity, buy_date)

            # Move to next bar
            self.data_provider.next_bar()
            bar_count += 1

        logger.info(
            "Backtest completed",
            extra={
                "bars_processed": bar_count,
                "trades_executed": len(self.portfolio.trades),
            },
        )

    def _evaluate_conditions(
        self,
        conditions: dict[str, "ConditionFunc"],
        symbol: str,
        condition_type: str,
    ) -> list[bool]:
        """Evaluate all conditions and handle errors.

        Args:
            conditions: Dictionary of condition functions
            symbol: Symbol to evaluate
            condition_type: 'buy' or 'sell' for logging

        Returns:
            List of boolean results
        """
        signals = []
        for name, condition_func in conditions.items():
            try:
                signal = condition_func(self, symbol)
                signals.append(signal)
            except Exception as e:
                logger.error(
                    f"Error in {condition_type} condition {name}",
                    extra={"symbol": symbol, "error": str(e)},
                    exc_info=True,
                )
                signals.append(False)

        return signals
