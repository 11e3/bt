"""Concrete backtest engine implementing simplified interfaces."""

from decimal import Decimal
from typing import Any

from bt.domain.models import BacktestConfig
from bt.domain.types import Amount, Fee, Percentage, Price
from bt.utils.logging import get_logger

from .interfaces.core import BacktestEngine, DataProvider, Portfolio, Strategy

logger = get_logger(__name__)


class SimpleBacktestEngine(BacktestEngine):
    """Simplified backtest engine with direct implementations."""

    def __init__(self, config: BacktestConfig, data_provider: DataProvider, portfolio: Portfolio):
        super().__init__(config, data_provider, portfolio)
        self._is_running = False
        self._current_bar_index = 0

    def run(self, symbols: list[str], **kwargs) -> None:
        """Run backtest."""
        if not symbols:
            return

        self._is_running = True
        logger.info("Starting backtest...")

        # Validate data availability
        for symbol in symbols:
            if not self.data_provider.has_more_data():
                logger.error(f"No data available for {symbol}")
                return

        # Initialize tracking
        self._current_bar_index += 1

        # Main backtest loop
        while self.data_provider.has_more_data() and self._current_bar_index < len(
            self.data_provider.data_provider.get_data(symbols[0])
        ):
            current_date = self.data_provider.get_current_datetime(symbols[0])
            current_prices = {}

            # Collect current prices (for portfolio valuation)
            for symbol in symbols:
                bar = self.data_provider.get_bar(symbol)
                if bar is not None:
                    current_prices[symbol] = Price(Decimal(str(bar["close"])))

            # Update equity at end of bar
            self.portfolio.update_equity(current_date, current_prices)

            # Process strategies for each symbol
            for symbol in symbols:
                self._process_symbol_strategy(symbol, kwargs)

            # Move to next time period
            self.data_provider.next_bar()
            self._current_bar_index += 1

        self._is_running = False
        logger.info("Backtest completed")

    def _process_symbol_strategy(self, symbol: str, _kwargs: dict[str, Any]) -> None:
        """Process strategy for a single symbol."""
        # Get current position and conditions
        position = self.portfolio.get_position(symbol)

        # Sell logic (check first)
        if position.is_open and self._evaluate_condition(
            self.strategy.get_sell_conditions(), symbol, "sell"
        ):
            current_price = self.strategy.get_sell_price_func()(self, symbol)
            sell_date = self.data_provider.get_current_datetime(symbol)

            if self.portfolio.sell(symbol, current_price, sell_date):
                logger.debug(f"Sell signal triggered for {symbol}")

        # Buy logic (check if no position and all conditions met)
        if not position.is_open and self._evaluate_condition(
            self.strategy.get_buy_conditions(), symbol, "buy"
        ):
            buy_price = self.strategy.get_buy_price_func()(self, symbol)

            # Calculate allocation
            allocation_amount = self.strategy.get_allocation_func()(self, symbol, buy_price)

            if allocation_amount > 0:
                # Execute buy order
                buy_date = self.data_provider.get_current_datetime(symbol)

                # Use simplified portfolio buy method
                if self.portfolio.buy(symbol, buy_price, allocation_amount, buy_date):
                    logger.debug(f"Buy signal for {symbol}")

    def _evaluate_condition(self, conditions: dict[str, Any], symbol: str, context: str) -> bool:
        """Evaluate all conditions for a symbol."""
        for name, condition in conditions.items():
            try:
                if not condition:
                    return False
                return condition(self, symbol, context)
            except Exception as e:
                logger.error(f"Error in condition {name}: {e}")
                return False
        return True


class SimpleStrategy(Strategy):
    """Simplified strategy implementation."""

    def __init__(self, name: str):
        self.name = name
        self._buy_conditions = {}
        self._sell_conditions = {}
        self._buy_price_func = None
        self._sell_price_func = None
        self._allocation_func = None

    def add_buy_condition(self, name: str, condition) -> None:
        """Add buy condition."""
        self._buy_conditions[name] = condition

    def add_sell_condition(self, name: str, condition) -> None:
        """Add sell condition."""
        self._sell_conditions[name] = condition

    def set_buy_conditions(self, conditions: dict[str, Any]) -> None:
        """Set all buy conditions."""
        self._buy_conditions = conditions.copy()

    def set_sell_conditions(self, conditions: dict[str, Any]) -> None:
        """Set all sell conditions."""
        self._sell_conditions = conditions.copy()

    def set_buy_price_func(self, func: Any) -> None:
        """Set buy price function."""
        self._buy_price_func = func

    def set_sell_price_func(self, func: Any) -> None:
        """Set sell price function."""
        self._sell_price_func = func

    def set_allocation_func(self, func: Any) -> None:
        """Set allocation function."""
        self._allocation_func = func


class SimpleBacktestConfig(BacktestConfig):
    """Simplified backtest configuration."""

    def __init__(self, **kwargs):
        # Extract parameters with validation
        initial_cash = kwargs.get("initial_cash", 10000000)
        fee = kwargs.get("fee", 0.0005)
        slippage = kwargs.get("slippage", 0.0005)
        multiplier = kwargs.get("multiplier", 2)
        lookback = kwargs.get("lookback", 5)
        interval = kwargs.get("interval", "day")

        # Validate
        if initial_cash <= 0:
            raise ValueError("Initial cash must be positive")
        if not (0 <= fee <= 1 and 0 <= slippage <= 1):
            raise ValueError("Invalid fee/slippage values")

        super().__init__(
            initial_cash=Amount(Decimal(str(initial_cash))),
            fee=Fee(Decimal(str(fee))),
            slippage=Percentage(Decimal(str(slippage))),
            multiplier=multiplier,
            lookback=lookback,
            interval=interval,
        )


class MomentumStrategy(SimpleStrategy):
    """Momentum-based strategy."""

    def __init__(self, top_n: int = 3, mom_lookback: int = 20):
        super().__init__("momentum")
        self.top_n = top_n
        self.mom_lookback = mom_lookback

    def get_buy_conditions(self) -> dict[str, Any]:
        """Momentum buy conditions."""
        return {
            "no_pos": lambda *_args: True,  # No open position
        }

    def get_sell_conditions(self) -> dict[str, Any]:
        """Momentum sell conditions."""
        return {
            "has_pos": lambda *_args: True,  # Has open position
            "stop_trend": lambda *_args: True,  # Price below MA
        }

    def get_buy_price_func(self) -> Any:
        """Get current close price."""
        return lambda engine, symbol: engine.get_bar(symbol)["close"]

    def get_sell_price_func(self) -> Any:
        """Get current close price for selling."""
        return lambda engine, symbol: engine.get_bar(symbol)["close"]

    def get_allocation_func(self, top_n: int = 3, mom_lookback: int = 20) -> Any:
        """Momentum allocation."""
        from bt.strategies.allocation import create_momentum_allocator

        return create_momentum_allocator(top_n, mom_lookback)


class VBOStrategy(SimpleStrategy):
    """VBO (Volatility Breakout) strategy."""

    def __init__(self):
        super().__init__("vbo")

    def get_buy_conditions(self) -> dict[str, Any]:
        """VBO buy conditions."""
        return {
            "no_pos": lambda *_args: True,
            "breakout": self._is_breakout_triggered,
            "trend_short": self._is_price_above_short_ma,
            "trend_long": self._is_price_above_long_ma,
        }

    def get_sell_conditions(self) -> dict[str, Any]:
        """VBO sell conditions."""
        return {
            "has_pos": lambda *_args: True,
            "stop_trend": self._is_close_below_short_ma,
        }

    def _is_breakout_triggered(self, symbol: str) -> bool:
        """Check if VBO breakout is triggered."""
        buy_price = self.get_buy_price_func()(self, symbol)
        current_bar = self.data_provider.get_bar(symbol)
        if current_bar is None or buy_price == 0:
            return False

        return Decimal(str(current_bar["high"])) >= buy_price

    def _is_price_above_short_ma(self, symbol: str) -> bool:
        """Check if price is above short moving average."""
        bars = self.data_provider.get_bars(symbol, self.config.lookback + 1)
        if bars is None or len(bars) < self.config.lookback + 1:
            return False

        current_bar = self.data_provider.get_bar(symbol)
        close_prices = [Decimal(str(bar["close"])) for bar in bars.iloc[:-1]]
        close_sma = sum(close_prices) / len(close_prices)

        return Decimal(str(current_bar["close"])) >= close_sma

    def _is_price_above_long_ma(self, symbol: str) -> bool:
        """Check if price is above long moving average."""
        long_lookback = self.config.multiplier * self.config.lookback

        bars = self.data_provider.get_bars(symbol, long_lookback + 1)
        if bars is None or len(bars) < long_lookback + 1:
            return False

        current_bar = self.data_provider.get_bar(symbol)
        close_prices = [Decimal(str(bar["close"])) for bar in bars[:-1]]
        close_sma_long = sum(close_prices) / len(close_prices)

        return Decimal(str(current_bar["close"])) >= close_sma_long

    def _is_close_below_short_ma(self, symbol: str) -> bool:
        """Check if previous close was below short MA (lagged signal)."""
        # Use previous bar (offset -1)
        prev_bar = self.data_provider.get_bar(symbol, offset=-1)
        if prev_bar is None:
            return False

        # Calculate short MA using previous bars only (excluding current and previous)
        ma_bars = self.data_provider.get_bars(symbol, self.config.lookback + 1)
        if ma_bars is None or len(ma_bars) < self.config.lookback:
            return False

        close_prices = [Decimal(str(bar["close"])) for bar in ma_bars[:-1]]
        close_sma = sum(close_prices) / len(close_prices)

        return Decimal(str(prev_bar["close"])) < close_sma
