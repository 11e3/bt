"""Orchestrator for coordinating backtest execution flow.

Separates high-level backtest coordination from low-level
execution, portfolio management, and performance tracking.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from bt.domain.types import Amount
from bt.interfaces.protocols import (
    IDataProvider,
    ILogger,
    IMetricsGenerator,
    IPortfolio,
    IStrategy,
)
from bt.utils.decimal_cache import get_decimal

if TYPE_CHECKING:
    from bt.core.container import IContainer


class BacktestConfig:
    """Configuration wrapper providing attribute access to config dict."""

    def __init__(self, config_dict: dict):
        self._config = config_dict
        # Set default values
        self.fee = config_dict.get("fee", 0.0005)
        self.slippage = config_dict.get("slippage", 0.001)
        self.lookback = config_dict.get("lookback", 5)
        self.multiplier = config_dict.get("multiplier", 2)
        self.k_factor = config_dict.get("k_factor", 0.5)
        self.top_n = config_dict.get("top_n", 3)
        self.mom_lookback = config_dict.get("mom_lookback", 20)
        self.initial_cash = config_dict.get("initial_cash", 1000000)

    def get(self, key: str, default=None):
        """Get config value with default."""
        return self._config.get(key, default)

    def __getattr__(self, name: str):
        """Fallback attribute access to config dict."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return self._config.get(name)


class BacktestOrchestrator:
    """High-level orchestrator for backtest execution.

    Coordinates:
    - Market data iteration
    - Strategy signal evaluation
    - Order execution via executor
    - Performance tracking via tracker
    """

    def __init__(
        self,
        container: "IContainer",
        config: dict | None = None,
        logger: ILogger | None = None,
    ):
        """Initialize orchestrator.

        Args:
            container: Dependency injection container
            config: Optional configuration overrides
            logger: Optional logger instance
        """
        self.container = container
        self.config = BacktestConfig(config or {})
        self.logger = logger or container.get(ILogger)

        # Get required services
        self.data_provider = container.get(IDataProvider)
        self.portfolio = container.get(IPortfolio)
        self.executor = OrderExecutor(container, self.logger)
        self.tracker = PerformanceTracker(container, self.logger)

    def run_backtest(
        self,
        strategy: IStrategy,
        symbols: list[str],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Run complete backtest with given strategy and data.

        Args:
            strategy: Strategy implementation
            symbols: Symbols to trade
            data: Market data (symbol -> DataFrame)

        Returns:
            Dictionary with backtest results
        """
        self.logger.info(
            "Starting backtest execution",
            extra={"strategy": strategy.get_name(), "symbols": symbols},
        )

        try:
            # Load data for all symbols
            for symbol, df in data.items():
                self.data_provider.load_data(symbol, df)

            # Initialize performance tracking
            self.tracker.initialize(symbols, get_decimal(self.config.get("initial_cash", 1000000)))

            # Set initial positions
            self.tracker.set_initial_positions(
                get_decimal(self.config.get("initial_cash", 1000000))
            )

            # Main execution loop
            self._execute_backtest_loop(strategy, symbols)

            # Generate final results
            results = self._generate_results(strategy)

            self.logger.info(
                "Backtest completed successfully", extra={"trades": len(results.get("trades", []))}
            )

            return results

        except Exception as e:
            self.logger.error(
                "Backtest execution failed",
                extra={"error": str(e), "strategy": strategy.get_name()},
            )
            raise

    def _execute_backtest_loop(self, strategy: IStrategy, symbols: list[str]) -> None:
        """Execute main backtest event loop."""
        # Calculate start index based on lookback configuration
        start_idx = self.config.lookback * self.config.multiplier
        for symbol in symbols:
            self.data_provider.set_current_bar(symbol, start_idx)

        while self.data_provider.has_more_data():
            current_date = None
            current_prices = {}

            # Get current market data
            for symbol in symbols:
                bar = self.data_provider.get_bar(symbol)
                if bar is not None:
                    current_prices[symbol] = get_decimal(bar["close"])
                    if current_date is None:
                        current_date = bar["datetime"]

            # Skip if no valid data for this iteration
            if current_date is None:
                self.data_provider.next_bar()
                continue

            # Update performance tracking
            self.tracker.update_equity(current_date, current_prices)

            # Process each symbol
            for symbol in symbols:
                self._process_symbol(strategy, symbol, current_date, current_prices)

            # Move to next period
            self.data_provider.next_bar()

    def _process_symbol(
        self, strategy: IStrategy, symbol: str, current_date, current_prices: dict[str, any]
    ) -> None:
        """Process trading logic for a single symbol."""
        position = self.portfolio.get_position(symbol)

        # Check sell conditions first
        if position.is_open:
            sell_signals = self._evaluate_strategy_conditions(
                strategy.get_sell_conditions(), symbol
            )

            if all(sell_signals):
                bar = self.data_provider.get_bar(symbol)
                if bar is not None:
                    sell_price = strategy.get_sell_price_func()(self, symbol)
                    quantity = position.quantity

                    # execute_sell_order already calls portfolio.sell() internally
                    self.executor.execute_sell_order(
                        symbol, sell_price, quantity, current_date, bar
                    )

                    self.tracker.record_trade(symbol, current_date, "sell", sell_price, quantity)
                    self.tracker.update_equity(current_date, current_prices)

        # Check buy conditions
        elif not position.is_open:
            buy_signals = self._evaluate_strategy_conditions(strategy.get_buy_conditions(), symbol)

            if all(buy_signals):
                bar = self.data_provider.get_bar(symbol)
                if bar is not None:
                    buy_price = strategy.get_buy_price_func()(self, symbol)

                    # Execute buy order
                    quantity = strategy.get_allocation_func()(self, symbol, buy_price)

                    # Skip if allocation returns 0 (e.g., momentum filter)
                    if quantity <= 0:
                        return

                    # execute_buy_order already calls portfolio.buy() internally
                    self.executor.execute_buy_order(symbol, buy_price, quantity, current_date, bar)

                    self.tracker.record_trade(symbol, current_date, "buy", buy_price, quantity)
                    self.tracker.update_equity(current_date, current_prices)

    def get_bar(self, symbol: str):
        """Get current bar for a symbol.

        Args:
            symbol: Symbol to get bar for

        Returns:
            Current bar data or None if not available
        """
        return self.data_provider.get_bar(symbol)

    def get_bars(self, symbol: str, n: int):
        """Get last n bars for a symbol.

        Args:
            symbol: Symbol to get bars for
            n: Number of bars to retrieve

        Returns:
            DataFrame with n bars or None if not available
        """
        return self.data_provider.get_bars(symbol, n)

    def _evaluate_strategy_conditions(self, conditions: dict[str, any], symbol: str) -> list[bool]:
        """Evaluate all strategy conditions for a symbol."""
        signals = []

        for condition_name, condition_func in conditions.items():
            try:
                signal = condition_func(self, symbol)
                signals.append(signal)
                self.logger.debug(
                    f"Condition {condition_name} for {symbol}: {signal}",
                    extra={"symbol": symbol, "condition": condition_name, "signal": signal},
                )
            except Exception as e:
                self.logger.error(
                    f"Error in condition {condition_name} for {symbol}: {e}",
                    extra={"symbol": symbol, "condition": condition_name, "error": str(e)},
                )
                signals.append(False)

        return signals

    def _generate_results(self, strategy: IStrategy) -> dict[str, any]:
        """Generate comprehensive backtest results."""
        trades = self.tracker.get_all_trades()
        performance_data = self.tracker.get_performance_data()

        return {
            "strategy": strategy.get_name(),
            "trades": trades,
            "performance": performance_data,
            "equity_curve": {
                "dates": self.tracker.get_dates(),
                "values": self.tracker.get_equity_values(),
            },
            "configuration": self.config,
        }


class OrderExecutor:
    """Handles order execution logic and validation."""

    def __init__(self, container: "IContainer", logger: ILogger | None = None):
        """Initialize order executor."""
        self.container = container
        self.logger = logger or container.get(ILogger)

    def execute_buy_order(
        self, symbol: str, price: Amount, quantity: Amount, date: datetime, _bar: Any
    ) -> None:
        """Execute buy order with validation."""
        self.logger.debug(
            "Executing buy order",
            extra={
                "symbol": symbol,
                "price": float(price),
                "quantity": float(quantity),
                "date": date.isoformat(),
            },
        )

        # Validate order
        if not self._validate_order(symbol, price, quantity):
            raise ValueError(f"Invalid buy order for {symbol}")

        # Execute order (delegated to portfolio)
        success = self.container.get(IPortfolio).buy(symbol, price, quantity, date)

        if success:
            self.logger.info(
                f"Buy order executed for {symbol}",
                extra={
                    "symbol": symbol,
                    "price": float(price),
                    "quantity": float(quantity),
                    "date": date.isoformat(),
                },
            )
        else:
            self.logger.warning(
                f"Buy order failed for {symbol}",
                extra={"symbol": symbol, "price": float(price), "quantity": float(quantity)},
            )

    def execute_sell_order(
        self, symbol: str, price: Amount, quantity: Amount, date: datetime, _bar: Any
    ) -> None:
        """Execute sell order with validation."""
        self.logger.debug(
            "Executing sell order",
            extra={
                "symbol": symbol,
                "price": float(price),
                "quantity": float(quantity),
                "date": date.isoformat(),
            },
        )

        # Validate order
        if not self._validate_order(symbol, price, quantity):
            raise ValueError(f"Invalid sell order for {symbol}")

        # Execute order (delegated to portfolio)
        success = self.container.get(IPortfolio).sell(symbol, price, quantity, date)

        if success:
            self.logger.info(
                f"Sell order executed for {symbol}",
                extra={
                    "symbol": symbol,
                    "price": float(price),
                    "quantity": float(quantity),
                    "date": date.isoformat(),
                },
            )
        else:
            self.logger.warning(
                f"Sell order failed for {symbol}",
                extra={"symbol": symbol, "price": float(price), "quantity": float(quantity)},
            )

    def _validate_order(self, symbol: str, price: Amount, quantity: Amount) -> bool:
        """Validate order parameters."""
        try:
            price_value = float(price)
            quantity_value = float(quantity)

            # Basic validation
            if price_value <= 0:
                self.logger.error(f"Invalid price for {symbol}: {price_value}")
                return False

            if quantity_value <= 0:
                self.logger.error(f"Invalid quantity for {symbol}: {quantity_value}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Order validation error for {symbol}: {e}")
            return False


class PerformanceTracker:
    """Tracks performance metrics and equity curve."""

    def __init__(self, container: "IContainer", logger: ILogger | None = None):
        """Initialize performance tracker."""
        self.container = container
        self.logger = logger or container.get(ILogger)
        self.metrics_generator = container.get(IMetricsGenerator)

        # Performance data
        self._dates = []
        self._equity_values = []
        self._trades = []
        self._initial_value = None

    def initialize(self, symbols: list[str], initial_value: Amount) -> None:
        """Initialize tracking with starting values."""
        self._initial_value = float(initial_value)
        # Initialize empty lists - first update_equity call will add initial state
        self._equity_values = []
        self._dates = []

        self.logger.info(
            "Performance tracking initialized",
            extra={"symbols": symbols, "initial_value": float(initial_value)},
        )

    def update_equity(self, date: datetime, prices: dict[str, Amount]) -> None:
        """Update equity curve for current period.

        Args:
            date: Current date
            prices: Dictionary of symbol -> current price (not position values)
        """
        portfolio = self.container.get(IPortfolio)
        cash = float(portfolio.cash)

        # Calculate position values: quantity * current price
        position_value = 0.0
        for symbol, price in prices.items():
            position = portfolio.get_position(symbol)
            if position.is_open:
                position_value += float(position.quantity) * float(price)

        total_value = cash + position_value

        self._dates.append(date)
        self._equity_values.append(total_value)

    def record_trade(
        self, symbol: str, date: datetime, action: str, price: Amount, quantity: Amount
    ) -> None:
        """Record a trade for performance tracking.

        For sell actions, matches with previous buy to calculate P&L.
        """
        if action == "sell":
            # Find matching buy trade for this symbol (FIFO)
            buy_trades = [
                (i, t)
                for i, t in enumerate(self._trades)
                if t.get("symbol") == symbol and t.get("action") == "buy" and not t.get("_matched")
            ]

            if buy_trades:
                idx, last_buy = buy_trades[0]  # FIFO: first unmatched buy
                entry_price = last_buy["price"]
                exit_price = float(price)
                qty = float(quantity)

                # Calculate P&L and return percentage
                pnl = (exit_price - entry_price) * qty
                return_pct = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0.0

                # Mark buy as matched
                self._trades[idx]["_matched"] = True

                # Record completed round-trip trade
                self._trades.append(
                    {
                        "symbol": symbol,
                        "entry_date": last_buy["date"],
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "quantity": qty,
                        "pnl": pnl,
                        "return_pct": return_pct,
                    }
                )

                self.logger.debug(
                    f"Round-trip trade: {symbol}",
                    extra={
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "return_pct": return_pct,
                    },
                )
            else:
                # No matching buy found - just record sell
                self._trades.append(
                    {
                        "symbol": symbol,
                        "date": date,
                        "action": action,
                        "price": float(price),
                        "quantity": float(quantity),
                    }
                )
        else:
            # Record buy for later matching
            self._trades.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "action": action,
                    "price": float(price),
                    "quantity": float(quantity),
                    "_matched": False,
                }
            )

        self.logger.debug(
            f"Trade recorded: {action} {symbol}",
            extra={
                "symbol": symbol,
                "action": action,
                "price": float(price),
                "quantity": float(quantity),
            },
        )

    def get_all_trades(self) -> list[dict[str, any]]:
        """Get all completed round-trip trades (excludes pending buys)."""
        return [
            t
            for t in self._trades
            if "entry_date" in t and "exit_date" in t  # Only round-trip trades
        ]

    def get_dates(self) -> list[datetime]:
        """Get all tracking dates."""
        return self._dates.copy()

    def get_equity_values(self) -> list[float]:
        """Get all equity values."""
        return self._equity_values.copy()

    def get_performance_data(self) -> dict[str, any]:
        """Generate performance metrics using metrics generator."""
        if self._initial_value is not None and len(self._equity_values) > 1:
            try:
                # Only pass completed round-trip trades (not pending buys)
                completed_trades = self.get_all_trades()
                return self.metrics_generator.calculate_metrics(
                    self._equity_values,
                    self._dates,
                    completed_trades,
                    Amount(get_decimal(self._initial_value)),
                )
            except Exception as e:
                self.logger.error(f"Performance calculation failed: {e}")
                return {}

        return {}

    def set_initial_positions(self, initial_value: Amount) -> None:
        """Set initial positions for performance tracking."""
        # Only set initial value - first equity point will be added by update_equity
        # to ensure dates and equity_values stay in sync
        self._initial_value = float(initial_value)

    def get_initial_value(self) -> float:
        """Get initial portfolio value."""
        return self._initial_value or 0.0
