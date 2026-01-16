"""Orchestrator for coordinating backtest execution flow.

Separates high-level backtest coordination from low-level
execution, portfolio management, and performance tracking.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

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
        self.config = config or {}
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
        data: dict[str, any],
    ) -> dict[str, any]:
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
        self.data_provider.set_current_bars_to_start()

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

            # Update performance tracking
            if current_date:
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

                    self.executor.execute_sell_order(
                        symbol, sell_price, quantity, current_date, bar
                    )

                    self.tracker.record_trade(symbol, current_date, "sell", sell_price, quantity)

                    self.portfolio.sell(symbol, sell_price, quantity, current_date)
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

                    self.executor.execute_buy_order(symbol, buy_price, quantity, current_date, bar)

                    self.tracker.record_trade(symbol, current_date, "buy", buy_price, quantity)

                    self.portfolio.buy(symbol, buy_price, quantity, current_date)
                    self.tracker.update_equity(current_date, current_prices)

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
        self, symbol: str, price: Amount, quantity: Amount, date: datetime, _bar: any
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
        self, symbol: str, price: Amount, quantity: Amount, date: datetime, _bar: any
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
        self._dates.append(datetime.now(timezone.utc))  # Start with current date
        self._equity_values.append(float(initial_value))

        self.logger.info(
            "Performance tracking initialized",
            extra={"symbols": symbols, "initial_value": float(initial_value)},
        )

    def update_equity(self, date: datetime, prices: dict[str, Amount]) -> None:
        """Update equity curve for current period."""
        portfolio_value = float(self.container.get(IPortfolio).value)
        total_value = portfolio_value + sum(float(prices.get(symbol, 0)) for symbol in prices)

        self._dates.append(date)
        self._equity_values.append(total_value)

    def record_trade(
        self, symbol: str, date: datetime, action: str, price: Amount, quantity: Amount
    ) -> None:
        """Record a trade for performance tracking."""
        self._trades.append(
            {
                "symbol": symbol,
                "date": date,
                "action": action,
                "price": float(price),
                "quantity": float(quantity),
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
        """Get all recorded trades."""
        return self._trades.copy()

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
                return self.metrics_generator.calculate_metrics(
                    self._equity_values,
                    self._dates,
                    self._trades,
                    Amount(get_decimal(self._initial_value)),
                )
            except Exception as e:
                self.logger.error(f"Performance calculation failed: {e}")
                return {}

        return {}

    def set_initial_positions(self, initial_value: Amount) -> None:
        """Set initial positions for performance tracking."""
        # This would be used for more complex tracking
        # For now, just record as if we started with this amount
        self._initial_value = float(initial_value)
        self._equity_values = [float(initial_value)]

    def get_initial_value(self) -> float:
        """Get initial portfolio value."""
        return self._initial_value or 0.0
