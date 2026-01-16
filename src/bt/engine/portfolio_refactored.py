"""Refactored Portfolio following SOLID principles.

The new Portfolio class delegates responsibilities to specialized components.
Follows Single Responsibility Principle.
"""

from datetime import datetime

from bt.domain.models import Position, Trade
from bt.domain.types import Amount, Fee, Percentage, Price, Quantity
from bt.engine.equity_tracker import EquityTracker
from bt.engine.order_executor import OrderExecutor
from bt.engine.trade_recorder import TradeRecorder
from bt.utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioRefactored:
    """Manages portfolio state with SOLID principles.

    Responsibilities (ONLY state management):
    - Track cash balance
    - Manage positions
    - Coordinate components

    Delegates to:
    - OrderExecutor: Order execution logic
    - TradeRecorder: Trade history
    - EquityTracker: Equity curve tracking

    Follows:
    - Single Responsibility: Only manages state
    - Open/Closed: Extensible through Order types
    - Liskov Substitution: Implements IPortfolio
    - Interface Segregation: Clean, focused interface
    - Dependency Inversion: Depends on abstractions
    """

    def __init__(
        self,
        initial_cash: Amount,
        fee: Fee,
        slippage: Percentage,
    ) -> None:
        """Initialize portfolio with dependency injection.

        Args:
            initial_cash: Starting capital
            fee: Trading fee as decimal
            slippage: Slippage as decimal
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.fee = fee
        self.slippage = slippage

        # Position tracking
        self._positions: dict[str, Position] = {}

        # Delegate to specialized components (SRP)
        self.order_executor = OrderExecutor(fee, slippage)
        self.trade_recorder = TradeRecorder()
        self.equity_tracker = EquityTracker(initial_cash)

        logger.info(
            "PortfolioRefactored initialized",
            extra={"initial_cash": float(initial_cash), "fee": float(fee)},
        )

    def get_position(self, symbol: str) -> Position:
        """Get or create position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position for the symbol
        """
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        return self._positions[symbol]

    def buy(
        self,
        symbol: str,
        price: Price,
        quantity: Quantity,
        date: datetime,
    ) -> bool:
        """Execute buy order.

        Args:
            symbol: Symbol to buy
            price: Target price
            quantity: Quantity to buy
            date: Execution datetime

        Returns:
            True if order executed successfully
        """
        # Create market buy order
        order = self.order_executor.create_market_buy_order(symbol, quantity, date)

        # Execute order
        success, execution_price, cost = self.order_executor.execute_order(order, price, self.cash)

        if not success:
            return False

        # Update cash
        self.cash = Amount(self.cash - cost)

        # Update position
        position = self.get_position(symbol)
        self.order_executor.update_position_on_buy(position, execution_price, quantity, date)

        logger.debug(
            "Buy order completed",
            extra={
                "symbol": symbol,
                "quantity": float(quantity),
                "price": float(execution_price),
                "remaining_cash": float(self.cash),
            },
        )

        return True

    def sell(
        self,
        symbol: str,
        price: Price,
        date: datetime,
    ) -> bool:
        """Execute sell order for entire position.

        Args:
            symbol: Symbol to sell
            price: Target price
            date: Execution datetime

        Returns:
            True if order executed successfully
        """
        position = self.get_position(symbol)

        if not position.is_open:
            logger.warning(
                "Cannot sell - no open position",
                extra={"symbol": symbol},
            )
            return False

        # Create market sell order for entire position
        order = self.order_executor.create_market_sell_order(symbol, position.quantity, date)

        # Execute order (sell doesn't need cash check)
        success, execution_price, proceeds = self.order_executor.execute_order(
            order, price, self.cash
        )

        if not success:
            return False

        # Calculate P&L
        pnl, return_pct = self.order_executor.calculate_pnl(position, execution_price)

        # Update cash
        self.cash = Amount(self.cash + proceeds)

        # Record trade
        entry_date = position.entry_date or date
        self.trade_recorder.record_trade(
            symbol=symbol,
            entry_date=entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=execution_price,
            quantity=position.quantity,
            pnl=pnl,
            return_pct=return_pct,
        )

        # Clear position
        self._positions[symbol] = Position(symbol=symbol)

        logger.debug(
            "Sell order completed",
            extra={
                "symbol": symbol,
                "price": float(execution_price),
                "pnl": float(pnl),
                "return_pct": float(return_pct),
                "new_cash": float(self.cash),
            },
        )

        return True

    def get_total_value(self, prices: dict[str, Price]) -> Amount:
        """Calculate total portfolio value.

        Args:
            prices: Current prices for all symbols

        Returns:
            Total portfolio value (cash + positions)
        """
        from decimal import Decimal

        total = Decimal(self.cash)

        for symbol, position in self._positions.items():
            if position.is_open and symbol in prices:
                total += Decimal(position.value(prices[symbol]))

        return Amount(total)

    def update_equity(self, date: datetime, prices: dict[str, Price]) -> None:
        """Update equity curve.

        Args:
            date: Current datetime
            prices: Current prices for all symbols
        """
        total_value = self.get_total_value(prices)
        self.equity_tracker.update(date, total_value)

    @property
    def equity_curve(self) -> list:
        """Get equity curve history."""
        return self.equity_tracker.get_equity_curve()

    @property
    def dates(self) -> list[datetime]:
        """Get date history."""
        return self.equity_tracker.get_dates()

    @property
    def trades(self) -> list[Trade]:
        """Get all completed trades."""
        return self.trade_recorder.get_all_trades()

    @property
    def value(self) -> Amount:
        """Get current portfolio value (cash only).

        Returns:
            Current cash amount
        """
        return self.cash

    @property
    def positions(self) -> dict[str, Position]:
        """Get all positions.

        Returns:
            Dictionary of positions by symbol
        """
        return self._positions

    def get_max_quantity_for_buy(self, price: Price) -> Quantity:
        """Calculate maximum quantity that can be purchased.

        Args:
            price: Purchase price

        Returns:
            Maximum affordable quantity
        """
        return self.order_executor.calculate_max_quantity(price, self.cash)
