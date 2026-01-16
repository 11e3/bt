"""Portfolio-related protocols following Interface Segregation Principle.

Smaller, more focused interfaces instead of one large IPortfolio interface.
Each interface has a single, clear purpose.
"""

from datetime import datetime
from decimal import Decimal
from typing import Protocol

from bt.domain.models import Position, Trade
from bt.domain.types import Amount, Price, Quantity


class IPositionManager(Protocol):
    """Interface for position management.

    Only handles position state - does not execute orders or record trades.
    Follows Interface Segregation Principle.
    """

    def get_position(self, symbol: str) -> Position:
        """Get or create position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position for the symbol
        """
        ...

    @property
    def positions(self) -> dict[str, Position]:
        """Get all positions.

        Returns:
            Dictionary of positions by symbol
        """
        ...


class ICashManager(Protocol):
    """Interface for cash management.

    Only handles cash balance - does not execute orders or manage positions.
    Follows Interface Segregation Principle.
    """

    @property
    def cash(self) -> Amount:
        """Get current cash balance.

        Returns:
            Available cash
        """
        ...

    @property
    def initial_cash(self) -> Amount:
        """Get initial cash balance.

        Returns:
            Starting capital
        """
        ...


class IOrderExecutor(Protocol):
    """Interface for order execution.

    Only executes orders - does not manage positions or record trades.
    Follows Interface Segregation Principle.
    """

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
        ...

    def sell(
        self,
        symbol: str,
        price: Price,
        date: datetime,
    ) -> bool:
        """Execute sell order.

        Args:
            symbol: Symbol to sell
            price: Target price
            date: Execution datetime

        Returns:
            True if order executed successfully
        """
        ...


class ITradeRecorder(Protocol):
    """Interface for trade recording.

    Only records trades - does not execute orders or manage positions.
    Follows Interface Segregation Principle.
    """

    def get_all_trades(self) -> list[Trade]:
        """Get all recorded trades.

        Returns:
            List of all trades
        """
        ...

    def get_trades_for_symbol(self, symbol: str) -> list[Trade]:
        """Get trades for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of trades for the symbol
        """
        ...

    @property
    def trades(self) -> list[Trade]:
        """Get all completed trades.

        Returns:
            List of trades
        """
        ...


class IEquityTracker(Protocol):
    """Interface for equity tracking.

    Only tracks equity curve - does not execute orders or manage positions.
    Follows Interface Segregation Principle.
    """

    def update_equity(self, date: datetime, prices: dict[str, Price]) -> None:
        """Update equity curve.

        Args:
            date: Current datetime
            prices: Current prices for all symbols
        """
        ...

    @property
    def equity_curve(self) -> list[Decimal]:
        """Get equity curve history.

        Returns:
            List of equity values
        """
        ...

    @property
    def dates(self) -> list[datetime]:
        """Get date history.

        Returns:
            List of datetime objects
        """
        ...


class IPortfolioValueCalculator(Protocol):
    """Interface for portfolio value calculation.

    Only calculates values - does not modify state.
    Follows Interface Segregation Principle.
    """

    def get_total_value(self, prices: dict[str, Price]) -> Amount:
        """Calculate total portfolio value.

        Args:
            prices: Current prices for all symbols

        Returns:
            Total portfolio value (cash + positions)
        """
        ...

    @property
    def value(self) -> Amount:
        """Get current portfolio value.

        Returns:
            Current value
        """
        ...


# Composite interface for full portfolio functionality
# Components can implement only the interfaces they need


class IFullPortfolio(
    IPositionManager,
    ICashManager,
    IOrderExecutor,
    ITradeRecorder,
    IEquityTracker,
    IPortfolioValueCalculator,
    Protocol,
):
    """Complete portfolio interface composing all portfolio capabilities.

    Use this when you need full portfolio functionality.
    Use individual interfaces when you only need specific capabilities.

    This follows Interface Segregation Principle by:
    - Providing small, focused interfaces
    - Allowing clients to depend on only what they need
    - Composing interfaces for full functionality
    """

    pass
