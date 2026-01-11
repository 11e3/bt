"""Portfolio management for tracking positions and trades.

Manages cash, positions, and trade execution with proper validation.
"""

from decimal import Decimal
from typing import TYPE_CHECKING

from bt.domain.models import Position, Trade
from bt.domain.types import Amount, Fee, Percentage, Price, Quantity
from bt.logging import get_logger

if TYPE_CHECKING:
    from datetime import datetime

logger = get_logger(__name__)


class Portfolio:
    """Manages portfolio state including cash, positions, and trades.

    Handles:
    - Cash management
    - Position tracking
    - Order execution with fees and slippage
    - Trade recording
    - Equity curve tracking
    """

    def __init__(
        self,
        initial_cash: Amount,
        fee: Fee,
        slippage: Percentage,
    ) -> None:
        """Initialize portfolio.

        Args:
            initial_cash: Starting capital
            fee: Trading fee as decimal (e.g., 0.0005 for 0.05%)
            slippage: Slippage as decimal
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.fee = fee
        self.slippage = slippage

        self._positions: dict[str, Position] = {}
        self._trades: list[Trade] = []
        self._equity_curve: list[Decimal] = [Decimal(initial_cash)]
        self._dates: list[datetime] = []

        logger.info(
            "Portfolio initialized",
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
        """Execute buy order with fees and slippage.

        Args:
            symbol: Symbol to buy
            price: Target price
            quantity: Quantity to buy
            date: Execution datetime

        Returns:
            True if order executed, False if insufficient funds
        """
        # Apply slippage (price increases when buying)
        execution_price = Price(Decimal(price) * (Decimal("1") + Decimal(self.slippage)))

        # Calculate total cost including fees
        cost = Amount(
            Decimal(execution_price) * Decimal(quantity) * (Decimal("1") + Decimal(self.fee))
        )

        if cost > self.cash:
            logger.warning(
                "Insufficient funds for buy order",
                extra={
                    "symbol": symbol,
                    "cost": float(cost),
                    "available_cash": float(self.cash),
                },
            )
            return False

        # Update cash
        self.cash = Amount(Decimal(self.cash) - Decimal(cost))

        # Update position
        self.get_position(symbol)
        self._positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=execution_price,
            entry_date=date,
        )

        logger.debug(
            "Buy order executed",
            extra={
                "symbol": symbol,
                "quantity": float(quantity),
                "price": float(execution_price),
                "cost": float(cost),
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
        """Execute sell order with fees and slippage.

        Args:
            symbol: Symbol to sell
            price: Target price
            date: Execution datetime

        Returns:
            True if order executed, False if no position
        """
        position = self.get_position(symbol)

        if not position.is_open:
            logger.warning(
                "Cannot sell - no open position",
                extra={"symbol": symbol},
            )
            return False

        # Apply slippage (price decreases when selling)
        execution_price = Price(Decimal(price) * (Decimal("1") - Decimal(self.slippage)))

        # Calculate proceeds after fees
        proceeds = Amount(
            Decimal(execution_price)
            * Decimal(position.quantity)
            * (Decimal("1") - Decimal(self.fee))
        )

        # Update cash
        self.cash = Amount(Decimal(self.cash) + Decimal(proceeds))

        # Calculate P&L
        pnl = Amount(
            (Decimal(execution_price) - Decimal(position.entry_price))
            * Decimal(position.quantity)
            * (Decimal("1") - Decimal(self.fee))
        )
        return_pct = Percentage(
            (Decimal(execution_price) / Decimal(position.entry_price) - Decimal("1"))
            * Decimal("100")
        )

        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,  # type: ignore
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=execution_price,
            quantity=position.quantity,
            pnl=pnl,
            return_pct=return_pct,
        )
        self._trades.append(trade)

        # Clear position
        self._positions[symbol] = Position(symbol=symbol)

        logger.debug(
            "Sell order executed",
            extra={
                "symbol": symbol,
                "quantity": float(position.quantity),
                "price": float(execution_price),
                "proceeds": float(proceeds),
                "pnl": float(pnl),
                "return_pct": float(return_pct),
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
        self._equity_curve.append(Decimal(total_value))
        self._dates.append(date)

    @property
    def equity_curve(self) -> list[Decimal]:
        """Get equity curve history."""
        return self._equity_curve

    @property
    def dates(self) -> list[datetime]:
        """Get date history."""
        return self._dates

    @property
    def trades(self) -> list[Trade]:
        """Get all completed trades."""
        return self._trades

    @property
    def value(self) -> Amount:
        """Get current portfolio value (cash only).

        Returns:
            Current cash amount
        """
        return self.cash
