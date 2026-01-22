"""Portfolio management for tracking positions and trades.

Manages cash, positions, and trade execution with proper validation.
"""

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from bt.domain.models import Position, Trade
from bt.domain.types import Amount, Fee, Percentage, Price, Quantity
from bt.interfaces.core import Portfolio as PortfolioABC
from bt.utils.constants import ONE
from bt.utils.decimal_cache import get_decimal
from bt.utils.logging import get_logger

logger = get_logger(__name__)


class Portfolio(PortfolioABC):
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
        self._cash = initial_cash
        self.fee = fee
        self.slippage = slippage

        self._positions: dict[str, Position] = {}
        self._trades: list[Trade] = []
        # Optimized storage using numpy arrays
        self._equity_curve = np.array([float(initial_cash)], dtype=np.float64)
        self._dates = np.array([], dtype="datetime64[ns]")

        logger.info(
            "Portfolio initialized",
            extra={"initial_cash": float(initial_cash), "fee": float(fee)},
        )

    @property
    def cash(self) -> Decimal:
        """Available cash."""
        return Decimal(self._cash)

    @cash.setter
    def cash(self, value: Amount) -> None:
        """Set available cash."""
        self._cash = value

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
        execution_price = Price(get_decimal(price) * (ONE + get_decimal(self.slippage)))

        # Calculate total cost including fees
        cost = Amount(
            get_decimal(execution_price) * get_decimal(quantity) * (ONE + get_decimal(self.fee))
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
        self.cash = Amount(self.cash - cost)

        # Update position (add to existing or create new)
        position = self.get_position(symbol)
        if position.is_open:
            # Add to existing position (weighted average pricing)
            total_cost = position.quantity * position.entry_price
            new_cost = quantity * execution_price
            total_quantity = position.quantity + quantity

            avg_price = (total_cost + new_cost) / total_quantity
            position.entry_price = Price(avg_price)
            position.quantity = Quantity(total_quantity)
        else:
            # Create new position
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=Quantity(quantity),
                entry_price=Price(execution_price),
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
        quantity: Quantity,
        date: datetime,
    ) -> bool:
        """Execute sell order with fees and slippage.

        Args:
            symbol: Symbol to sell
            price: Target price
            quantity: Quantity to sell (if None or exceeds position, sells entire position)
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

        # Determine actual quantity to sell (use position quantity if not specified or exceeds)
        sell_quantity = min(get_decimal(quantity), get_decimal(position.quantity))

        # Apply slippage (price decreases when selling)
        execution_price = Price(get_decimal(price) * (ONE - get_decimal(self.slippage)))

        # Calculate proceeds after fees
        proceeds = Amount(
            get_decimal(execution_price) * sell_quantity * (ONE - get_decimal(self.fee))
        )

        # Update cash
        self.cash = Amount(self.cash + Decimal(proceeds))

        # Calculate P&L for the sold quantity (actual proceeds minus proportional cost)
        proportional_entry_cost = (
            position.entry_price
            * sell_quantity
            * (Decimal("1") + Decimal(self.slippage))
            * (Decimal("1") + Decimal(self.fee))
        )
        actual_proceeds = Decimal(proceeds)
        pnl = Amount(actual_proceeds - proportional_entry_cost)
        return_pct = Percentage(
            (Decimal(execution_price) / Decimal(position.entry_price) - Decimal("1"))
            * Decimal("100")
        )

        # Record trade
        entry_date = position.entry_date
        if entry_date is None:
            entry_date = date  # This shouldn't happen in normal operation, but prevents crashes

        trade = Trade(
            symbol=symbol,
            entry_date=entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=execution_price,
            quantity=Quantity(sell_quantity),
            pnl=pnl,
            return_pct=return_pct,
        )
        self._trades.append(trade)

        # Update position (reduce quantity or clear if fully sold)
        remaining_quantity = get_decimal(position.quantity) - sell_quantity
        if remaining_quantity <= Decimal("0"):
            self._positions[symbol] = Position(symbol=symbol)
        else:
            position.quantity = Quantity(remaining_quantity)

        logger.debug(
            "Sell order executed",
            extra={
                "symbol": symbol,
                "quantity": float(sell_quantity),
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
        # Optimized array operations
        self._equity_curve = np.append(self._equity_curve, float(total_value))
        self._dates = np.append(self._dates, np.datetime64(date))

    @property
    def equity_curve(self) -> list[Decimal]:
        """Get equity curve history."""
        return [get_decimal(val) for val in self._equity_curve.tolist()]

    @property
    def dates(self) -> list[datetime]:
        """Get date history."""
        return [pd.Timestamp(val).to_pydatetime() for val in self._dates.tolist() if pd.notna(val)]

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
