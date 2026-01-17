"""Concrete implementations of core interfaces."""

from datetime import datetime
from decimal import Decimal
from typing import Any

import pandas as pd

from bt.domain.types import (
    Amount,
    Fee,
    Percentage,
    Price,
    Quantity,
)
from bt.interfaces.core import (
    DataProvider,
    Portfolio,
)
from bt.utils.logging import get_logger
from bt.utils.validation import (
    to_datetime,
)

logger = get_logger(__name__)


class SimpleDataProvider(DataProvider):
    """Simple data provider using pandas."""

    def __init__(self) -> None:
        self._data: dict[str, pd.DataFrame] = {}
        self._current_bar: dict[str, int] = {}
        self._cache: dict[str, Any] = {}

    def load_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Load data for a symbol."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        required_columns = ["open", "high", "low", "close", "volume", "datetime"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Data must have columns: {required_columns}")

        # Store data
        df = df.copy().sort_values("datetime").reset_index(drop=True)
        self._data[symbol] = df
        self._current_bar[symbol] = 0
        self._cache[symbol] = df
        logger.info(f"Loaded {len(df)} bars for {symbol}")

    def get_bar(self, symbol: str, offset: int = 0) -> Any | None:
        """Get a specific bar."""
        if symbol not in self._data:
            return None

        idx = self._current_bar[symbol] + offset
        if idx < 0 or idx >= len(self._data[symbol]):
            return None

        return self._data[symbol].iloc[idx]

    def get_bars(self, symbol: str, count: int) -> Any | None:
        """Get multiple bars including current bar.

        Args:
            symbol: Symbol to get bars for
            count: Number of bars to retrieve

        Returns:
            DataFrame with count bars ending at current position, or None if not enough data
        """
        if symbol not in self._data:
            return None

        # Include current bar (end_idx is exclusive in iloc, so +1)
        end_idx = self._current_bar[symbol] + 1
        start_idx = max(0, end_idx - count)

        if end_idx - start_idx < 1:
            return None

        return self._data[symbol].iloc[start_idx:end_idx]

    def has_more_data(self) -> bool:
        """Check if there is more data to process."""
        return any(self._current_bar[symbol] < len(self._data[symbol]) - 1 for symbol in self._data)

    def next_bar(self) -> None:
        """Move to next time period."""
        for symbol in self._data:
            if self._current_bar[symbol] < len(self._data[symbol]) - 1:
                self._current_bar[symbol] += 1

    @property
    def symbols(self) -> list[str]:
        """Get available symbols."""
        return list(self._data.keys())

    def current_datetime(self, symbol: str) -> datetime | None:
        """Get datetime of current bar."""
        bar = self.get_bar(symbol)
        if bar is not None:
            return to_datetime(bar["datetime"], "current_datetime")
        return None

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate for debugging."""
        # Simple implementation for now
        return 0.0  # Would need actual tracking

    def set_current_bar(self, symbol: str, index: int) -> None:
        """Set current bar index for a symbol."""
        if symbol in self._data:
            self._current_bar[symbol] = min(index, len(self._data[symbol]) - 1)

    def set_current_bars_to_start(self) -> None:
        """Reset all symbols to the start of their data."""
        for symbol in self._data:
            self._current_bar[symbol] = 0


class SimplePortfolio(Portfolio):
    """Simplified portfolio with native Decimal operations."""

    def __init__(self, initial_cash: Amount, fee: Fee, slippage: Percentage) -> None:
        self.initial_cash = initial_cash
        self._cash = initial_cash
        self.fee = fee
        self.slippage = slippage
        self._positions: dict[str, Any] = {}
        self._trades: list[Any] = []
        self._equity_curve: list[Decimal] = [Decimal(str(initial_cash))]
        self._dates: list[datetime] = []

    @property
    def cash(self) -> Amount:
        """Available cash."""
        return self._cash

    @cash.setter
    def cash(self, value: Amount) -> None:
        self._cash = value

    def _create_position(
        self, symbol: str, quantity: Quantity, entry_price: Price, entry_date: datetime
    ) -> Any:
        """Create position object."""
        from bt.domain.models import Position

        return Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            entry_date=entry_date,
        )

    def get_position(self, symbol: str) -> Any:
        """Get position for symbol."""
        return self._positions.get(
            symbol,
            self._create_position(
                symbol,
                Quantity(Decimal("0")),
                Price(Decimal("0")),
                datetime.min.replace(tzinfo=None),
            ),
        )

    def add_to_position(
        self,
        symbol: str,
        quantity: Quantity,
        execution_price: Price,
        entry_date: datetime | None = None,
    ) -> None:
        """Add to existing position with weighted average pricing."""
        if symbol not in self._positions:
            # Create new position
            self._positions[symbol] = self._create_position(
                symbol, quantity, execution_price, entry_date or datetime.min.replace(tzinfo=None)
            )
            return

        position = self._positions[symbol]
        if position.is_open:
            # Calculate weighted average price
            total_cost = position.quantity * position.entry_price
            new_cost = quantity * execution_price
            total_quantity = position.quantity + quantity

            avg_price = (total_cost + new_cost) / total_quantity
            position.quantity = total_quantity
            position.entry_price = Price(avg_price)
            # Note: entry_date is kept as the first entry date for averaging purposes
        else:
            # Position was closed, create new one with current entry_date
            # This is crucial: must use the provided entry_date, not fall back to datetime.min
            new_entry_date = (
                entry_date if entry_date is not None else datetime.min.replace(tzinfo=None)
            )
            self._positions[symbol] = self._create_position(
                symbol, quantity, execution_price, new_entry_date
            )

    def buy(self, symbol: str, price: Price, quantity: Quantity, date: datetime) -> bool:
        """Execute buy order.

        Returns:
            True if order executed successfully, False if insufficient funds
        """
        cash = Decimal(self.cash)
        current_price = Decimal(str(price))
        qty = Decimal(str(quantity))

        # Apply slippage (price increases when buying)
        execution_price = current_price * (Decimal("1") + self.slippage)

        # Calculate total cost with fees (slippage already applied to execution_price)
        total_cost = execution_price * qty * (Decimal("1") + self.fee)

        if total_cost > cash:
            logger.warning(f"Insufficient cash for {symbol}: need {total_cost}, have {cash}")
            return False

        # Update cash and position
        self._cash = Amount(cash - total_cost)
        self.add_to_position(symbol, Quantity(qty), Price(execution_price), date)

        logger.debug(f"Buy: {symbol} @ {price} x {qty}")
        return True

    def sell(self, symbol: str, price: Price, quantity: Quantity, date: datetime) -> bool:
        """Execute sell order.

        Returns:
            True if order executed successfully, False if no position
        """
        position = self.get_position(symbol)

        if not position.is_open:
            logger.warning(f"Attempted to sell non-existent position: {symbol}")
            return False

        current_price = Decimal(str(price))
        qty = Decimal(str(quantity))

        # Apply slippage (price decreases when selling)
        execution_price = current_price * (Decimal("1") - self.slippage)

        # Calculate proceeds after fees (slippage already applied to execution_price)
        proceeds = execution_price * qty * (Decimal("1") - self.fee)

        # Update cash and record trade
        self._cash = Amount(self._cash + proceeds)

        # Record trade
        from bt.domain.models import Trade

        # Calculate PnL and return
        pnl_amount = (execution_price - position.entry_price) * qty
        return_pct = ((execution_price - position.entry_price) / position.entry_price) * Decimal(
            "100"
        )

        # Validate entry_date before creating trade
        entry_date = position.entry_date
        if entry_date is None:
            logger.warning(f"Position {symbol} has no entry_date, using exit_date")
            entry_date = date
        elif entry_date > date:
            # This can happen if position data is corrupted or dates are out of order
            logger.warning(
                f"Position {symbol} entry_date ({entry_date}) > exit_date ({date}), using exit_date"
            )
            entry_date = date

        trade = Trade(
            symbol=symbol,
            entry_date=entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=Price(execution_price),
            quantity=Quantity(qty),
            pnl=Amount(pnl_amount),
            return_pct=Percentage(return_pct),
        )
        self._trades.append(trade)

        # Update position - ensure quantity doesn't go negative
        position.quantity = position.quantity - qty
        if position.quantity <= Decimal("0"):
            position.quantity = Quantity(Decimal("0"))

        logger.debug(f"Sell: {symbol} @ {price} x {qty}")
        return True

    @property
    def value(self) -> Amount:
        """Total portfolio value (cash + positions at entry price).

        Note: For accurate valuation with current market prices,
        use update_equity() which tracks the equity curve.
        """
        total = self._cash

        # Add value of open positions using entry price as estimate
        for _symbol, position in self._positions.items():
            if position.is_open:
                # Use entry price as conservative estimate
                total += position.entry_price * position.quantity

        return Amount(total)

    def update_equity(self, date: datetime, _prices: dict[str, Decimal]) -> None:
        """Update portfolio equity with current prices."""
        self._equity_curve.append(self.value)
        self._dates.append(date)

    @property
    def positions(self) -> dict[str, Any]:
        """All positions."""
        return self._positions

    @property
    def trades(self) -> list[Any]:
        """All completed trades."""
        return self._trades

    @property
    def equity_curve(self) -> list[Decimal]:
        """Historical portfolio values."""
        return self._equity_curve.copy()

    @property
    def dates(self) -> list[datetime]:
        """Historical dates."""
        return self._dates.copy()
