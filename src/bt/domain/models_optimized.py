"""Optimized domain models with memory-efficient implementations.

Provides memory-efficient alternatives to the original models using
dataclasses with slots and other optimizations.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from bt.domain.types import Amount, Percentage, Price, Quantity


@dataclass(slots=True)
class OptimizedPosition:
    """Memory-efficient position representation using slots.

    Uses __slots__ to reduce memory overhead by ~40% compared to
    regular classes/Pydantic models. Suitable for high-volume trading
    with many positions.
    """

    __slots__ = ["symbol", "quantity", "entry_price", "entry_date"]

    symbol: str
    quantity: Decimal
    entry_price: Decimal
    entry_date: datetime | None

    @property
    def is_open(self) -> bool:
        """Check if position is currently open."""
        return self.quantity > 0

    def value(self, current_price: Price) -> Amount:
        """Calculate current position value.

        Args:
            current_price: Current market price

        Returns:
            Position value in base currency
        """
        if not self.is_open:
            return Amount(Decimal("0"))
        return Amount(self.quantity * Decimal(current_price))

    def pnl(self, current_price: Price) -> Amount:
        """Calculate unrealized profit/loss.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        if not self.is_open:
            return Amount(Decimal("0"))
        return Amount((Decimal(current_price) - self.entry_price) * self.quantity)


@dataclass(slots=True)
class OptimizedTrade:
    """Memory-efficient trade representation using slots."""

    __slots__ = [
        "symbol",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "quantity",
        "pnl",
        "return_pct",
    ]

    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: Price
    exit_price: Price
    quantity: Quantity
    pnl: Amount
    return_pct: Percentage

    @property
    def is_winning(self) -> bool:
        """Check if trade is profitable."""
        return self.pnl > 0

    @property
    def duration_days(self) -> int:
        """Calculate trade duration in days."""
        return (self.exit_date - self.entry_date).days


class PositionPool:
    """Memory-efficient pool for position management.

    Recycles position objects to reduce garbage collection overhead
    in high-frequency trading scenarios.
    """

    def __init__(self, initial_size: int = 100):
        """Initialize position pool.

        Args:
            initial_size: Initial number of pre-allocated positions
        """
        self._pool: list[OptimizedPosition] = []
        self._available_indices: list[int] = []
        self._positions: dict[str, OptimizedPosition] = {}

        # Pre-allocate positions
        for _ in range(initial_size):
            self._pool.append(
                OptimizedPosition(
                    symbol="", quantity=Decimal("0"), entry_price=Decimal("0"), entry_date=None
                )
            )
            self._available_indices.append(len(self._pool) - 1)

    def get_position(self, symbol: str) -> OptimizedPosition:
        """Get or create position for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position for the symbol
        """
        if symbol not in self._positions:
            if self._available_indices:
                # Reuse existing position
                idx = self._available_indices.pop()
                position = self._pool[idx]
                position.symbol = symbol
                position.quantity = Decimal("0")
                position.entry_price = Decimal("0")
                position.entry_date = None
            else:
                # Create new position
                position = OptimizedPosition(
                    symbol=symbol, quantity=Decimal("0"), entry_price=Decimal("0"), entry_date=None
                )

            self._positions[symbol] = position

        return self._positions[symbol]

    def release_position(self, symbol: str) -> None:
        """Release position back to pool.

        Args:
            symbol: Symbol to release
        """
        if symbol in self._positions:
            position = self._positions[symbol]
            idx = self._pool.index(position) if position in self._pool else -1

            if idx >= 0:
                self._available_indices.append(idx)

            del self._positions[symbol]

    def get_all_positions(self) -> dict[str, OptimizedPosition]:
        """Get all active positions.

        Returns:
            Dictionary of active positions
        """
        return self._positions.copy()

    def pool_stats(self) -> dict[str, Any]:
        """Get pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        return {
            "pool_size": len(self._pool),
            "available_positions": len(self._available_indices),
            "active_positions": len(self._positions),
            "utilization_rate": len(self._positions) / len(self._pool) if self._pool else 0,
        }


# Factory function for optimized position creation
def create_optimized_position(
    symbol: str, quantity: Quantity = None, entry_price: Price = None, entry_date: datetime = None
) -> OptimizedPosition:
    """Create optimized position with default values.

    Args:
        symbol: Trading symbol
        quantity: Position quantity (defaults to 0)
        entry_price: Entry price (defaults to 0)
        entry_date: Entry date (defaults to None)

    Returns:
        OptimizedPosition instance
    """
    return OptimizedPosition(
        symbol=symbol,
        quantity=Decimal(quantity) if quantity is not None else Decimal("0"),
        entry_price=Decimal(entry_price) if entry_price is not None else Decimal("0"),
        entry_date=entry_date,
    )
