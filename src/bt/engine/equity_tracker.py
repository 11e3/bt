"""Equity curve tracking service.

Responsible for tracking portfolio equity over time.
Follows Single Responsibility Principle.
"""

from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from bt.domain.types import Amount
from bt.utils.decimal_cache import get_decimal
from bt.utils.logging import get_logger

logger = get_logger(__name__)


class EquityTracker:
    """Tracks portfolio equity curve over time.

    Responsibilities:
    - Record equity snapshots
    - Store date/time history
    - Provide equity curve data

    Does NOT handle:
    - Trade execution (OrderExecutor)
    - Position management (Portfolio)
    - Performance calculation (MetricsCalculator)
    """

    def __init__(self, initial_equity: Amount):
        """Initialize equity tracker.

        Args:
            initial_equity: Starting portfolio value
        """
        self.initial_equity = initial_equity

        # Optimized storage using numpy arrays
        self._equity_curve = np.array([float(initial_equity)], dtype=np.float64)
        self._dates = np.array([], dtype="datetime64[ns]")

        logger.info(
            "EquityTracker initialized",
            extra={"initial_equity": float(initial_equity)},
        )

    def update(self, date: datetime, equity: Amount) -> None:
        """Update equity curve with new value.

        Args:
            date: Current datetime
            equity: Current portfolio value
        """
        # Optimized array operations
        self._equity_curve = np.append(self._equity_curve, float(equity))
        self._dates = np.append(self._dates, np.datetime64(date))

    def get_equity_curve(self) -> list[Decimal]:
        """Get equity curve history as Decimal list.

        Returns:
            List of equity values
        """
        return [get_decimal(val) for val in self._equity_curve.tolist()]

    def get_dates(self) -> list[datetime]:
        """Get date history.

        Returns:
            List of datetime objects
        """
        return [pd.Timestamp(val).to_pydatetime() for val in self._dates.tolist() if pd.notna(val)]

    def get_current_equity(self) -> Amount:
        """Get most recent equity value.

        Returns:
            Current equity
        """
        if len(self._equity_curve) > 0:
            return Amount(get_decimal(self._equity_curve[-1]))
        return self.initial_equity

    def get_equity_at_index(self, index: int) -> Amount:
        """Get equity at specific index.

        Args:
            index: Index in equity curve

        Returns:
            Equity value at index

        Raises:
            IndexError: If index is out of range
        """
        return Amount(get_decimal(self._equity_curve[index]))

    def get_date_at_index(self, index: int) -> datetime:
        """Get date at specific index.

        Args:
            index: Index in date history

        Returns:
            Datetime at index

        Raises:
            IndexError: If index is out of range
        """
        return pd.Timestamp(self._dates[index]).to_pydatetime()

    def get_total_return(self) -> Decimal:
        """Calculate total return.

        Returns:
            Total return as decimal (e.g., 1.5 = 150% return)
        """
        if len(self._equity_curve) < 2:
            return Decimal("0")

        initial = self._equity_curve[0]
        final = self._equity_curve[-1]

        if initial == 0:
            return Decimal("0")

        return get_decimal((final / initial) - 1)

    def get_max_equity(self) -> Amount:
        """Get maximum equity reached.

        Returns:
            Maximum equity value
        """
        if len(self._equity_curve) == 0:
            return self.initial_equity

        return Amount(get_decimal(self._equity_curve.max()))

    def get_min_equity(self) -> Amount:
        """Get minimum equity reached.

        Returns:
            Minimum equity value
        """
        if len(self._equity_curve) == 0:
            return self.initial_equity

        return Amount(get_decimal(self._equity_curve.min()))

    def get_drawdown_series(self) -> np.ndarray:
        """Calculate drawdown series.

        Returns:
            Numpy array of drawdown values (negative percentages)
        """
        cummax = np.maximum.accumulate(self._equity_curve)
        return (self._equity_curve - cummax) / cummax

    def get_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown.

        Returns:
            Maximum drawdown as decimal (negative value)
        """
        if len(self._equity_curve) < 2:
            return Decimal("0")

        drawdown = self.get_drawdown_series()
        return get_decimal(drawdown.min())

    def get_length(self) -> int:
        """Get number of equity snapshots.

        Returns:
            Number of recorded equity values
        """
        return len(self._equity_curve)

    def reset(self, initial_equity: Amount) -> None:
        """Reset equity tracker with new initial value.

        Args:
            initial_equity: New starting portfolio value
        """
        self.initial_equity = initial_equity
        self._equity_curve = np.array([float(initial_equity)], dtype=np.float64)
        self._dates = np.array([], dtype="datetime64[ns]")

        logger.info(
            "EquityTracker reset",
            extra={"initial_equity": float(initial_equity)},
        )
