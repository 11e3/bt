"""Trade recording service.

Responsible for recording and managing trade history.
Follows Single Responsibility Principle.
"""

from datetime import datetime

from bt.domain.models import Trade
from bt.domain.types import Amount, Percentage, Price, Quantity
from bt.utils.logging import get_logger

logger = get_logger(__name__)


class TradeRecorder:
    """Records and manages trade history.

    Responsibilities:
    - Record completed trades
    - Store trade history
    - Provide trade statistics

    Does NOT handle:
    - Order execution (OrderExecutor)
    - Position tracking (Portfolio)
    - Performance metrics (MetricsCalculator)
    """

    def __init__(self):
        """Initialize trade recorder."""
        self._trades: list[Trade] = []

    def record_trade(
        self,
        symbol: str,
        entry_date: datetime,
        exit_date: datetime,
        entry_price: Price,
        exit_price: Price,
        quantity: Quantity,
        pnl: Amount,
        return_pct: Percentage,
    ) -> None:
        """Record a completed trade.

        Args:
            symbol: Trading symbol
            entry_date: Entry datetime
            exit_date: Exit datetime
            entry_price: Entry price
            exit_price: Exit price
            quantity: Quantity traded
            pnl: Profit/loss
            return_pct: Return percentage
        """
        trade = Trade(
            symbol=symbol,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            return_pct=return_pct,
        )

        self._trades.append(trade)

        logger.debug(
            f"Trade recorded: {symbol}",
            extra={
                "symbol": symbol,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pnl": float(pnl),
                "return_pct": float(return_pct),
            },
        )

    def get_all_trades(self) -> list[Trade]:
        """Get all recorded trades.

        Returns:
            List of all trades
        """
        return self._trades.copy()

    def get_trades_for_symbol(self, symbol: str) -> list[Trade]:
        """Get trades for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of trades for the symbol
        """
        return [trade for trade in self._trades if trade.symbol == symbol]

    def get_winning_trades(self) -> list[Trade]:
        """Get all winning trades.

        Returns:
            List of trades with positive P&L
        """
        return [trade for trade in self._trades if trade.pnl > 0]

    def get_losing_trades(self) -> list[Trade]:
        """Get all losing trades.

        Returns:
            List of trades with negative or zero P&L
        """
        return [trade for trade in self._trades if trade.pnl <= 0]

    def get_trade_count(self) -> int:
        """Get total number of trades.

        Returns:
            Number of trades
        """
        return len(self._trades)

    def get_win_count(self) -> int:
        """Get number of winning trades.

        Returns:
            Number of winning trades
        """
        return len(self.get_winning_trades())

    def get_loss_count(self) -> int:
        """Get number of losing trades.

        Returns:
            Number of losing trades
        """
        return len(self.get_losing_trades())

    def get_win_rate(self) -> float:
        """Calculate win rate.

        Returns:
            Win rate as percentage (0-100)
        """
        if len(self._trades) == 0:
            return 0.0

        return (self.get_win_count() / len(self._trades)) * 100

    def clear_trades(self) -> None:
        """Clear all recorded trades."""
        self._trades.clear()
        logger.info("Trade history cleared")
