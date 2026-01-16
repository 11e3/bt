"""Domain models for backtesting system.

All financial calculations use Decimal for precision.
Models are validated using Pydantic.
"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from bt.domain.types import Amount, Fee, Percentage, Price, Quantity


class BacktestConfig(BaseModel):
    """Configuration for backtesting engine.

    All financial values use Decimal for precision.
    Validates ranges and constraints.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    initial_cash: Amount = Field(
        default=Amount(Decimal("10000000")),
        description="Initial capital in KRW",
    )
    fee: Fee = Field(
        default=Fee(Decimal("0.0005")),
        description="Trading fee as decimal (0.0005 = 0.05%)",
    )
    slippage: Percentage = Field(
        default=Percentage(Decimal("0.0005")),
        description="Slippage as decimal (0.0005 = 0.05%)",
    )
    multiplier: int = Field(
        default=2,
        ge=1,
        description="Multiplier for long-term indicators",
    )
    lookback: int = Field(
        default=5,
        ge=1,
        description="Lookback period for short-term indicators",
    )
    interval: str = Field(
        default="days",
        description="Time interval for data",
    )

    @field_validator("initial_cash")
    @classmethod
    def validate_initial_cash(cls, v: Amount) -> Amount:
        """Ensure initial cash is positive."""
        if v <= 0:
            raise ValueError("Initial cash must be positive")
        return v

    @field_validator("fee", "slippage")
    @classmethod
    def validate_percentage(cls, v: Percentage | Fee) -> Percentage | Fee:
        """Ensure percentages are valid."""
        if not (0 <= v < 1):
            raise ValueError("Fee/slippage must be between 0 and 1")
        return v


class Position(BaseModel):
    """Represents a trading position.

    Tracks quantity, entry price, and entry date for a symbol.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    symbol: str = Field(description="Trading symbol")
    quantity: Quantity = Field(default=Quantity(Decimal("0")), ge=0)
    entry_price: Price = Field(default=Price(Decimal("0")), ge=0)
    entry_date: datetime | None = Field(default=None)

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
        return Amount(Decimal(self.quantity) * Decimal(current_price))

    def pnl(self, current_price: Price) -> Amount:
        """Calculate unrealized profit/loss.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        if not self.is_open:
            return Amount(Decimal("0"))
        return Amount((Decimal(current_price) - Decimal(self.entry_price)) * Decimal(self.quantity))


class Trade(BaseModel):
    """Represents a completed trade.

    Records entry/exit details and calculates profit/loss.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    symbol: str = Field(description="Trading symbol")
    entry_date: datetime = Field(description="Entry datetime")
    exit_date: datetime = Field(description="Exit datetime")
    entry_price: Price = Field(ge=0, description="Entry price")
    exit_price: Price = Field(ge=0, description="Exit price")
    quantity: Quantity = Field(gt=0, description="Trade quantity")
    pnl: Amount = Field(description="Profit/Loss")
    return_pct: Percentage = Field(description="Return percentage")

    @field_validator("exit_date")
    @classmethod
    def validate_dates(cls, v: datetime, info: ValidationInfo) -> datetime:
        """Ensure exit is after entry."""
        if "entry_date" in info.data and v < info.data["entry_date"]:
            raise ValueError("Exit date must be after entry date")
        return v


class PerformanceMetrics(BaseModel):
    """Performance metrics for backtest results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_return: Percentage = Field(description="Total return percentage")
    cagr: Percentage = Field(description="Compound Annual Growth Rate")
    mdd: Percentage = Field(description="Maximum Drawdown (negative)")
    sharpe_ratio: Decimal = Field(default=Decimal("0"), description="Sharpe ratio")
    sortino_ratio: Decimal = Field(description="Sortino ratio")
    win_rate: Percentage = Field(ge=0, le=100, description="Win rate percentage")
    profit_factor: Decimal = Field(ge=0, description="Profit factor")
    num_trades: int = Field(ge=0, description="Number of trades")
    avg_win: Amount = Field(description="Average winning trade")
    avg_loss: Amount = Field(description="Average losing trade")
    final_equity: Amount = Field(gt=0, description="Final equity")

    # These are excluded from repr to avoid clutter
    equity_curve: list[Decimal] = Field(default_factory=list, repr=False)
    dates: list[datetime] = Field(default_factory=list, repr=False)
    trades: list[Trade] = Field(default_factory=list, repr=False)
    yearly_returns: dict[int, Percentage] = Field(default_factory=dict)


# Rebuild models to resolve forward references
_types_namespace = {"datetime": datetime, "Decimal": Decimal, "Trade": Trade}
Position.model_rebuild(_types_namespace=_types_namespace)
Trade.model_rebuild(_types_namespace=_types_namespace)
PerformanceMetrics.model_rebuild(_types_namespace=_types_namespace)
