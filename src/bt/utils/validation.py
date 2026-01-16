"""Utility functions for the backtesting framework."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from bt.interfaces.core import ValidationError


def safe_decimal(value: Any, field_name: str = "value") -> Decimal:
    """Convert value to Decimal safely."""
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (ValueError, TypeError) as err:
        raise ValidationError(f"Invalid {field_name}: {value}") from err


def safe_int(value: Any, field_name: str = "value") -> int:
    """Convert value to int safely."""
    if value is None:
        return 0
    try:
        return int(value)
    except (ValueError, TypeError) as err:
        raise ValidationError(f"Invalid {field_name}: {value}") from err


def safe_float(value: Any, field_name: str = "value") -> float:
    """Convert value to float safely."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError) as err:
        raise ValidationError(f"Invalid {field_name}: {value}") from err


def to_datetime(value: Any, field_name: str = "value") -> datetime:
    """Convert pandas datetime to python datetime safely."""
    if value is None:
        raise ValidationError(f"Missing {field_name}")

    if isinstance(value, datetime):
        return value

    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()  # type: ignore[no-any-return]
        except Exception:
            pass

    # Try pandas conversion first
    try:
        import pandas as pd

        return pd.to_datetime(value).to_pydatetime()  # type: ignore[no-any-return]
    except Exception:
        # Fallback to string parsing
        return datetime.fromisoformat(str(value))


def format_decimal(value: Decimal, decimals: int = 2) -> str:
    """Format Decimal for display."""
    return f"{value:.{decimals}f}"


def format_currency(amount: Decimal) -> str:
    """Format amount as currency."""
    if amount < 0:
        return f"({format_decimal(abs(amount))})"
    return f"{format_decimal(amount)}"


def validate_range(
    value: Decimal,
    min_val: Decimal | None = None,
    max_val: Decimal | None = None,
    field_name: str = "value",
) -> None:
    """Validate value is within range [min_val, max_val]."""
    value = safe_decimal(value, field_name)

    if min_val is not None and value < min_val:
        raise ValidationError(f"{field_name} cannot be less than {min_val}")

    if max_val is not None and value > max_val:
        raise ValidationError(f"{field_name} cannot be greater than {max_val}")


# Validation logic was removed - this function is for input validation only


def validate_percentage(value: Any, field_name: str = "value") -> Decimal:
    """Validate value is a valid percentage (0-100).

    Args:
        value: Percentage value (0-100 scale)
        field_name: Name for error messages

    Returns:
        Decimal value converted to decimal scale (0-1)

    Raises:
        ValidationError: If value is negative or greater than 100
    """
    decimal_value = safe_decimal(value, field_name)

    if decimal_value < Decimal("0"):
        raise ValidationError(f"{field_name} cannot be negative")

    if decimal_value > Decimal("100"):
        raise ValidationError(f"{field_name} cannot be greater than 100%")

    return decimal_value * Decimal("0.01")


def validate_positive(value: Any, field_name: str = "value") -> bool:
    """Validate value is positive."""
    return safe_decimal(value, field_name) > Decimal("0")


def validate_non_negative(value: Any, field_name: str = "value") -> bool:
    """Validate value is non-negative."""
    return safe_decimal(value, field_name) >= Decimal("0")


def validate_symbol(value: Any) -> bool:
    """Validate trading symbol."""
    if not value:
        return False

    value_str = str(value).strip().upper()
    return len(value_str) > 0 and all(c.isalpha() or c in "-_." for c in value_str)


def validate_date_range(
    start_date: datetime, end_date: datetime, field_name: str = "date_range"
) -> None:
    """Validate date range is logical."""
    if start_date and end_date and start_date > end_date:
        raise ValidationError(f"{field_name}: start_date must be before end_date")
    return


def calculate_percentage_change(old_value: Decimal, new_value: Decimal) -> Decimal:
    """Calculate percentage change."""
    if old_value == Decimal("0"):
        return new_value if new_value > Decimal("0") else Decimal("0")

    change = new_value - old_value
    return (change / old_value) * Decimal("100") if old_value > 0 else Decimal("0")


def round_to_nearest(value: Decimal, increment: Decimal = Decimal("0.01")) -> Decimal:
    """Round to nearest increment."""
    from decimal import ROUND_HALF_UP

    return (value / increment).quantize(increment, rounding=ROUND_HALF_UP) * increment
