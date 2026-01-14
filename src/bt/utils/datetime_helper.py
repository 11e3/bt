"""Datetime utility to fix pandas conversion issues."""

from datetime import datetime
from typing import Any

import pandas as pd


def to_datetime_safe(dt: Any) -> datetime:
    """Convert pandas datetime to python datetime safely."""
    if isinstance(dt, datetime):
        return dt
    if hasattr(dt, "to_pydatetime"):
        return dt.to_pydatetime()  # type: ignore[no-any-return]
    return pd.to_datetime(dt).to_pydatetime()  # type: ignore[no-any-return]


def to_datetime_safe_ms(dt: Any) -> int:
    """Convert pandas datetime to timestamp milliseconds safely."""
    return int(to_datetime_safe(dt).timestamp() * 1000)
