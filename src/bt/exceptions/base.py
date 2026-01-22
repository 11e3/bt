"""Base exception class."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bt.exceptions.codes import ErrorCode


class BacktestError(Exception):
    """Base exception for all backtesting framework errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc)
        self.cause = cause

    def __str__(self) -> str:
        """String representation with error code."""
        return f"[{self.error_code}] {super().__str__()}"

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "error_code": self.error_code,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
        }
