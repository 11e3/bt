"""Specific exception classes."""

from __future__ import annotations

from typing import Any

from bt.exceptions.base import BacktestError
from bt.exceptions.codes import ErrorCode


class ConfigurationError(BacktestError):
    """Exception for configuration-related errors."""

    def __init__(
        self,
        message: str,
        parameter: str | None = None,
        value: Any | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if parameter:
            context["parameter"] = parameter
        if value is not None:
            context["value"] = value

        super().__init__(message, error_code=ErrorCode.INVALID_CONFIG, context=context, **kwargs)


class DataError(BacktestError):
    """Exception for data-related errors."""

    def __init__(
        self,
        message: str,
        symbol: str | None = None,
        expected_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if symbol:
            context["symbol"] = symbol
        if expected_type:
            context["expected_type"] = expected_type

        super().__init__(
            message, error_code=ErrorCode.DATA_VALIDATION_FAILED, context=context, **kwargs
        )


class ValidationError(BacktestError):
    """Exception for input validation errors."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        constraint: str | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = value
        if constraint:
            context["constraint"] = constraint

        super().__init__(message, error_code=ErrorCode.VALIDATION_ERROR, context=context, **kwargs)


class InsufficientFundsError(BacktestError):
    """Exception for insufficient funds during trading."""

    def __init__(self, required: float, available: float, symbol: str, **kwargs: Any) -> None:
        context = kwargs.get("context", {})
        context.update(
            {
                "required": required,
                "available": available,
                "symbol": symbol,
                "shortage": required - available,
            }
        )

        message = (
            f"Insufficient funds for {symbol}: required {required:.2f}, available {available:.2f}"
        )

        super().__init__(
            message, error_code=ErrorCode.INSUFFICIENT_FUNDS, context=context, **kwargs
        )


class StrategyError(BacktestError):
    """Exception for strategy-related errors."""

    def __init__(
        self,
        message: str,
        strategy: str | None = None,
        validation_errors: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if strategy:
            context["strategy"] = strategy
        if validation_errors:
            context["validation_errors"] = validation_errors

        super().__init__(
            message, error_code=ErrorCode.STRATEGY_VALIDATION_FAILED, context=context, **kwargs
        )


class BacktestExecutionError(BacktestError):
    """Exception for backtest execution failures."""

    def __init__(
        self,
        message: str,
        stage: str | None = None,
        progress: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.get("context", {})
        if stage:
            context["stage"] = stage
        if progress is not None:
            context["progress"] = progress

        super().__init__(
            message, error_code=ErrorCode.BACKTEST_EXECUTION_FAILED, context=context, **kwargs
        )


class InsufficientDataError(BacktestError):
    """Raised when insufficient data is available for analysis."""

    def __init__(
        self,
        message: str = "Insufficient data for analysis",
        required_samples: int | None = None,
        available_samples: int | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INSUFFICIENT_DATA,
            context={
                "required_samples": required_samples,
                "available_samples": available_samples,
                **(context or {}),
            },
        )


class SecurityError(BacktestError):
    """Raised when security violations are detected."""

    def __init__(
        self,
        message: str = "Security violation detected",
        violation_type: str | None = None,
        severity: str = "medium",
        context: dict[str, Any] | None = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            context={
                "violation_type": violation_type,
                "severity": severity,
                **(context or {}),
            },
        )

    @staticmethod
    def create_validation_error(
        message: str,
        field: str,
        value: Any,
        constraint: str,
        context: dict[str, Any] | None = None,
    ) -> ValidationError:
        """Create a standardized validation error."""
        return ValidationError(
            message=f"Validation failed for {field}: {message}",
            field=field,
            value=value,
            constraint=constraint,
            context=context,
        )
