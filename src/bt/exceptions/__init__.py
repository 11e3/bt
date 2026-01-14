"""Custom exception hierarchy and centralized error management.

Provides consistent error handling across the backtesting framework
with structured error codes and context information.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Standardized error codes for better error tracking."""

    # Configuration errors
    INVALID_CONFIG = "INVALID_CONFIG"
    MISSING_CONFIG = "MISSING_CONFIG"
    CONFIG_RANGE_ERROR = "CONFIG_RANGE_ERROR"

    # Data errors
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    DATA_VALIDATION_FAILED = "DATA_VALIDATION_FAILED"
    MISSING_DATA = "MISSING_DATA"
    DATA_CORRUPTION = "DATA_CORRUPTION"

    # Portfolio/Trading errors
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    POSITION_NOT_FOUND = "POSITION_NOT_FOUND"
    EXECUTION_FAILED = "EXECUTION_FAILED"

    # Strategy errors
    STRATEGY_NOT_FOUND = "STRATEGY_NOT_FOUND"
    INVALID_STRATEGY_CONFIG = "INVALID_STRATEGY_CONFIG"
    STRATEGY_VALIDATION_FAILED = "STRATEGY_VALIDATION_FAILED"

    # Backtest errors
    BACKTEST_INITIALIZATION_FAILED = "BACKTEST_INITIALIZATION_FAILED"
    BACKTEST_EXECUTION_FAILED = "BACKTEST_EXECUTION_FAILED"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"

    # General errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"


class BacktestException(Exception):
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


class ConfigurationError(BacktestException):
    """Exception for configuration-related errors."""

    def __init__(
        self, message: str, parameter: str | None = None, value: Any | None = None, **kwargs
    ):
        context = kwargs.get("context", {})
        if parameter:
            context["parameter"] = parameter
        if value is not None:
            context["value"] = value

        super().__init__(message, error_code=ErrorCode.INVALID_CONFIG, context=context, **kwargs)


class DataError(BacktestException):
    """Exception for data-related errors."""

    def __init__(
        self,
        message: str,
        symbol: str | None = None,
        expected_type: str | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if symbol:
            context["symbol"] = symbol
        if expected_type:
            context["expected_type"] = expected_type

        super().__init__(
            message, error_code=ErrorCode.DATA_VALIDATION_FAILED, context=context, **kwargs
        )


class ValidationError(BacktestException):
    """Exception for input validation errors."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        constraint: str | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = value
        if constraint:
            context["constraint"] = constraint

        super().__init__(message, error_code=ErrorCode.VALIDATION_ERROR, context=context, **kwargs)


class InsufficientFundsError(BacktestException):
    """Exception for insufficient funds during trading."""

    def __init__(self, required: float, available: float, symbol: str, **kwargs):
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


class StrategyError(BacktestException):
    """Exception for strategy-related errors."""

    def __init__(
        self,
        message: str,
        strategy: str | None = None,
        validation_errors: list[str] | None = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if strategy:
            context["strategy"] = strategy
        if validation_errors:
            context["validation_errors"] = validation_errors

        super().__init__(
            message, error_code=ErrorCode.STRATEGY_VALIDATION_FAILED, context=context, **kwargs
        )


class BacktestExecutionError(BacktestException):
    """Exception for backtest execution failures."""

    def __init__(
        self, message: str, stage: str | None = None, progress: float | None = None, **kwargs
    ):
        context = kwargs.get("context", {})
        if stage:
            context["stage"] = stage
        if progress is not None:
            context["progress"] = progress

        super().__init__(
            message, error_code=ErrorCode.BACKTEST_EXECUTION_FAILED, context=context, **kwargs
        )


class InsufficientDataError(BacktestException):
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


class SecurityError(BacktestException):
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
            error_code=ErrorCode.VALIDATION_ERROR,  # Could add SECURITY_ERROR to ErrorCode
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


# === ERROR HANDLER ===


class ErrorHandler:
    """Centralized error handling utilities."""

    @staticmethod
    def handle_error(
        error: Exception,
        logger,
        reraise: bool = False,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Handle an exception with logging and optional re-raising."""
        error_data = {
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context or {},
        }
        logger.error(f"Error handled: {error_data}", exc_info=error)
        ErrorHandler._send_to_monitoring(error_data)
        if reraise:
            raise error

    @staticmethod
    def create_validation_error(
        message: str, field: str, value: Any, constraint: str | None = None
    ) -> ValidationError:
        """Create a standardized validation error."""
        return ValidationError(
            message=f"Validation failed for {field}: {message}",
            field=field,
            value=value,
            constraint=constraint,
        )

    @staticmethod
    def _send_to_monitoring(error_data: dict[str, Any]) -> None:
        """Send error to monitoring system (placeholder for production)."""
        # Placeholder for production monitoring integration
        pass


# === DECORATORS ===


def handle_errors(logger, reraise: bool = False):
    """Decorator for automatic error handling on functions.

    Args:
        logger: Logger instance
        reraise: Whether to re-raise exceptions

    Returns:
        Decorated function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ErrorHandler.handle_error(e, logger, reraise, context={"function": func.__name__})
                if not reraise:
                    return None

        return wrapper

    return decorator


def validate_parameters(**validators):
    """Decorator for parameter validation.

    Args:
        **validators: Dict of parameter_name -> validation_function

    Returns:
        Decorated function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    try:
                        validator(kwargs[param_name])
                    except Exception as e:
                        raise ErrorHandler.create_validation_error(
                            message=str(e), field=param_name, value=kwargs[param_name]
                        )
            return func(*args, **kwargs)

        return wrapper

    return decorator
