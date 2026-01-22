"""Custom exception hierarchy and centralized error management.

Provides consistent error handling across the backtesting framework
with structured error codes and context information.
"""

from bt.exceptions.base import BacktestError
from bt.exceptions.codes import ErrorCode
from bt.exceptions.decorators import handle_errors, validate_parameters
from bt.exceptions.exceptions import (
    BacktestExecutionError,
    ConfigurationError,
    DataError,
    InsufficientDataError,
    InsufficientFundsError,
    SecurityError,
    StrategyError,
    ValidationError,
)
from bt.exceptions.handler import ErrorHandler

__all__ = [
    # Error codes
    "ErrorCode",
    # Base exception
    "BacktestError",
    # Specific exceptions
    "ConfigurationError",
    "DataError",
    "ValidationError",
    "InsufficientFundsError",
    "StrategyError",
    "BacktestExecutionError",
    "InsufficientDataError",
    "SecurityError",
    # Handler
    "ErrorHandler",
    # Decorators
    "handle_errors",
    "validate_parameters",
]
