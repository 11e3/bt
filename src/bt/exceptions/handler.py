"""Centralized error handling utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bt.exceptions.exceptions import ValidationError

if TYPE_CHECKING:
    from logging import Logger


class ErrorHandler:
    """Centralized error handling utilities."""

    @staticmethod
    def handle_error(
        error: Exception,
        logger: Logger,
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
