"""Structured logging with context and performance tracking."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from bt.monitoring.metrics import MetricsCollector
from bt.utils.logging import get_logger


class StructuredLogger:
    """Enhanced logging with structured context and performance tracking."""

    def __init__(self, base_logger: Any | None = None):
        self.logger = base_logger or get_logger(__name__)
        self._context: dict[str, Any] = {}
        self._metrics_collector = MetricsCollector()

    def set_context(self, **context: Any) -> None:
        """Set logging context."""
        self._context.update(context)

    def clear_context(self) -> None:
        """Clear logging context."""
        self._context.clear()

    @contextmanager
    def context(self, **context: Any):
        """Context manager for temporary logging context."""
        old_context = self._context.copy()
        self._context.update(context)
        try:
            yield
        finally:
            self._context = old_context

    def log_performance(
        self, operation: str, duration: float, success: bool = True, **extra: Any
    ) -> None:
        """Log performance information."""
        log_data = {
            "operation": operation,
            "duration": duration,
            "success": success,
            **self._context,
            **extra,
        }

        if success:
            self.logger.info(f"Operation completed: {operation}", extra=log_data)
        else:
            self.logger.error(f"Operation failed: {operation}", extra=log_data)

        # Record performance metrics
        self._metrics_collector.record_metric(
            f"operation.duration.{operation}", duration, {"success": str(success)}, unit="seconds"
        )

    def log_backtest_event(self, event_type: str, backtest_id: str, **extra: Any) -> None:
        """Log backtest-specific events."""
        log_data = {
            "event_type": event_type,
            "backtest_id": backtest_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self._context,
            **extra,
        }

        self.logger.info(f"Backtest event: {event_type}", extra=log_data)
