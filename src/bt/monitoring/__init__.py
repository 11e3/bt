"""Performance monitoring and metrics collection system.

Provides comprehensive monitoring capabilities including:
- Structured logging with context
- Performance metrics collection
- Alerting framework
- Prometheus-style metrics export
- Performance profiling and tracing
"""

import inspect
import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Optional, Union

import psutil

from bt.utils.logging import get_logger

from .exporters import (
    EmailAlerter,
    JSONMetricsExporter,
    MetricsServer,
    PrometheusExporter,
    SlackAlerter,
    start_metrics_server,
    stop_metrics_server,
)

logger = get_logger(__name__)


@dataclass
class MetricValue:
    """Represents a single metric measurement."""

    name: str
    value: int | float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)
    unit: str | None = None


@dataclass
class AlertRule:
    """Defines an alerting rule."""

    name: str
    condition: Callable[[float], bool]
    message: str
    severity: str = "warning"  # info, warning, error, critical
    cooldown_minutes: int = 5


class MetricsCollector:
    """Collects and manages performance metrics."""

    def __init__(self, max_history: int = 10000):
        self._metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._gauges: dict[str, float] = {}
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.RLock()
        self._alert_rules: dict[str, AlertRule] = {}
        self._alert_cooldowns: dict[str, datetime] = {}

    def record_metric(
        self,
        name: str,
        value: int | float,
        labels: dict[str, str] | None = None,
        unit: str | None = None,
    ) -> None:
        """Record a metric value."""
        with self._lock:
            metric = MetricValue(
                name=name,
                value=value,
                timestamp=datetime.now(timezone.utc),
                labels=labels or {},
                unit=unit,
            )
            self._metrics[name].append(metric)

            # Check alert rules
            self._check_alerts(name, value)

    def set_gauge(
        self, name: str, value: int | float, labels: dict[str, str] | None = None
    ) -> None:
        """Set a gauge metric (current value)."""
        with self._lock:
            self._gauges[name] = value
            self.record_metric(name, value, labels, "gauge")

    def increment_counter(
        self, name: str, value: int = 1, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value
            self.record_metric(name, self._counters[name], labels, "counter")

    def observe_histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Observe a histogram metric."""
        with self._lock:
            self._histograms[name].append(value)
            self.record_metric(name, value, labels, "histogram")

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._alert_rules[rule.name] = rule

    def get_metrics(
        self,
        name: str | None = None,
        labels: dict[str, str] | None = None,
        since: datetime | None = None,
    ) -> list[MetricValue]:
        """Get metrics matching criteria."""
        with self._lock:
            if name:
                metrics = list(self._metrics[name])
            else:
                metrics = []
                for metric_list in self._metrics.values():
                    metrics.extend(metric_list)

            # Filter by labels
            if labels:
                metrics = [
                    m for m in metrics if all(m.labels.get(k) == v for k, v in labels.items())
                ]

            # Filter by time
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]

            return metrics

    def get_metric_stats(self, name: str, since: datetime | None = None) -> dict[str, Any]:
        """Get statistics for a metric."""
        metrics = self.get_metrics(name, since=since)
        if not metrics:
            return {}

        values = [m.value for m in metrics]

        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1],
            "unit": metrics[-1].unit,
        }

    def _check_alerts(self, metric_name: str, value: float) -> None:
        """Check alert rules for a metric."""
        for rule in self._alert_rules.values():
            if rule.name in self._alert_cooldowns:
                cooldown_end = self._alert_cooldowns[rule.name]
                if datetime.now(timezone.utc) < cooldown_end:
                    continue

            try:
                if rule.condition(value):
                    self._trigger_alert(rule, metric_name, value)
                    self._alert_cooldowns[rule.name] = datetime.now(timezone.utc) + timedelta(
                        minutes=rule.cooldown_minutes
                    )
            except Exception as e:
                logger.error(f"Error checking alert rule {rule.name}: {e}")

    def _trigger_alert(self, rule: AlertRule, metric_name: str, value: float) -> None:
        """Trigger an alert."""
        alert_data = {
            "rule": rule.name,
            "metric": metric_name,
            "value": value,
            "threshold": getattr(rule.condition, "__name__", "custom"),
            "severity": rule.severity,
            "message": rule.message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }.get(rule.severity, logging.WARNING)

        logger.log(log_level, f"Alert triggered: {rule.message}", extra=alert_data)

        # Here you could integrate with external alerting systems
        # (PagerDuty, Slack, email, etc.)


class PerformanceMonitor:
    """Monitors system and application performance."""

    def __init__(self, metrics_collector: MetricsCollector | None = None):
        self.metrics = metrics_collector or MetricsCollector()
        self._process = psutil.Process()
        self._start_time = time.time()

    def record_system_metrics(self) -> None:
        """Record system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.record_metric("system.cpu_percent", cpu_percent, unit="percent")

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.record_metric("system.memory_percent", memory.percent, unit="percent")
            self.metrics.record_metric("system.memory_used", memory.used / 1024 / 1024, unit="MB")

            # Process-specific metrics
            process_memory = self._process.memory_info()
            self.metrics.record_metric(
                "process.memory_rss", process_memory.rss / 1024 / 1024, unit="MB"
            )

            process_cpu = self._process.cpu_percent()
            self.metrics.record_metric("process.cpu_percent", process_cpu, unit="percent")

            # Uptime
            uptime = time.time() - self._start_time
            self.metrics.record_metric("process.uptime", uptime, unit="seconds")

        except Exception as e:
            logger.error(f"Error recording system metrics: {e}")

    def record_backtest_metrics(
        self,
        backtest_id: str,
        duration: float,
        symbols_count: int,
        bars_count: int,
        trades_count: int,
        total_return: float,
    ) -> None:
        """Record backtest performance metrics."""
        self.metrics.record_metric(
            "backtest.duration", duration, {"backtest_id": backtest_id}, unit="seconds"
        )
        self.metrics.record_metric(
            "backtest.symbols_count", symbols_count, {"backtest_id": backtest_id}
        )
        self.metrics.record_metric("backtest.bars_count", bars_count, {"backtest_id": backtest_id})
        self.metrics.record_metric(
            "backtest.trades_count", trades_count, {"backtest_id": backtest_id}
        )
        self.metrics.record_metric(
            "backtest.total_return", total_return, {"backtest_id": backtest_id}, unit="percent"
        )

        # Performance ratios
        if duration > 0:
            bars_per_second = bars_count / duration
            trades_per_second = trades_count / duration

            self.metrics.record_metric(
                "backtest.bars_per_second", bars_per_second, {"backtest_id": backtest_id}
            )
            self.metrics.record_metric(
                "backtest.trades_per_second", trades_per_second, {"backtest_id": backtest_id}
            )


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


# Global instances
_metrics_collector = MetricsCollector()
_performance_monitor = PerformanceMonitor(_metrics_collector)
_structured_logger = StructuredLogger()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _performance_monitor


def get_structured_logger() -> StructuredLogger:
    """Get the global structured logger."""
    return _structured_logger


# Decorators for performance monitoring


def monitor_performance(operation_name: str | None = None, log_result: bool = True):
    """Decorator to monitor function performance."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            exception = None

            try:
                return func(*args, **kwargs)
            except Exception as e:
                success = False
                exception = e
                raise
            finally:
                duration = time.time() - start_time

                # Record metrics
                metrics = get_metrics_collector()
                metrics.record_metric(
                    f"function.duration.{func.__name__}",
                    duration,
                    {"success": str(success)},
                    unit="seconds",
                )

                # Log performance if requested
                if log_result:
                    logger = get_structured_logger()
                    logger.log_performance(
                        operation=operation_name or func.__name__,
                        duration=duration,
                        success=success,
                        function=func.__name__,
                        args_count=len(args),
                        kwargs_count=len(kwargs),
                        exception=str(exception) if exception else None,
                    )

        return wrapper

    return decorator


def monitor_backtest(backtest_id_param: str = "backtest_id"):
    """Decorator to monitor backtest operations."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Extract backtest ID from parameters
            backtest_id = "unknown"
            if backtest_id_param in kwargs:
                backtest_id = kwargs[backtest_id_param]
            elif len(args) > 0 and hasattr(args[0], backtest_id_param):
                backtest_id = getattr(args[0], backtest_id_param)

            monitor = get_performance_monitor()

            try:
                result = func(*args, **kwargs)

                # Record backtest metrics if result contains relevant data
                if isinstance(result, dict) and "performance" in result:
                    perf = result["performance"]
                    duration = time.time() - start_time

                    symbols_count = len(result.get("symbols", []))
                    bars_count = sum(len(data) for data in result.get("market_data", {}).values())
                    trades_count = len(result.get("trades", []))
                    total_return = perf.get("total_return", 0)

                    monitor.record_backtest_metrics(
                        backtest_id=backtest_id,
                        duration=duration,
                        symbols_count=symbols_count,
                        bars_count=bars_count,
                        trades_count=trades_count,
                        total_return=total_return,
                    )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger = get_structured_logger()
                logger.log_performance(
                    operation=f"backtest.{func.__name__}",
                    duration=duration,
                    success=False,
                    backtest_id=backtest_id,
                    exception=str(e),
                )
                raise

        return wrapper

    return decorator


# Alert rule presets


def create_performance_alerts(metrics_collector: MetricsCollector) -> None:
    """Create standard performance alert rules."""

    # Backtest duration alert
    metrics_collector.add_alert_rule(
        AlertRule(
            name="slow_backtest",
            condition=lambda x: x > 300,  # 5 minutes
            message="Backtest execution took longer than 5 minutes",
            severity="warning",
        )
    )

    # Memory usage alert
    metrics_collector.add_alert_rule(
        AlertRule(
            name="high_memory_usage",
            condition=lambda x: x > 80,  # 80% memory usage
            message="System memory usage above 80%",
            severity="warning",
        )
    )

    # CPU usage alert
    metrics_collector.add_alert_rule(
        AlertRule(
            name="high_cpu_usage",
            condition=lambda x: x > 90,  # 90% CPU usage
            message="System CPU usage above 90%",
            severity="error",
        )
    )

    # Low Sharpe ratio alert
    metrics_collector.add_alert_rule(
        AlertRule(
            name="low_sharpe_ratio",
            condition=lambda x: x < 0.5,  # Sharpe ratio below 0.5
            message="Strategy Sharpe ratio below acceptable threshold",
            severity="info",
        )
    )


# Initialize default alerts
create_performance_alerts(_metrics_collector)
