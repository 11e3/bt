"""Metrics collection and alerting."""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from bt.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

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
