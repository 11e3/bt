"""Performance monitoring and metrics collection system.

Provides comprehensive monitoring capabilities including:
- Structured logging with context
- Performance metrics collection
- Alerting framework
- Prometheus-style metrics export
- Performance profiling and tracing
"""

from bt.monitoring.alerts import create_performance_alerts
from bt.monitoring.decorators import monitor_backtest, monitor_performance
from bt.monitoring.exporters import (
    EmailAlerter,
    JSONMetricsExporter,
    MetricsServer,
    PrometheusExporter,
    SlackAlerter,
    start_metrics_server,
    stop_metrics_server,
)
from bt.monitoring.logger import StructuredLogger
from bt.monitoring.metrics import AlertRule, MetricsCollector, MetricValue
from bt.monitoring.monitor import PerformanceMonitor

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


def get_monitor() -> PerformanceMonitor:
    """Alias for get_performance_monitor()."""
    return _performance_monitor


# Initialize default alerts
create_performance_alerts(_metrics_collector)

__all__ = [
    # Metrics
    "MetricValue",
    "AlertRule",
    "MetricsCollector",
    # Monitor
    "PerformanceMonitor",
    # Logger
    "StructuredLogger",
    # Decorators
    "monitor_performance",
    "monitor_backtest",
    # Alerts
    "create_performance_alerts",
    # Exporters
    "PrometheusExporter",
    "JSONMetricsExporter",
    "MetricsServer",
    "EmailAlerter",
    "SlackAlerter",
    "start_metrics_server",
    "stop_metrics_server",
    # Global accessors
    "get_metrics_collector",
    "get_performance_monitor",
    "get_structured_logger",
    "get_monitor",
]
