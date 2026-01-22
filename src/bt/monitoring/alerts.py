"""Alert rule presets."""

from __future__ import annotations

from bt.monitoring.metrics import AlertRule, MetricsCollector


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
