"""Performance monitoring."""

from __future__ import annotations

import time

import psutil

from bt.monitoring.metrics import MetricsCollector
from bt.utils.logging import get_logger

logger = get_logger(__name__)


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
