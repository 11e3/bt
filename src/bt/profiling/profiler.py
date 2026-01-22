"""Performance profiler implementation."""

from __future__ import annotations

import cProfile
import functools
import logging
import time
import tracemalloc
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import pandas as pd
import psutil

from bt.profiling.stats import PerformanceStats, ProfilingConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Comprehensive performance profiling tool."""

    def __init__(self, config: ProfilingConfig | None = None):
        self.config = config or ProfilingConfig()
        self.stats: dict[str, PerformanceStats] = {}
        self.memory_traces: dict[str, list] = {}
        self._profiler = cProfile.Profile()
        self._memory_tracer = None

        if self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.track_memory_leaks:
            tracemalloc.start()
            self._memory_tracer = tracemalloc

    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile a function."""
        if not self.config.enabled:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self._profile_execution(func, *args, **kwargs)

        return wrapper

    @contextmanager
    def profile_context(self, name: str):
        """Context manager for profiling code blocks."""
        if not self.config.enabled:
            yield
            return

        start_time = time.time()
        memory_before = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            memory_after = self._get_memory_usage()
            execution_time = end_time - start_time

            self._record_stats(name, execution_time, memory_before, memory_after)

    def _profile_execution(self, func: Callable, *args, **kwargs):
        """Execute and profile a function."""
        func_name = f"{func.__module__}.{func.__qualname__}"

        # Memory tracking
        memory_before = self._get_memory_usage() if self.config.profile_memory else 0

        # CPU profiling
        if self.config.profile_cpu:
            self._profiler.enable()

        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.time()
            execution_time = end_time - start_time

            if self.config.profile_cpu:
                self._profiler.disable()

            # Memory tracking
            memory_after = self._get_memory_usage() if self.config.profile_memory else 0

            self._record_stats(func_name, execution_time, memory_before, memory_after)

    def _record_stats(
        self, name: str, execution_time: float, memory_before: int, memory_after: int
    ):
        """Record profiling statistics."""
        if name not in self.stats:
            self.stats[name] = PerformanceStats(function_name=name)

        self.stats[name].update(execution_time, memory_before, memory_after)

        # Check alert thresholds
        self._check_alerts(name, execution_time, memory_after)

    def _check_alerts(self, name: str, execution_time: float, memory_mb: float):
        """Check if performance metrics exceed alert thresholds."""
        alerts = []

        if execution_time > self.config.alert_thresholds["max_execution_time"]:
            alerts.append(f"Execution time {execution_time:.2f}s exceeds threshold")

        if memory_mb > self.config.alert_thresholds["max_memory_mb"]:
            alerts.append(f"Memory usage {memory_mb:.2f}MB exceeds threshold")

        if alerts:
            logger.warning(f"Performance alert for {name}: {'; '.join(alerts)}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB

    def get_stats_report(self) -> pd.DataFrame:
        """Generate performance statistics report."""
        if not self.stats:
            return pd.DataFrame()

        data = [stats.to_dict() for stats in self.stats.values()]
        df = pd.DataFrame(data)

        # Sort by total time descending
        if not df.empty:
            df = df.sort_values("total_time", ascending=False)

        return df

    def save_profile_report(self, filename: str | None = None) -> Path | None:
        """Save profiling report to file."""
        if not self.config.save_results or not self.config.output_dir:
            return None

        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"profile_report_{timestamp}.html"

        output_path = self.config.output_dir / filename

        # Generate HTML report
        report_html = self._generate_html_report()
        output_path.write_text(report_html)

        logger.info(f"Profile report saved to {output_path}")
        return output_path

    def _generate_html_report(self) -> str:
        """Generate HTML profiling report."""
        df = self.get_stats_report()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BT Framework Performance Profile Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .alert {{ color: red; font-weight: bold; }}
                .warning {{ color: orange; }}
            </style>
        </head>
        <body>
            <h1>BT Framework Performance Profile Report</h1>
            <p>Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>

            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <th>Function</th>
                    <th>Calls</th>
                    <th>Total Time (s)</th>
                    <th>Avg Time (s)</th>
                    <th>Min Time (s)</th>
                    <th>Max Time (s)</th>
                    <th>Memory Peak (MB)</th>
                    <th>Memory Delta (MB)</th>
                </tr>
        """

        for _, row in df.iterrows():
            html += f"""
                <tr>
                    <td>{row["function_name"]}</td>
                    <td>{row["calls"]}</td>
                    <td>{row["total_time"]:.4f}</td>
                    <td>{row["avg_time"]:.4f}</td>
                    <td>{row["min_time"]:.4f}</td>
                    <td>{row["max_time"]:.4f}</td>
                    <td>{row["memory_peak"]:.2f}</td>
                    <td>{row["memory_delta"]:.2f}</td>
                </tr>
            """

        html += """
            </table>

            <h2>System Information</h2>
            <ul>
        """

        try:
            import platform

            html += f"<li>Platform: {platform.platform()}</li>"
            html += f"<li>Python: {platform.python_version()}</li>"
        except Exception:
            pass

        html += f"<li>CPU profiling: {'Enabled' if self.config.profile_cpu else 'Disabled'}</li>"
        html += (
            f"<li>Memory profiling: {'Enabled' if self.config.profile_memory else 'Disabled'}</li>"
        )

        html += """
            </ul>
        </body>
        </html>
        """

        return html

    def detect_memory_leaks(self) -> list[dict[str, Any]]:
        """Detect potential memory leaks."""
        if not self.config.track_memory_leaks or not self._memory_tracer:
            return []

        # Get memory snapshots
        snapshots = self._memory_tracer.take_snapshot()
        top_stats = snapshots.statistics("lineno")

        leaks = []
        for stat in top_stats[:10]:  # Top 10 memory consumers
            if stat.size > self.config.alert_thresholds["memory_leak_threshold"] * 1024 * 1024:
                leaks.append(
                    {
                        "file": stat.traceback[0].filename,
                        "line": stat.traceback[0].lineno,
                        "size_mb": stat.size / (1024 * 1024),
                        "count": stat.count,
                    }
                )

        return leaks

    def reset(self):
        """Reset all profiling data."""
        self.stats.clear()
        self.memory_traces.clear()
