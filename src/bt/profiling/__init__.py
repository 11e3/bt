"""
Performance profiling and code quality analysis tools.

Provides comprehensive profiling capabilities including:
- CPU and memory profiling
- Function-level performance analysis
- Memory leak detection
- Code complexity analysis
- Performance regression detection
"""

import cProfile
import functools
import inspect
import io
import logging
import pstats
import time
import tracemalloc
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """Performance statistics for a profiled function."""

    function_name: str
    calls: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    memory_peak: int = 0
    memory_delta: int = 0
    timestamp: float = field(default_factory=time.time)

    def update(self, execution_time: float, memory_before: int, memory_after: int):
        """Update statistics with new measurement."""
        self.calls += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.calls
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.memory_peak = max(self.memory_peak, memory_after)
        self.memory_delta = memory_after - memory_before

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "function_name": self.function_name,
            "calls": self.calls,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "memory_peak": self.memory_peak,
            "memory_delta": self.memory_delta,
            "timestamp": self.timestamp,
        }


@dataclass
class ProfilingConfig:
    """Configuration for profiling."""

    enabled: bool = True
    profile_cpu: bool = True
    profile_memory: bool = True
    track_memory_leaks: bool = True
    output_dir: Path = Path("profiling_output")
    save_results: bool = True
    alert_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "max_execution_time": 5.0,  # seconds
            "max_memory_mb": 100.0,  # MB
            "memory_leak_threshold": 10.0,  # MB
        }
    )


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
        except:
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


class CodeQualityAnalyzer:
    """Code quality analysis tool."""

    def __init__(self):
        self.metrics: dict[str, Any] = {}

    def analyze_file(self, file_path: Path) -> dict[str, Any]:
        """Analyze code quality metrics for a file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            return self._analyze_code(content, str(file_path))
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return {}

    def analyze_directory(self, directory: Path) -> dict[str, Any]:
        """Analyze code quality for all Python files in directory."""
        results = {}

        for py_file in directory.rglob("*.py"):
            if not self._should_analyze_file(py_file):
                continue

            metrics = self.analyze_file(py_file)
            if metrics:
                results[str(py_file)] = metrics

        return results

    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if file should be analyzed."""
        # Skip common exclusions
        exclusions = {
            "__pycache__",
            ".git",
            "node_modules",
            "build",
            "dist",
            "venv",
            ".env",
            ".tox",
            "migrations",
        }

        return all(not (part in exclusions or part.startswith(".")) for part in file_path.parts)

    def _analyze_code(self, content: str, filename: str) -> dict[str, Any]:
        """Analyze code quality metrics."""
        lines = content.split("\n")
        total_lines = len(lines)

        # Basic metrics
        metrics = {
            "filename": filename,
            "total_lines": total_lines,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "functions": 0,
            "classes": 0,
            "complexity_score": 0,
            "avg_function_length": 0,
            "max_function_length": 0,
        }

        # Analyze each line
        functions = []
        current_function_lines = 0
        in_function = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                metrics["blank_lines"] += 1
            elif stripped.startswith("#"):
                metrics["comment_lines"] += 1
            elif stripped.startswith("def "):
                metrics["functions"] += 1
                if in_function:
                    functions.append(current_function_lines)
                current_function_lines = 0
                in_function = True
            elif stripped.startswith("class "):
                metrics["classes"] += 1
                if in_function:
                    functions.append(current_function_lines)
                    current_function_lines = 0
                    in_function = False
            else:
                metrics["code_lines"] += 1
                if in_function:
                    current_function_lines += 1

                # Simple complexity indicators
                if any(
                    keyword in stripped
                    for keyword in ["if ", "elif ", "else:", "for ", "while ", "try:", "except "]
                ):
                    metrics["complexity_score"] += 1

        if in_function:
            functions.append(current_function_lines)

        # Calculate function metrics
        if functions:
            metrics["avg_function_length"] = sum(functions) / len(functions)
            metrics["max_function_length"] = max(functions)

        # Calculate comment ratio
        total_code_and_comments = metrics["code_lines"] + metrics["comment_lines"]
        metrics["comment_ratio"] = (
            metrics["comment_lines"] / total_code_and_comments if total_code_and_comments > 0 else 0
        )

        return metrics

    def generate_quality_report(self, results: dict[str, Any]) -> str:
        """Generate code quality report."""
        if not results:
            return "No files analyzed."

        # Aggregate metrics
        total_files = len(results)
        total_lines = sum(m.get("total_lines", 0) for m in results.values())
        total_functions = sum(m.get("functions", 0) for m in results.values())
        avg_comment_ratio = sum(m.get("comment_ratio", 0) for m in results.values()) / total_files

        # Find files with issues
        complex_files = [(f, m) for f, m in results.items() if m.get("complexity_score", 0) > 50]

        long_functions = [
            (f, m) for f, m in results.items() if m.get("max_function_length", 0) > 50
        ]

        report = f"""
Code Quality Analysis Report
============================

Summary:
- Total files analyzed: {total_files}
- Total lines of code: {total_lines}
- Total functions: {total_functions}
- Average comment ratio: {avg_comment_ratio:.2%}

Potential Issues:
"""

        if complex_files:
            report += f"\nHighly complex files (>50 complexity score): {len(complex_files)}\n"
            for filename, metrics in complex_files[:5]:  # Show top 5
                report += f"- {filename}: {metrics.get('complexity_score', 0)} complexity score\n"

        if long_functions:
            report += f"\nFunctions with high line count (>50 lines): {len(long_functions)}\n"
            for filename, metrics in long_functions[:5]:  # Show top 5
                report += f"- {filename}: {metrics.get('max_function_length', 0)} max lines\n"

        if not complex_files and not long_functions:
            report += "\n✅ No major code quality issues detected."

        return report


# Global profiler instance
_profiler: PerformanceProfiler | None = None


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function performance."""
    return get_profiler().profile_function(func)


def profile_context(name: str):
    """Context manager for profiling code blocks."""
    return get_profiler().profile_context(name)


def run_quality_analysis(path: Path = Path()) -> str:
    """Run code quality analysis on directory."""
    analyzer = CodeQualityAnalyzer()
    results = analyzer.analyze_directory(path)
    return analyzer.generate_quality_report(results)


# Convenience functions
def start_profiling():
    """Start performance profiling."""
    global _profiler
    _profiler = PerformanceProfiler()


def stop_profiling() -> Path | None:
    """Stop profiling and save report."""
    if _profiler:
        return _profiler.save_profile_report()
    return None


def get_performance_stats() -> pd.DataFrame:
    """Get current performance statistics."""
    return get_profiler().get_stats_report()
