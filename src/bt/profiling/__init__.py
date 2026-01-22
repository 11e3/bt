"""
Performance profiling and code quality analysis tools.

Provides comprehensive profiling capabilities including:
- CPU and memory profiling
- Function-level performance analysis
- Memory leak detection
- Code complexity analysis
- Performance regression detection
"""

from pathlib import Path

import pandas as pd

from bt.profiling.analyzer import CodeQualityAnalyzer
from bt.profiling.profiler import PerformanceProfiler
from bt.profiling.stats import PerformanceStats, ProfilingConfig

# Global profiler instance
_profiler: PerformanceProfiler | None = None


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler


def profile_function(func):
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


__all__ = [
    # Stats
    "PerformanceStats",
    "ProfilingConfig",
    # Profiler
    "PerformanceProfiler",
    # Analyzer
    "CodeQualityAnalyzer",
    # Convenience functions
    "get_profiler",
    "profile_function",
    "profile_context",
    "run_quality_analysis",
    "start_profiling",
    "stop_profiling",
    "get_performance_stats",
]
