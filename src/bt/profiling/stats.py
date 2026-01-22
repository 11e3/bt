"""Performance statistics dataclasses."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
