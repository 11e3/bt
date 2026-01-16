"""Intelligent caching with TTL, size limits, and eviction policies."""

import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    data: Any
    timestamp: float
    ttl: float | None = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl

    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheConfig:
    """Configuration for caching behavior."""

    enabled: bool = True
    max_size_mb: float = 100.0  # 100MB default
    default_ttl_seconds: float | None = 3600  # 1 hour
    eviction_policy: str = "lru"  # lru, lfu, fifo
    compression: bool = False


class IntelligentCache:
    """Intelligent caching with TTL, size limits, and eviction policies."""

    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._total_size = 0

    def get(self, key: str) -> Any | None:
        """Get cached item."""
        if not self.config.enabled:
            return None

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            if entry.is_expired():
                self._remove_entry(key)
                return None

            entry.touch()
            return entry.data

    def put(self, key: str, data: Any, ttl: float | None = None) -> bool:
        """Put item in cache."""
        if not self.config.enabled:
            return False

        with self._lock:
            size_bytes = self._calculate_size(data)

            if self._total_size + size_bytes > self.config.max_size_mb * 1024 * 1024:
                self._evict_entries(size_bytes)

            if key in self._cache:
                self._remove_entry(key)

            ttl = ttl or self.config.default_ttl_seconds
            entry = CacheEntry(data=data, timestamp=time.time(), ttl=ttl, size_bytes=size_bytes)

            self._cache[key] = entry
            self._total_size += size_bytes
            return True

    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        with self._lock:
            return self._remove_entry(key)

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_size = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(1 for entry in self._cache.values() if entry.is_expired())
            hit_rate = sum(entry.access_count for entry in self._cache.values()) / max(
                total_entries, 1
            )

            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "total_size_mb": self._total_size / (1024 * 1024),
                "max_size_mb": self.config.max_size_mb,
                "average_hit_rate": hit_rate,
                "eviction_policy": self.config.eviction_policy,
            }

    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            if isinstance(data, np.ndarray):
                return data.nbytes
            if isinstance(data, (list, tuple)):
                return sum(self._calculate_size(item) for item in data)
            if isinstance(data, dict):
                return sum(len(str(k)) + self._calculate_size(v) for k, v in data.items())
            return len(pickle.dumps(data))
        except Exception:
            return 1024  # Default 1KB estimate

    def _evict_entries(self, required_space: int):
        """Evict entries based on policy to free up space."""
        if self.config.eviction_policy == "lru":
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)
        elif self.config.eviction_policy == "lfu":
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].access_count)
        elif self.config.eviction_policy == "fifo":
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].timestamp)
        else:
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)

        target_size = self.config.max_size_mb * 1024 * 1024 - required_space
        for key, _entry in sorted_entries:
            if self._total_size <= target_size:
                break
            self._remove_entry(key)

    def _remove_entry(self, key: str) -> bool:
        """Remove cache entry."""
        if key in self._cache:
            entry = self._cache[key]
            self._total_size -= entry.size_bytes
            del self._cache[key]
            return True
        return False
