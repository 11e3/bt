"""
Data abstraction layer for flexible storage backends and intelligent caching.

Provides unified interface for different storage backends including:
- In-memory storage
- File-based storage (JSON, Parquet, HDF5)
- Database storage (SQLite, PostgreSQL)
- Intelligent caching with TTL and size limits
"""

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import numpy as np
import pandas as pd

from ...interfaces.core import DataError, ValidationError

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Available storage backends."""

    MEMORY = "memory"
    JSON = "json"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    SQLITE = "sqlite"
    PICKLE = "pickle"


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


class IStorageBackend(ABC):
    """Interface for storage backends."""

    @abstractmethod
    def store(self, key: str, data: Any, **kwargs) -> bool:
        """Store data with given key."""
        pass

    @abstractmethod
    def retrieve(self, key: str, **kwargs) -> Any | None:
        """Retrieve data by key."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data by key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    def list_keys(self, pattern: str | None = None) -> list[str]:
        """List all keys, optionally filtered by pattern."""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all data."""
        pass


class MemoryStorageBackend(IStorageBackend):
    """In-memory storage backend."""

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._lock = threading.RLock()

    def store(self, key: str, data: Any, **kwargs) -> bool:  # noqa: ARG002
        with self._lock:
            self._data[key] = data
            return True

    def retrieve(self, key: str, **kwargs) -> Any | None:  # noqa: ARG002
        with self._lock:
            return self._data.get(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def list_keys(self, pattern: str | None = None) -> list[str]:
        with self._lock:
            keys = list(self._data.keys())
            if pattern:
                import re

                regex = re.compile(pattern)
                keys = [k for k in keys if regex.match(k)]
            return keys

    def clear(self) -> bool:
        with self._lock:
            self._data.clear()
            return True


class FileStorageBackend(IStorageBackend):
    """File-based storage backend."""

    def __init__(self, base_path: Path, backend_type: StorageBackend):
        self.base_path = Path(base_path)
        self.backend_type = backend_type
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, key: str) -> Path:
        """Get file path for key."""
        # Sanitize key for filename
        safe_key = "".join(c for c in key if c.isalnum() or c in "._-").strip()
        if not safe_key:
            safe_key = hashlib.md5(key.encode()).hexdigest()

        if self.backend_type == StorageBackend.JSON:
            return self.base_path / f"{safe_key}.json"
        if self.backend_type == StorageBackend.PARQUET:
            return self.base_path / f"{safe_key}.parquet"
        if self.backend_type == StorageBackend.HDF5:
            return self.base_path / f"{safe_key}.h5"
        if self.backend_type == StorageBackend.PICKLE:
            return self.base_path / f"{safe_key}.pkl"
        raise ValueError(f"Unsupported backend type: {self.backend_type}")

    def store(self, key: str, data: Any, **kwargs) -> bool:
        try:
            file_path = self._get_file_path(key)

            if self.backend_type == StorageBackend.JSON:
                if isinstance(data, pd.DataFrame):
                    data.to_json(file_path, orient="records", date_format="iso")
                else:
                    with file_path.open("w") as f:
                        json.dump(data, f, default=str, indent=2)
            elif self.backend_type == StorageBackend.PARQUET:
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(file_path, **kwargs)
                else:
                    raise ValueError("Parquet backend requires DataFrame data")
            elif self.backend_type == StorageBackend.HDF5:
                if isinstance(data, pd.DataFrame):
                    data.to_hdf(file_path, key="data", mode="w", **kwargs)
                else:
                    raise ValueError("HDF5 backend requires DataFrame data")
            elif self.backend_type == StorageBackend.PICKLE:
                with file_path.open("wb") as f:
                    pickle.dump(data, f)

            return True
        except Exception as e:
            logger.error(f"Failed to store {key}: {e}")
            return False

    def retrieve(self, key: str, **kwargs) -> Any | None:
        try:
            file_path = self._get_file_path(key)
            if not file_path.exists():
                return None

            if self.backend_type == StorageBackend.JSON:
                with file_path.open() as f:
                    return json.load(f)
            elif self.backend_type == StorageBackend.PARQUET:
                return pd.read_parquet(file_path, **kwargs)
            elif self.backend_type == StorageBackend.HDF5:
                return pd.read_hdf(file_path, key="data", **kwargs)
            elif self.backend_type == StorageBackend.PICKLE:
                with file_path.open("rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to retrieve {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        file_path = self._get_file_path(key)
        return file_path.exists()

    def list_keys(self, pattern: str | None = None) -> list[str]:
        try:
            keys = []
            for file_path in self.base_path.glob("*"):
                if file_path.is_file():
                    # Extract key from filename
                    stem = file_path.stem
                    # Remove extension and reconstruct key
                    key = stem
                    if pattern:
                        import re

                        if re.match(pattern, key):
                            keys.append(key)
                    else:
                        keys.append(key)
            return keys
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return []

    def clear(self) -> bool:
        try:
            for file_path in self.base_path.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to clear storage: {e}")
            return False


class DatabaseStorageBackend(IStorageBackend):
    """Database storage backend using SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        try:
            import sqlite3

            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS storage (
                        key TEXT PRIMARY KEY,
                        data BLOB,
                        data_type TEXT,
                        created_at REAL,
                        updated_at REAL
                    )
                """)
                conn.commit()
        except ImportError:
            logger.warning("SQLite not available, database backend disabled")

    def store(self, key: str, data: Any, **kwargs) -> bool:  # noqa: ARG002
        try:
            import sqlite3

            with sqlite3.connect(str(self.db_path)) as conn:
                # Serialize data
                data_blob = pickle.dumps(data)
                data_type = type(data).__name__

                now = time.time()
                conn.execute(
                    """
                    INSERT OR REPLACE INTO storage
                    (key, data, data_type, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (key, data_blob, data_type, now, now),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to store {key}: {e}")
            return False

    def retrieve(self, key: str, **kwargs) -> Any | None:  # noqa: ARG002
        try:
            import sqlite3

            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT data FROM storage WHERE key = ?", (key,))
                row = cursor.fetchone()
                if row:
                    return pickle.loads(row[0])
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        try:
            import sqlite3

            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("DELETE FROM storage WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        try:
            import sqlite3

            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT 1 FROM storage WHERE key = ? LIMIT 1", (key,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check existence of {key}: {e}")
            return False

    def list_keys(self, pattern: str | None = None) -> list[str]:
        try:
            import sqlite3

            with sqlite3.connect(str(self.db_path)) as conn:
                if pattern:
                    # Simple LIKE pattern matching
                    like_pattern = pattern.replace("*", "%").replace("?", "_")
                    cursor = conn.execute(
                        "SELECT key FROM storage WHERE key LIKE ?", (like_pattern,)
                    )
                else:
                    cursor = conn.execute("SELECT key FROM storage")

                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to list keys: {e}")
            return []

    def clear(self) -> bool:
        try:
            import sqlite3

            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("DELETE FROM storage")
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to clear storage: {e}")
            return False


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
            # Calculate data size
            size_bytes = self._calculate_size(data)

            # Check if we need to evict
            if self._total_size + size_bytes > self.config.max_size_mb * 1024 * 1024:
                self._evict_entries(size_bytes)

            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Add new entry
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
            # Least Recently Used
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)
        elif self.config.eviction_policy == "lfu":
            # Least Frequently Used
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].access_count)
        elif self.config.eviction_policy == "fifo":
            # First In First Out
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].timestamp)
        else:
            # Default to LRU
            sorted_entries = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)

        # Remove entries until we have enough space
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


class DataManager:
    """Unified data management with multiple storage backends and caching."""

    def __init__(
        self,
        cache_config: CacheConfig | None = None,
        default_backend: StorageBackend = StorageBackend.MEMORY,
    ):
        self.cache_config = cache_config or CacheConfig()
        self.default_backend = default_backend
        self.cache = IntelligentCache(self.cache_config)
        self.backends: dict[StorageBackend, IStorageBackend] = {}

        # Initialize default backends
        self._init_default_backends()

    def _init_default_backends(self):
        """Initialize default storage backends."""
        # Memory backend (always available)
        self.backends[StorageBackend.MEMORY] = MemoryStorageBackend()

        # File-based backends
        data_dir = Path.home() / ".bt_data"
        self.backends[StorageBackend.JSON] = FileStorageBackend(
            data_dir / "json", StorageBackend.JSON
        )
        self.backends[StorageBackend.PARQUET] = FileStorageBackend(
            data_dir / "parquet", StorageBackend.PARQUET
        )
        self.backends[StorageBackend.HDF5] = FileStorageBackend(
            data_dir / "hdf5", StorageBackend.HDF5
        )
        self.backends[StorageBackend.PICKLE] = FileStorageBackend(
            data_dir / "pickle", StorageBackend.PICKLE
        )

        # Database backend
        self.backends[StorageBackend.SQLITE] = DatabaseStorageBackend(data_dir / "bt_data.db")

    def add_backend(self, backend_type: StorageBackend, backend: IStorageBackend):
        """Add custom storage backend."""
        self.backends[backend_type] = backend

    def store(
        self,
        key: str,
        data: Any,
        backend: StorageBackend | None = None,
        cache: bool = True,
        ttl: float | None = None,
        **kwargs,
    ) -> bool:
        """Store data with optional caching."""
        backend = backend or self.default_backend
        backend_instance = self.backends.get(backend)

        if backend_instance is None:
            raise DataError(f"Storage backend {backend} not available")

        # Store in backend
        success = backend_instance.store(key, data, **kwargs)
        if not success:
            return False

        # Cache if enabled
        if cache and self.cache_config.enabled:
            self.cache.put(key, data, ttl)

        return True

    def retrieve(
        self, key: str, backend: StorageBackend | None = None, use_cache: bool = True, **kwargs
    ) -> Any | None:
        """Retrieve data with cache-first lookup."""
        # Try cache first
        if use_cache and self.cache_config.enabled:
            cached_data = self.cache.get(key)
            if cached_data is not None:
                return cached_data

        # Get from backend
        backend = backend or self.default_backend
        backend_instance = self.backends.get(backend)

        if backend_instance is None:
            raise DataError(f"Storage backend {backend} not available")

        data = backend_instance.retrieve(key, **kwargs)

        # Cache retrieved data
        if data is not None and use_cache and self.cache_config.enabled:
            self.cache.put(key, data)

        return data

    def delete(self, key: str, backend: StorageBackend | None = None) -> bool:
        """Delete data from storage and cache."""
        # Remove from cache
        self.cache.invalidate(key)

        # Remove from backend
        backend = backend or self.default_backend
        backend_instance = self.backends.get(backend)

        if backend_instance is None:
            return False

        return backend_instance.delete(key)

    def exists(self, key: str, backend: StorageBackend | None = None) -> bool:
        """Check if key exists in storage."""
        backend = backend or self.default_backend
        backend_instance = self.backends.get(backend)

        if backend_instance is None:
            return False

        return backend_instance.exists(key)

    def list_keys(
        self, pattern: str | None = None, backend: StorageBackend | None = None
    ) -> list[str]:
        """List keys in storage."""
        backend = backend or self.default_backend
        backend_instance = self.backends.get(backend)

        if backend_instance is None:
            return []

        return backend_instance.list_keys(pattern)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        return self.cache.get_stats()

    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()

    def backup_data(
        self, target_backend: StorageBackend, source_backends: list[StorageBackend] | None = None
    ) -> bool:
        """Backup data from one backend to another."""
        if source_backends is None:
            source_backends = [self.default_backend]

        target = self.backends.get(target_backend)
        if target is None:
            return False

        success = True
        for source_type in source_backends:
            source = self.backends.get(source_type)
            if source is None:
                continue

            try:
                keys = source.list_keys()
                for key in keys:
                    data = source.retrieve(key)
                    if data is not None:
                        target.store(f"backup_{source_type.value}_{key}", data)
            except Exception as e:
                logger.error(f"Failed to backup from {source_type}: {e}")
                success = False

        return success


# Global data manager instance
_data_manager: DataManager | None = None


def get_data_manager() -> DataManager:
    """Get global data manager instance."""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager


def store_data(key: str, data: Any, backend: StorageBackend | None = None, **kwargs) -> bool:
    """Convenience function to store data."""
    return get_data_manager().store(key, data, backend, **kwargs)


def retrieve_data(key: str, backend: StorageBackend | None = None, **kwargs) -> Any | None:
    """Convenience function to retrieve data."""
    return get_data_manager().retrieve(key, backend, **kwargs)
