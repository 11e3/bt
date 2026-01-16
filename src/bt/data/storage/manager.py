"""Unified data management with multiple storage backends and caching."""

import logging
from pathlib import Path
from typing import Any

from bt.interfaces.core import DataError

from .backends import (
    DatabaseStorageBackend,
    FileStorageBackend,
    IStorageBackend,
    MemoryStorageBackend,
    StorageBackend,
)
from .cache import CacheConfig, IntelligentCache

logger = logging.getLogger(__name__)


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

        self._init_default_backends()

    def _init_default_backends(self):
        """Initialize default storage backends."""
        self.backends[StorageBackend.MEMORY] = MemoryStorageBackend()

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

        success = backend_instance.store(key, data, **kwargs)
        if not success:
            return False

        if cache and self.cache_config.enabled:
            self.cache.put(key, data, ttl)

        return True

    def retrieve(
        self, key: str, backend: StorageBackend | None = None, use_cache: bool = True, **kwargs
    ) -> Any | None:
        """Retrieve data with cache-first lookup."""
        if use_cache and self.cache_config.enabled:
            cached_data = self.cache.get(key)
            if cached_data is not None:
                return cached_data

        backend = backend or self.default_backend
        backend_instance = self.backends.get(backend)

        if backend_instance is None:
            raise DataError(f"Storage backend {backend} not available")

        data = backend_instance.retrieve(key, **kwargs)

        if data is not None and use_cache and self.cache_config.enabled:
            self.cache.put(key, data)

        return data

    def delete(self, key: str, backend: StorageBackend | None = None) -> bool:
        """Delete data from storage and cache."""
        self.cache.invalidate(key)

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
