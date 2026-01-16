"""
Data abstraction layer for flexible storage backends and intelligent caching.

Provides unified interface for different storage backends including:
- In-memory storage
- File-based storage (JSON, Parquet, HDF5)
- Database storage (SQLite, PostgreSQL)
- Intelligent caching with TTL and size limits
"""

from .backends import (
    DatabaseStorageBackend,
    FileStorageBackend,
    IStorageBackend,
    MemoryStorageBackend,
    StorageBackend,
)
from .cache import CacheConfig, CacheEntry, IntelligentCache
from .manager import DataManager, get_data_manager, retrieve_data, store_data

__all__ = [
    # Backends
    "StorageBackend",
    "IStorageBackend",
    "MemoryStorageBackend",
    "FileStorageBackend",
    "DatabaseStorageBackend",
    # Cache
    "CacheConfig",
    "CacheEntry",
    "IntelligentCache",
    # Manager
    "DataManager",
    "get_data_manager",
    "store_data",
    "retrieve_data",
]
