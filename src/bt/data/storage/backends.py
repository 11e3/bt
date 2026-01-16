"""Storage backend implementations."""

import hashlib
import json
import logging
import pickle
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Available storage backends."""

    MEMORY = "memory"
    JSON = "json"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    SQLITE = "sqlite"
    PICKLE = "pickle"


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

    def store(self, key: str, data: Any, **_kwargs) -> bool:
        with self._lock:
            self._data[key] = data
            return True

    def retrieve(self, key: str, **_kwargs) -> Any | None:
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
                    key = file_path.stem
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

    def store(self, key: str, data: Any, **_kwargs) -> bool:
        try:
            import sqlite3

            with sqlite3.connect(str(self.db_path)) as conn:
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

    def retrieve(self, key: str, **_kwargs) -> Any | None:
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
