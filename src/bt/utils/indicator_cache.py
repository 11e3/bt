"""Indicator caching utilities for strategy optimization."""

import functools
import hashlib
from typing import Any

import numpy as np
import pandas as pd


class IndicatorCache:
    """Cache for technical indicators to avoid repeated calculations."""

    def __init__(self, max_size: int = 1000):
        """Initialize indicator cache.

        Args:
            max_size: Maximum number of cached indicators
        """
        self._cache: dict[str, float] = {}
        self._max_size = max_size
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_data_hash(self, data: pd.Series | np.ndarray) -> str:
        """Generate hash for data to detect changes.

        Args:
            data: Price data series

        Returns:
            Hash string representing data content
        """
        # Convert to bytes and hash
        if isinstance(data, pd.Series):
            arr = data.to_numpy()
            data_bytes = arr.tobytes()
        else:
            data_bytes = data.tobytes()

        return hashlib.md5(data_bytes).hexdigest()[:16]

    @functools.lru_cache(maxsize=500)
    def get_sma(
        self, symbol: str, lookback: int, data_hash: str, values: tuple[float, ...]
    ) -> float:
        """Get Simple Moving Average with caching.

        Args:
            symbol: Trading symbol
            lookback: SMA lookback period
            data_hash: Hash of the data
            values: Tuple of price values

        Returns:
            SMA value
        """
        cache_key = f"sma_{symbol}_{lookback}_{data_hash}"

        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        # Calculate SMA
        if len(values) < lookback:
            return 0.0

        sma_value = np.mean(values[-lookback:])

        # Cache the result
        self._cache_misses += 1
        self._cache[cache_key] = float(sma_value)

        # Prevent cache from growing too large
        if len(self._cache) > self._max_size:
            keys_to_remove = list(self._cache.keys())[: self._max_size // 2]
            for key in keys_to_remove:
                del self._cache[key]

        return float(sma_value)

    @functools.lru_cache(maxsize=500)
    def get_ema(
        self, symbol: str, lookback: int, data_hash: str, values: tuple[float, ...]
    ) -> float:
        """Get Exponential Moving Average with caching.

        Args:
            symbol: Trading symbol
            lookback: EMA lookback period
            data_hash: Hash of the data
            values: Tuple of price values

        Returns:
            EMA value
        """
        cache_key = f"ema_{symbol}_{lookback}_{data_hash}"

        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        # Calculate EMA using pandas
        if len(values) < lookback:
            return 0.0

        series = pd.Series(list(values))
        ema_value = series.ewm(span=lookback).mean().iloc[-1]

        # Cache the result
        self._cache_misses += 1
        self._cache[cache_key] = float(ema_value)

        # Prevent cache from growing too large
        if len(self._cache) > self._max_size:
            keys_to_remove = list(self._cache.keys())[: self._max_size // 2]
            for key in keys_to_remove:
                del self._cache[key]

        return float(ema_value)

    @functools.lru_cache(maxsize=500)
    def get_rsi(
        self, symbol: str, lookback: int, data_hash: str, values: tuple[float, ...]
    ) -> float:
        """Get Relative Strength Index with caching.

        Args:
            symbol: Trading symbol
            lookback: RSI lookback period
            data_hash: Hash of the data
            values: Tuple of price values

        Returns:
            RSI value
        """
        cache_key = f"rsi_{symbol}_{lookback}_{data_hash}"

        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        # Calculate RSI
        if len(values) < lookback + 1:
            return 50.0  # Default neutral value

        series = pd.Series(list(values))
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=lookback).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=lookback).mean()

        # Handle division by zero
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi_value = rsi.iloc[-1]

        # Cache the result
        self._cache_misses += 1
        self._cache[cache_key] = float(rsi_value) if not np.isnan(rsi_value) else 50.0

        # Prevent cache from growing too large
        if len(self._cache) > self._max_size:
            keys_to_remove = list(self._cache.keys())[: self._max_size // 2]
            for key in keys_to_remove:
                del self._cache[key]

        return float(rsi_value) if not np.isnan(rsi_value) else 50.0

    def calculate_indicator(
        self, symbol: str, indicator_type: str, lookback: int, data: pd.Series | np.ndarray
    ) -> float:
        """Calculate any cached indicator.

        Args:
            symbol: Trading symbol
            indicator_type: Type of indicator ('sma', 'ema', 'rsi')
            lookback: Lookback period
            data: Price data

        Returns:
            Indicator value
        """
        data_hash = self._get_data_hash(data)
        values = tuple(float(v) for v in data)

        if indicator_type == "sma":
            return self.get_sma(symbol, lookback, data_hash, values)
        if indicator_type == "ema":
            return self.get_ema(symbol, lookback, data_hash, values)
        if indicator_type == "rsi":
            return self.get_rsi(symbol, lookback, data_hash, values)
        raise ValueError(f"Unknown indicator type: {indicator_type}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_size": self._max_size,
        }

    def clear_cache(self) -> None:
        """Clear the indicator cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        # Clear lru_cache
        self.get_sma.cache_clear()
        self.get_ema.cache_clear()
        self.get_rsi.cache_clear()


# Global indicator cache instance
_indicator_cache = IndicatorCache()


def get_indicator_cache() -> IndicatorCache:
    """Get the global indicator cache instance.

    Returns:
        Global IndicatorCache instance
    """
    return _indicator_cache


def clear_global_cache() -> None:
    """Clear the global indicator cache."""
    _indicator_cache.clear_cache()
