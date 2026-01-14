"""Decimal caching utilities for performance optimization."""

from decimal import Decimal


class DecimalCache:
    """Cache common Decimal values to avoid repeated conversions."""

    _cache: dict[str | int | float, Decimal] = {}
    _max_size = 10000

    @classmethod
    def get(cls, value: str | int | float | Decimal) -> Decimal:
        """Get Decimal from cache, creating and caching if not present.

        Args:
            value: Value to convert to Decimal

        Returns:
            Cached Decimal value
        """
        # Handle Decimal types directly
        if isinstance(value, Decimal):
            return value

        if value not in cls._cache:
            cls._cache[value] = Decimal(str(value))

            # Prevent cache from growing too large
            if len(cls._cache) > cls._max_size:
                # Clear oldest half of cache when limit reached
                keys_to_remove = list(cls._cache.keys())[: cls._max_size // 2]
                for key in keys_to_remove:
                    del cls._cache[key]

        return cls._cache[value]

    @classmethod
    def clear(cls) -> None:
        """Clear the cache."""
        cls._cache.clear()

    @classmethod
    def size(cls) -> int:
        """Get current cache size."""
        return len(cls._cache)


# Convenience function for common usage
def get_decimal(value: str | int | float) -> Decimal:
    """Convenience function to get cached Decimal."""
    return DecimalCache.get(value)
