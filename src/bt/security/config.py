"""Security configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SecurityConfig:
    """Configuration for security features."""

    # Input validation settings
    max_data_size: int = 100_000_000  # 100MB max data size
    max_string_length: int = 10_000  # Max string length
    max_list_items: int = 10_000  # Max list items
    max_dict_keys: int = 1_000  # Max dict keys

    # File security
    allowed_extensions: list[str] = field(
        default_factory=lambda: [".csv", ".json", ".parquet", ".pkl", ".h5", ".feather"]
    )
    blocked_paths: list[str] = field(
        default_factory=lambda: ["/etc", "/proc", "/sys", "/dev", "/boot", "/root", "/home"]
    )

    # Data validation
    numeric_bounds: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "price": {"min": 0.00000001, "max": 1_000_000_000_000},  # 1 trillion (KRW)
            "volume": {"min": 0, "max": 1_000_000_000},
            "returns": {"min": -1.0, "max": 10.0},  # Allow up to 1000% returns
            "fee_rate": {"min": 0.0, "max": 0.1},  # Max 10% fee
            "slippage_rate": {"min": 0.0, "max": 0.1},  # Max 10% slippage
        }
    )

    # Strategy parameter validation
    strategy_params: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "lookback": {"type": int, "min": 1, "max": 500},
            "multiplier": {"type": float, "min": 0.1, "max": 10.0},
            "k_factor": {"type": float, "min": 0.0, "max": 5.0},
            "top_n": {"type": int, "min": 1, "max": 100},
            "threshold": {"type": float, "min": -1.0, "max": 1.0},
            "window": {"type": int, "min": 1, "max": 1000},
        }
    )
