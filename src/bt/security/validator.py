"""Input validation and sanitization."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from bt.exceptions import SecurityError
from bt.interfaces.core import ValidationError
from bt.security.config import SecurityConfig

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation and sanitization."""

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()
        self._validators: dict[str, Callable] = {
            "dataframe": self._validate_dataframe,
            "dict": self._validate_dict,
            "list": self._validate_list,
            "string": self._validate_string,
            "numeric": self._validate_numeric,
            "file_path": self._validate_file_path,
            "symbol": self._validate_symbol,
            "strategy_config": self._validate_strategy_config,
        }

    def validate(self, data: Any, data_type: str, **kwargs) -> Any:
        """Validate and sanitize input data."""
        if data_type not in self._validators:
            raise ValidationError(f"Unknown data type: {data_type}")

        try:
            return self._validators[data_type](data, **kwargs)
        except Exception as e:
            logger.error(f"Validation failed for {data_type}: {e}")
            raise ValidationError(f"Invalid {data_type}: {str(e)}") from e

    def _validate_dataframe(self, df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
        """Validate pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")

        # Check size limits
        memory_usage = df.memory_usage(deep=True).sum()
        if memory_usage > self.config.max_data_size:
            raise SecurityError(f"DataFrame too large: {memory_usage} bytes")

        # Validate required columns for OHLCV data
        required_cols = {"datetime", "open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValidationError(f"Missing required columns: {missing}")

        # Validate data types
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValidationError(f"Column {col} must be numeric")

        # Check for infinite or NaN values
        for col in numeric_cols:
            if df[col].isna().any():
                raise ValidationError(f"Column {col} contains NaN values")
            if np.isinf(df[col]).any():
                raise ValidationError(f"Column {col} contains infinite values")

        # Validate OHLC relationships
        if not ((df["low"] <= df["open"]) & (df["open"] <= df["high"])).all():
            raise ValidationError("Invalid OHLC relationships for open prices")
        if not ((df["low"] <= df["close"]) & (df["close"] <= df["high"])).all():
            raise ValidationError("Invalid OHLC relationships for close prices")

        # Sanitize datetime column
        if "datetime" in df.columns:
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            if df["datetime"].isna().any():
                raise ValidationError("Invalid datetime values found")

        return df

    def _validate_dict(self, data: dict, **_kwargs) -> dict:
        """Validate dictionary input."""
        if not isinstance(data, dict):
            raise ValidationError("Input must be a dictionary")

        if len(data) > self.config.max_dict_keys:
            raise SecurityError(f"Dictionary too large: {len(data)} keys")

        # Sanitize keys
        sanitized = {}
        for key, value in data.items():
            if not isinstance(key, str):
                raise ValidationError("Dictionary keys must be strings")
            if len(key) > self.config.max_string_length:
                raise ValidationError(f"Key too long: {key}")

            # Validate/sanitize value recursively if it's a nested dict/list
            if isinstance(value, dict):
                sanitized[key] = self._validate_dict(value)
            elif isinstance(value, list):
                sanitized[key] = self._validate_list(value)
            else:
                sanitized[key] = self._sanitize_value(value)

        return sanitized

    def _validate_list(self, data: list, **_kwargs) -> list:
        """Validate list input."""
        if not isinstance(data, list):
            raise ValidationError("Input must be a list")

        if len(data) > self.config.max_list_items:
            raise SecurityError(f"List too large: {len(data)} items")

        # Sanitize each item
        return [self._sanitize_value(item) for item in data]

    def _validate_string(self, data: str, **_kwargs) -> str:
        """Validate string input."""
        if not isinstance(data, str):
            raise ValidationError("Input must be a string")

        if len(data) > self.config.max_string_length:
            raise SecurityError(f"String too long: {len(data)} characters")

        # Remove potentially dangerous characters
        return self._sanitize_string(data)

    def _validate_numeric(
        self, data: int | float, field_name: str = None, **_kwargs
    ) -> int | float:
        """Validate numeric input."""
        if not isinstance(data, (int, float)):
            raise ValidationError("Input must be numeric")

        if field_name and field_name in self.config.numeric_bounds:
            bounds = self.config.numeric_bounds[field_name]
            if data < bounds["min"] or data > bounds["max"]:
                raise ValidationError(
                    f"{field_name} value {data} outside valid range [{bounds['min']}, {bounds['max']}]"
                )

        return data

    def _validate_file_path(self, path: str | Path, **_kwargs) -> Path:
        """Validate file path for security."""
        path = Path(path).resolve()

        # Check for dangerous paths
        for blocked in self.config.blocked_paths:
            if str(path).startswith(blocked):
                raise SecurityError(f"Access to blocked path: {path}")

        # Check file extension
        if path.suffix.lower() not in self.config.allowed_extensions:
            raise SecurityError(f"File extension not allowed: {path.suffix}")

        # Check file exists and is readable
        if not path.exists():
            raise ValidationError(f"File does not exist: {path}")
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        if not os.access(path, os.R_OK):
            raise SecurityError(f"File not readable: {path}")

        return path

    def _validate_symbol(self, symbol: str, **_kwargs) -> str:
        """Validate trading symbol."""
        if not isinstance(symbol, str):
            raise ValidationError("Symbol must be a string")

        # Allow alphanumeric, hyphen, underscore, dot
        if not re.match(r"^[A-Z0-9\-_.]+$", symbol.upper()):
            raise ValidationError(f"Invalid symbol format: {symbol}")

        if len(symbol) > 10:  # Reasonable symbol length limit
            raise ValidationError(f"Symbol too long: {symbol}")

        return symbol.upper()

    def _validate_strategy_config(
        self, config: dict, _strategy_name: str = None, **_kwargs
    ) -> dict:
        """Validate strategy configuration parameters."""
        if not isinstance(config, dict):
            raise ValidationError("Strategy config must be a dictionary")

        validated = {}
        for param_name, value in config.items():
            if param_name not in self.config.strategy_params:
                # Allow custom parameters but log warning
                logger.warning(f"Unknown strategy parameter: {param_name}")
                validated[param_name] = self._sanitize_value(value)
                continue

            param_spec = self.config.strategy_params[param_name]

            # Type validation
            expected_type = param_spec["type"]
            if not isinstance(value, expected_type):
                try:
                    # Try to convert
                    if expected_type is int:
                        value = int(value)
                    elif expected_type is float:
                        value = float(value)
                except (ValueError, TypeError) as e:
                    raise ValidationError(
                        f"Parameter {param_name} must be {expected_type.__name__}"
                    ) from e

            # Range validation
            if "min" in param_spec and value < param_spec["min"]:
                raise ValidationError(
                    f"Parameter {param_name} too small: {value} < {param_spec['min']}"
                )
            if "max" in param_spec and value > param_spec["max"]:
                raise ValidationError(
                    f"Parameter {param_name} too large: {value} > {param_spec['max']}"
                )

            validated[param_name] = value

        return validated

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a single value."""
        if isinstance(value, str):
            return self._sanitize_string(value)
        if isinstance(value, (int, float)):
            return value  # Numerics are generally safe
        if isinstance(value, bool) or value is None:
            return value
        # Pass through DataFrames without modification (validated separately)
        if isinstance(value, pd.DataFrame):
            return value
        # For complex objects, convert to string representation
        # This is conservative - complex objects should be validated separately
        return str(value)

    def _sanitize_string(self, s: str) -> str:
        """Sanitize string input to prevent injection attacks."""
        # Remove null bytes and other dangerous characters
        s = s.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")

        # Limit length
        if len(s) > self.config.max_string_length:
            s = s[: self.config.max_string_length]

        return s
