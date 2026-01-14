"""
Security module for BT Framework.

Provides comprehensive security features including:
- Input validation and sanitization
- Secure configuration management
- Security scanning integration
- Safe defaults and validation rules
"""

import hashlib
import logging
import os
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from ..exceptions import SecurityError
from ..interfaces.core import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


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
            "price": {"min": 0.00000001, "max": 1_000_000},
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

        # Validate value ranges
        bounds = self.config.numeric_bounds
        for col in ["open", "high", "low", "close"]:
            if (df[col] < bounds["price"]["min"]).any() or (df[col] > bounds["price"]["max"]).any():
                raise ValidationError(f"Price values in {col} outside valid range")

        if (df["volume"] < bounds["volume"]["min"]).any() or (
            df["volume"] > bounds["volume"]["max"]
        ).any():
            raise ValidationError("Volume values outside valid range")

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


class SecureConfigManager:
    """Secure configuration management with secret handling."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path.home() / ".bt_config"
        self._secrets: dict[str, str] = {}
        self._load_secrets()

    def _load_secrets(self):
        """Load encrypted secrets from secure storage."""
        secrets_file = self.config_dir / "secrets.enc"
        if secrets_file.exists():
            try:
                # In a real implementation, this would decrypt the file
                # For now, we'll use environment variables as fallback
                pass
            except Exception as e:
                logger.warning(f"Could not load secrets: {e}")

    def get_secret(self, key: str, default: str = None) -> str | None:
        """Get a secret value securely."""
        # First check environment variables
        env_key = f"BT_{key.upper()}"
        value = os.getenv(env_key)

        if value:
            return value

        # Then check loaded secrets
        return self._secrets.get(key, default)

    def set_secret(self, key: str, value: str):
        """Set a secret value securely."""
        # In production, this would encrypt and store the secret
        # For now, we'll just validate it's not empty
        if not value or not isinstance(value, str):
            raise ValidationError("Secret value must be a non-empty string")

        self._secrets[key] = value

    def validate_config(self, config: dict) -> dict:
        """Validate configuration for security issues."""

        # Check for sensitive data in plain text
        sensitive_keys = {"password", "secret", "key", "token", "api_key"}
        for key, value in config.items():
            key_lower = key.lower()
            if (
                any(sensitive in key_lower for sensitive in sensitive_keys)
                and isinstance(value, str)
                and len(value) > 10
            ):
                # Suggest using secrets manager
                logger.warning(
                    f"Sensitive config key detected: {key}. Consider using secrets management."
                )

        return config


class SecurityScanner:
    """Security scanning and vulnerability detection."""

    def __init__(self):
        self.vulnerabilities: list[dict] = []

    def scan_codebase(self, path: Path) -> list[dict]:
        """Scan codebase for security issues."""
        self.vulnerabilities = []

        # Check for common security issues
        self._scan_files(path)
        self._scan_dependencies()
        self._scan_configuration()

        return self.vulnerabilities

    def _scan_files(self, path: Path):
        """Scan files for security issues."""
        for file_path in path.rglob("*.py"):
            try:
                with file_path.open(encoding="utf-8") as f:
                    content = f.read()

                self._check_file_security(file_path, content)
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")

    def _check_file_security(self, file_path: Path, content: str):
        """Check individual file for security issues."""
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for hardcoded secrets
            if self._has_hardcoded_secret(line):
                self.vulnerabilities.append(
                    {
                        "type": "hardcoded_secret",
                        "file": str(file_path),
                        "line": i,
                        "severity": "high",
                        "description": "Potential hardcoded secret detected",
                    }
                )

            # Check for dangerous functions
            if self._has_dangerous_function(line):
                self.vulnerabilities.append(
                    {
                        "type": "dangerous_function",
                        "file": str(file_path),
                        "line": i,
                        "severity": "medium",
                        "description": "Use of potentially dangerous function",
                    }
                )

            # Check for SQL injection vulnerabilities
            if self._has_sql_injection_risk(line):
                self.vulnerabilities.append(
                    {
                        "type": "sql_injection",
                        "file": str(file_path),
                        "line": i,
                        "severity": "high",
                        "description": "Potential SQL injection vulnerability",
                    }
                )

    def _has_hardcoded_secret(self, line: str) -> bool:
        """Check for hardcoded secrets."""
        # Simple pattern matching for common secret indicators
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
        ]

        return any(re.search(pattern, line, re.IGNORECASE) for pattern in secret_patterns)

    def _has_dangerous_function(self, line: str) -> bool:
        """Check for dangerous function usage."""
        dangerous_functions = [
            "eval(",
            "exec(",
            "pickle.loads(",
            "subprocess.call(",
            "os.system(",
            "os.popen(",
            "shell=True",
        ]

        return any(func in line for func in dangerous_functions)

    def _has_sql_injection_risk(self, line: str) -> bool:
        """Check for SQL injection vulnerabilities."""
        # Look for string formatting in SQL queries
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE"]
        format_indicators = ["%", ".format(", 'f"', "+"]

        has_sql = any(keyword in line.upper() for keyword in sql_keywords)
        has_formatting = any(indicator in line for indicator in format_indicators)

        return has_sql and has_formatting

    def _scan_dependencies(self):
        """Scan dependencies for known vulnerabilities."""
        try:
            # This would integrate with safety or similar tools
            # For now, just check if requirements.txt exists
            req_file = Path("requirements.txt")
            if req_file.exists():
                self.vulnerabilities.append(
                    {
                        "type": "dependency_scan",
                        "file": "requirements.txt",
                        "severity": "info",
                        "description": "Dependencies should be scanned with safety or pip-audit",
                    }
                )
        except Exception as e:
            logger.warning(f"Could not scan dependencies: {e}")

    def _scan_configuration(self):
        """Scan configuration for security issues."""
        config_files = ["config.yaml", "config.json", "settings.py", ".env"]

        for config_file in config_files:
            if Path(config_file).exists():
                self.vulnerabilities.append(
                    {
                        "type": "config_security",
                        "file": config_file,
                        "severity": "medium",
                        "description": "Configuration file should not contain sensitive data",
                    }
                )


# Global security instance
_security_instance: Optional["SecurityManager"] = None


class SecurityManager:
    """Central security management."""

    def __init__(self):
        self.validator = InputValidator()
        self.config_manager = SecureConfigManager()
        self.scanner = SecurityScanner()

    @classmethod
    def get_instance(cls) -> "SecurityManager":
        """Get singleton instance."""
        global _security_instance
        if _security_instance is None:
            _security_instance = cls()
        return _security_instance

    def validate_input(self, data: Any, data_type: str, **kwargs) -> Any:
        """Validate input data."""
        return self.validator.validate(data, data_type, **kwargs)

    def scan_security(self, path: Path) -> list[dict]:
        """Perform security scan."""
        return self.scanner.scan_codebase(path)

    def get_secure_config(self) -> SecureConfigManager:
        """Get secure configuration manager."""
        return self.config_manager


# Convenience functions
def validate_input(data: Any, data_type: str, **kwargs) -> Any:
    """Global input validation function."""
    return SecurityManager.get_instance().validate_input(data, data_type, **kwargs)


def scan_security(path: Path = Path()) -> list[dict]:
    """Global security scanning function."""
    return SecurityManager.get_instance().scan_security(path)
