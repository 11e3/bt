"""Secure configuration management with secret handling."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from bt.interfaces.core import ValidationError

logger = logging.getLogger(__name__)


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
