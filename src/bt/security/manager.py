"""Central security management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bt.security.config_manager import SecureConfigManager
from bt.security.scanner import SecurityScanner
from bt.security.validator import InputValidator

# Global security instance
_security_instance: SecurityManager | None = None


class SecurityManager:
    """Central security management."""

    def __init__(self):
        self.validator = InputValidator()
        self.config_manager = SecureConfigManager()
        self.scanner = SecurityScanner()

    @classmethod
    def get_instance(cls) -> SecurityManager:
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
