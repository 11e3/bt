"""Security scanning and vulnerability detection."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


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
