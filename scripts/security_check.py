#!/usr/bin/env python3
"""
Security scanning and validation script for BT Framework.

This script performs comprehensive security checks including:
- Input validation testing
- Security scanning for vulnerabilities
- Dependency security analysis
- Configuration security validation
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from bt.security import SecurityManager, SecurityScanner
from bt.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def scan_codebase(path: Path, output_format: str = "text") -> int:
    """Scan codebase for security issues."""
    print("🔍 Scanning codebase for security vulnerabilities...")

    scanner = SecurityScanner()
    vulnerabilities = scanner.scan_codebase(path)

    if not vulnerabilities:
        print("✅ No security vulnerabilities found!")
        return 0

    print(f"⚠️  Found {len(vulnerabilities)} potential security issues:")

    for vuln in vulnerabilities:
        severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢", "info": "ℹ️"}.get(
            vuln["severity"], "❓"
        )

        print(f"\n{severity_icon} {vuln['severity'].upper()}: {vuln['description']}")
        print(f"   File: {vuln['file']}")
        if "line" in vuln:
            print(f"   Line: {vuln['line']}")

    # Return non-zero exit code for high/medium severity issues
    high_severity = [v for v in vulnerabilities if v["severity"] in ["high", "medium"]]
    if high_severity:
        print(f"\n❌ Found {len(high_severity)} high/medium severity issues")
        return 1

    return 0


def validate_inputs() -> int:
    """Test input validation functionality."""
    print("🛡️  Testing input validation...")

    validator = SecurityManager.get_instance().validator

    test_cases = [
        # DataFrame validation
        (pd.DataFrame({"a": [1, 2, 3]}), "dataframe", True),
        # Invalid DataFrame
        ({"a": [1, 2, 3]}, "dataframe", False),
        # String validation
        ("valid_string", "string", True),
        ("a" * 10000, "string", False),  # Too long
        # Symbol validation
        ("BTC", "symbol", True),
        ("btc", "symbol", True),  # Should be uppercased
        ("INVALID@SYMBOL", "symbol", False),
        # Numeric validation
        (100.0, "numeric", True),
        ("not_a_number", "numeric", False),
    ]

    failed = 0
    for i, (data, data_type, should_pass) in enumerate(test_cases):
        try:
            validator.validate(data, data_type)
            if should_pass:
                print(f"✅ Test {i + 1}: PASS")
            else:
                print(f"❌ Test {i + 1}: Expected failure but passed")
                failed += 1
        except Exception as e:
            if not should_pass:
                print(f"✅ Test {i + 1}: PASS (correctly rejected: {e})")
            else:
                print(f"❌ Test {i + 1}: Expected success but failed: {e}")
                failed += 1

    if failed == 0:
        print("✅ All input validation tests passed!")
        return 0
    print(f"❌ {failed} input validation tests failed!")
    return 1


def check_dependencies() -> int:
    """Check dependencies for security issues."""
    print("📦 Checking dependencies for security vulnerabilities...")

    try:
        import json
        import subprocess

        # Check if safety is available
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True,
            )
            packages = json.loads(result.stdout)

            # Look for known vulnerable packages (simplified check)
            vulnerable = []
            for pkg in packages:
                name = pkg.get("name", "").lower()
                version = pkg.get("version", "")

                # Simple checks for known vulnerabilities
                if "insecure" in name or "vulnerable" in name:
                    vulnerable.append(f"{name}@{version}")

            if vulnerable:
                print(f"⚠️  Found potentially vulnerable packages: {', '.join(vulnerable)}")
                return 1
            print("✅ No obvious dependency vulnerabilities found")
            return 0

        except subprocess.CalledProcessError:
            print("⚠️  Could not check dependencies with pip")
            return 1

    except ImportError:
        print("⚠️  pip not available for dependency checking")
        return 1


def validate_configuration() -> int:
    """Validate configuration security."""
    print("⚙️  Validating configuration security...")

    config_manager = SecurityManager.get_instance().config_manager

    # Test secure config
    test_config = {
        "database_url": "postgresql://user:password@localhost/db",
        "api_key": "sk-1234567890abcdef",
        "normal_setting": "value",
    }

    try:
        config_manager.validate_config(test_config)
        print("✅ Configuration validation passed")
        return 0
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1


def main():
    """Main security check function."""
    parser = argparse.ArgumentParser(description="BT Framework Security Scanner")
    parser.add_argument(
        "--path",
        "-p",
        type=Path,
        default=Path(),
        help="Path to scan (default: current directory)",
    )
    parser.add_argument(
        "--format", "-f", choices=["text", "json"], default="text", help="Output format"
    )
    parser.add_argument(
        "--skip-input-validation", action="store_true", help="Skip input validation tests"
    )
    parser.add_argument(
        "--skip-dependency-check", action="store_true", help="Skip dependency security check"
    )
    parser.add_argument(
        "--skip-config-validation", action="store_true", help="Skip configuration validation"
    )

    args = parser.parse_args()

    setup_logging(level="INFO")
    print("🔒 BT Framework Security Check")
    print("=" * 50)

    exit_code = 0

    # Run security scan
    scan_result = scan_codebase(args.path, args.format)
    exit_code = max(exit_code, scan_result)

    # Test input validation
    if not args.skip_input_validation:
        validation_result = validate_inputs()
        exit_code = max(exit_code, validation_result)

    # Check dependencies
    if not args.skip_dependency_check:
        dep_result = check_dependencies()
        exit_code = max(exit_code, dep_result)

    # Validate configuration
    if not args.skip_config_validation:
        config_result = validate_configuration()
        exit_code = max(exit_code, config_result)

    print("\n" + "=" * 50)
    if exit_code == 0:
        print("🎉 All security checks passed!")
    else:
        print(f"⚠️  Security checks completed with issues (exit code: {exit_code})")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
