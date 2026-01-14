#!/usr/bin/env python3
"""
Code quality analysis and performance profiling script for BT Framework.

This script performs comprehensive code quality checks including:
- Code complexity analysis
- Performance profiling
- Code quality metrics
- Automated code review suggestions
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bt.profiling import get_profiler, run_quality_analysis
from bt.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def run_performance_profiling():
    """Run performance profiling on key framework components."""
    print("🔬 Running performance profiling...")

    profiler = get_profiler()

    # Import and profile key components
    try:
        from bt import BacktestFramework  # noqa: I001
        from bt.data.storage import get_data_manager  # noqa: I001
        from bt.security import SecurityManager  # noqa: I001

        # Profile framework initialization
        with profiler.profile_context("framework_initialization"):
            BacktestFramework()

        # Profile data manager operations
        with profiler.profile_context("data_manager_operations"):
            dm = get_data_manager()
            dm.store("test_key", {"test": "data"})
            dm.retrieve("test_key")
            dm.delete("test_key")

        # Profile security validation
        with profiler.profile_context("security_validation"):
            security = SecurityManager.get_instance()
            security.validator.validate("test_string", "string")
            security.validator.validate([1, 2, 3], "list")

        print("✅ Performance profiling completed")

        # Generate stats
        stats = profiler.get_stats_report()
        if not stats.empty:
            print("\n📊 Performance Summary:")
            print(stats.to_string(index=False))

        return True

    except Exception as e:
        print(f"❌ Performance profiling failed: {e}")
        return False


def run_code_quality_analysis(path: Path):
    """Run code quality analysis."""
    print(f"🔍 Analyzing code quality in {path}...")

    try:
        report = run_quality_analysis(path)
        print("\n" + report)

        # Check for critical issues
        issues = []
        if "complexity" in report.lower() and "50" in report:
            issues.append("High complexity detected")
        if "function" in report.lower() and "50 lines" in report:
            issues.append("Long functions detected")

        if issues:
            print(f"\n⚠️  Code quality issues found: {', '.join(issues)}")
            return False
        print("\n✅ Code quality analysis passed")
        return True

    except Exception as e:
        print(f"❌ Code quality analysis failed: {e}")
        return False


def check_dependencies_quality():
    """Check code quality of dependencies."""
    print("📦 Analyzing dependency quality...")

    try:
        # Check for required dependencies
        required_deps = [
            "pandas",
            "numpy",
            "pydantic",
            "pyupbit",
            "plotly",
            "psutil",
            "pytest",
            "ruff",
            "mypy",
        ]

        missing = []
        for dep in required_deps:
            try:
                __import__(dep.replace("-", "_"))
            except ImportError:
                missing.append(dep)

        if missing:
            print(f"⚠️  Missing recommended dependencies: {', '.join(missing)}")
            return False
        print("✅ All recommended dependencies available")
        return True

    except Exception as e:
        print(f"❌ Dependency check failed: {e}")
        return False


def generate_quality_report(results: dict[str, Any], output_file: Path = None):
    """Generate comprehensive quality report."""
    report = """
BT Framework Code Quality Report
=================================

Analysis Results:
"""

    passed = sum(results.values())
    total = len(results)

    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        report += f"- {check}: {status}\n"

    report += f"""
Summary:
- Passed: {passed}/{total}
- Overall Status: {"✅ GOOD" if passed == total else "⚠️  NEEDS ATTENTION"}
"""

    if output_file:
        output_file.write_text(report)
        print(f"📄 Detailed report saved to {output_file}")

    return report


def main():
    """Main code quality check function."""
    parser = argparse.ArgumentParser(description="BT Framework Code Quality Analyzer")
    parser.add_argument(
        "--path", "-p", type=Path, default=Path("src"), help="Path to analyze (default: src)"
    )
    parser.add_argument("--output", "-o", type=Path, help="Output file for detailed report")
    parser.add_argument(
        "--skip-performance", action="store_true", help="Skip performance profiling"
    )
    parser.add_argument(
        "--skip-dependency-check", action="store_true", help="Skip dependency quality check"
    )

    args = parser.parse_args()

    setup_logging(level="INFO")
    print("🧪 BT Framework Code Quality Analysis")
    print("=" * 50)

    results = {}

    # Run code quality analysis
    results["code_quality"] = run_code_quality_analysis(args.path)

    # Run performance profiling
    if not args.skip_performance:
        results["performance"] = run_performance_profiling()

    # Check dependencies
    if not args.skip_dependency_check:
        results["dependencies"] = check_dependencies_quality()

    # Generate report
    report = generate_quality_report(results, args.output)
    print("\n" + report)

    # Exit with appropriate code
    passed = sum(results.values())
    total = len(results)

    if passed == total:
        print("🎉 All quality checks passed!")
        return 0
    print(f"⚠️  {total - passed} quality checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
