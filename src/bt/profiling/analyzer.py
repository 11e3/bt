"""Code quality analysis tool."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CodeQualityAnalyzer:
    """Code quality analysis tool."""

    def __init__(self):
        self.metrics: dict[str, Any] = {}

    def analyze_file(self, file_path: Path) -> dict[str, Any]:
        """Analyze code quality metrics for a file."""
        try:
            with file_path.open(encoding="utf-8") as f:
                content = f.read()

            return self._analyze_code(content, str(file_path))
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return {}

    def analyze_directory(self, directory: Path) -> dict[str, Any]:
        """Analyze code quality for all Python files in directory."""
        results = {}

        for py_file in directory.rglob("*.py"):
            if not self._should_analyze_file(py_file):
                continue

            metrics = self.analyze_file(py_file)
            if metrics:
                results[str(py_file)] = metrics

        return results

    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if file should be analyzed."""
        # Skip common exclusions
        exclusions = {
            "__pycache__",
            ".git",
            "node_modules",
            "build",
            "dist",
            "venv",
            ".env",
            ".tox",
            "migrations",
        }

        return all(not (part in exclusions or part.startswith(".")) for part in file_path.parts)

    def _analyze_code(self, content: str, filename: str) -> dict[str, Any]:
        """Analyze code quality metrics."""
        lines = content.split("\n")
        total_lines = len(lines)

        # Basic metrics
        metrics = {
            "filename": filename,
            "total_lines": total_lines,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "functions": 0,
            "classes": 0,
            "complexity_score": 0,
            "avg_function_length": 0,
            "max_function_length": 0,
        }

        # Analyze each line
        functions = []
        current_function_lines = 0
        in_function = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                metrics["blank_lines"] += 1
            elif stripped.startswith("#"):
                metrics["comment_lines"] += 1
            elif stripped.startswith("def "):
                metrics["functions"] += 1
                if in_function:
                    functions.append(current_function_lines)
                current_function_lines = 0
                in_function = True
            elif stripped.startswith("class "):
                metrics["classes"] += 1
                if in_function:
                    functions.append(current_function_lines)
                    current_function_lines = 0
                    in_function = False
            else:
                metrics["code_lines"] += 1
                if in_function:
                    current_function_lines += 1

                # Simple complexity indicators
                if any(
                    keyword in stripped
                    for keyword in ["if ", "elif ", "else:", "for ", "while ", "try:", "except "]
                ):
                    metrics["complexity_score"] += 1

        if in_function:
            functions.append(current_function_lines)

        # Calculate function metrics
        if functions:
            metrics["avg_function_length"] = sum(functions) / len(functions)
            metrics["max_function_length"] = max(functions)

        # Calculate comment ratio
        total_code_and_comments = metrics["code_lines"] + metrics["comment_lines"]
        metrics["comment_ratio"] = (
            metrics["comment_lines"] / total_code_and_comments if total_code_and_comments > 0 else 0
        )

        return metrics

    def generate_quality_report(self, results: dict[str, Any]) -> str:
        """Generate code quality report."""
        if not results:
            return "No files analyzed."

        # Aggregate metrics
        total_files = len(results)
        total_lines = sum(m.get("total_lines", 0) for m in results.values())
        total_functions = sum(m.get("functions", 0) for m in results.values())
        avg_comment_ratio = sum(m.get("comment_ratio", 0) for m in results.values()) / total_files

        # Find files with issues
        complex_files = [(f, m) for f, m in results.items() if m.get("complexity_score", 0) > 50]

        long_functions = [
            (f, m) for f, m in results.items() if m.get("max_function_length", 0) > 50
        ]

        report = f"""
Code Quality Analysis Report
============================

Summary:
- Total files analyzed: {total_files}
- Total lines of code: {total_lines}
- Total functions: {total_functions}
- Average comment ratio: {avg_comment_ratio:.2%}

Potential Issues:
"""

        if complex_files:
            report += f"\nHighly complex files (>50 complexity score): {len(complex_files)}\n"
            for filename, metrics in complex_files[:5]:  # Show top 5
                report += f"- {filename}: {metrics.get('complexity_score', 0)} complexity score\n"

        if long_functions:
            report += f"\nFunctions with high line count (>50 lines): {len(long_functions)}\n"
            for filename, metrics in long_functions[:5]:  # Show top 5
                report += f"- {filename}: {metrics.get('max_function_length', 0)} max lines\n"

        if not complex_files and not long_functions:
            report += "\n✅ No major code quality issues detected."

        return report
