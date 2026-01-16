"""Performance report generation.

Responsible for generating performance reports and charts.
Follows Single Responsibility Principle.
"""

from pathlib import Path
from typing import Any

from bt.interfaces.protocols import ILogger
from bt.utils.logging import get_logger


class ReportGenerator:
    """Generates performance reports and visualizations.

    Responsibilities:
    - Generate performance charts
    - Create report files
    - Format report output

    Does NOT handle:
    - Backtest execution
    - Strategy management
    - Data loading
    """

    def __init__(self, report_directory: str = "reports", logger: ILogger | None = None):
        """Initialize report generator.

        Args:
            report_directory: Directory to save reports
            logger: Logger instance
        """
        self.report_directory = report_directory
        self.logger = logger or get_logger(__name__)

    def generate_full_report(self, results: dict[str, Any]) -> None:
        """Generate complete performance report.

        Args:
            results: Backtest results dictionary
        """
        from bt.reporting.charts import generate_all_charts

        performance_data = results.get("performance", {})
        equity_curve = results.get("equity_curve", {})

        # Generate charts
        generate_all_charts(equity_curve, performance_data, self.report_directory)

        self.logger.info(f"Performance report generated in {self.report_directory}")

    def generate_charts(self, results: dict[str, Any]) -> None:
        """Generate only charts without full report.

        Args:
            results: Backtest results dictionary
        """
        from bt.reporting.charts import generate_all_charts

        equity_curve = results.get("equity_curve", {})
        performance_data = results.get("performance", {})

        generate_all_charts(equity_curve, performance_data, self.report_directory)
        self.logger.info(f"Charts generated in {self.report_directory}")

    def generate_summary_json(self, results: dict[str, Any], output_path: str) -> None:
        """Generate JSON summary report.

        Args:
            results: Backtest results dictionary
            output_path: Path to save JSON file
        """
        import json
        from decimal import Decimal

        # Convert Decimal to float for JSON serialization
        def decimal_to_float(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            raise TypeError

        try:
            with Path(output_path).open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=decimal_to_float)

            self.logger.info(f"Summary JSON saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error generating JSON summary: {e}")

    def print_summary(self, results: dict[str, Any]) -> None:
        """Print performance summary to console.

        Args:
            results: Backtest results dictionary
        """
        from bt.reporting.metrics import print_performance_report, print_sample_trades

        performance = results.get("performance", {})
        trades = results.get("trades", [])

        # Print metrics
        if hasattr(performance, "__dict__"):
            print_performance_report(performance)
        else:
            self.logger.warning("Performance data not in expected format")

        # Print sample trades
        print_sample_trades(trades, max_trades=10)
