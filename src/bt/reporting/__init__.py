"""Performance reporting and metrics."""

from bt.reporting.metrics import (
    calculate_performance_metrics,
    print_performance_report,
    print_sample_trades,
)
from bt.reporting.visualization import (
    plot_equity_curve,
    plot_market_regime_analysis,
    plot_wfa_results,
    plot_yearly_returns,
    save_all_charts,
)

__all__ = [
    "calculate_performance_metrics",
    "print_performance_report",
    "print_sample_trades",
    "plot_equity_curve",
    "plot_yearly_returns",
    "plot_wfa_results",
    "plot_market_regime_analysis",
    "save_all_charts",
]
