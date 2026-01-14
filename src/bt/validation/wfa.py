"""Walk Forward Analysis (WFA) validation method.

Splits data into rolling windows and performs walk-forward testing
to evaluate strategy robustness over time.
"""

from collections.abc import Callable
from typing import Any

import pandas as pd

from bt.utils.logging import get_logger

logger = get_logger(__name__)


class WalkForwardAnalysis:
    """Walk Forward Analysis for strategy validation.

    Splits data into sequential train/test windows and evaluates
    strategy performance on out-of-sample data.

    Why WFA:
    - Prevents overfitting by testing on unseen data
    - Simulates realistic rolling optimization scenario
    - Provides distribution of performance across time periods
    """

    def __init__(
        self,
        train_periods: int = 12,
        test_periods: int = 3,
        step_periods: int = 3,
        anchored: bool = False,
    ) -> None:
        """Initialize WFA.

        Args:
            train_periods: Number of bars for training window
            test_periods: Number of bars for testing window
            step_periods: Number of bars to step forward
            anchored: If True, training window grows (anchored WFA)
        """
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.step_periods = step_periods
        self.anchored = anchored

        logger.info(
            "WFA initialized",
            extra={
                "train_periods": train_periods,
                "test_periods": test_periods,
                "step_periods": step_periods,
                "anchored": anchored,
            },
        )

    def split_data(self, df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Split data into train/test windows.

        Args:
            df: DataFrame with datetime column

        Returns:
            List of (train_df, test_df) tuples
        """
        df = df.sort_values("datetime").reset_index(drop=True)
        total_rows = len(df)

        splits = []
        start_idx = 0

        while True:
            # Calculate indices
            train_end = start_idx + self.train_periods
            test_end = train_end + self.test_periods

            # Check if we have enough data
            if test_end > total_rows:
                break

            # Get train/test sets
            if self.anchored:
                train_df = df.iloc[0:train_end].copy()
            else:
                train_df = df.iloc[start_idx:train_end].copy()

            test_df = df.iloc[train_end:test_end].copy()

            splits.append((train_df, test_df))

            # Move forward
            start_idx += self.step_periods

        logger.debug("Data split complete", extra={"num_windows": len(splits)})
        return splits

    def run(
        self,
        data: dict[str, pd.DataFrame],
        backtest_func: Callable[..., Any],
        optimize_func: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Run Walk Forward Analysis.

        Args:
            data: Dictionary of symbol -> DataFrame
            backtest_func: Function to run backtest on data
            optimize_func: Optional function to optimize parameters

        Returns:
            Dictionary with WFA results and summary
        """
        # Get splits for first symbol (assuming all have same length)
        first_symbol = next(iter(data.keys()))
        splits = self.split_data(data[first_symbol])

        logger.info(
            "Starting WFA",
            extra={"num_windows": len(splits), "symbols": list(data.keys())},
        )

        all_results = []

        for i, (train_df, test_df) in enumerate(splits):
            logger.info(
                f"Processing window {i + 1}/{len(splits)}",
                extra={
                    "train_start": train_df["datetime"].min().isoformat(),
                    "train_end": train_df["datetime"].max().isoformat(),
                    "test_start": test_df["datetime"].min().isoformat(),
                    "test_end": test_df["datetime"].max().isoformat(),
                },
            )

            # Prepare train/test data for all symbols
            train_data = {}
            test_data = {}

            for symbol, df in data.items():
                train_mask = (df["datetime"] >= train_df["datetime"].min()) & (
                    df["datetime"] <= train_df["datetime"].max()
                )
                test_mask = (df["datetime"] >= test_df["datetime"].min()) & (
                    df["datetime"] <= test_df["datetime"].max()
                )

                train_data[symbol] = df[train_mask].copy()
                test_data[symbol] = df[test_mask].copy()

            # Optimize on training set (if optimizer provided)
            best_params = optimize_func(train_data) if optimize_func else {}

            # Test on test set
            test_results = backtest_func(test_data, best_params)

            window_result = {
                "window": i + 1,
                "train_start": train_df["datetime"].min(),
                "train_end": train_df["datetime"].max(),
                "test_start": test_df["datetime"].min(),
                "test_end": test_df["datetime"].max(),
                "params": best_params,
                "results": test_results,
            }

            all_results.append(window_result)

            logger.info(
                f"Window {i + 1} complete",
                extra={
                    "test_cagr": test_results.get("cagr", 0),
                    "test_mdd": test_results.get("mdd", 0),
                },
            )

        # Aggregate results
        summary = self._summarize_results(all_results)

        logger.info("WFA complete", extra={"summary": summary})

        return {"window_results": all_results, "summary": summary}

    def _summarize_results(self, all_results: list[dict[str, Any]]) -> dict[str, float]:
        """Summarize WFA results across all windows.

        Args:
            all_results: List of window results

        Returns:
            Summary statistics
        """
        import numpy as np

        cagrs = [r["results"].get("cagr", 0) for r in all_results]
        mdds = [r["results"].get("mdd", 0) for r in all_results]
        win_rates = [r["results"].get("win_rate", 0) for r in all_results]

        return {
            "avg_cagr": float(np.mean(cagrs)),
            "median_cagr": float(np.median(cagrs)),
            "std_cagr": float(np.std(cagrs)),
            "min_cagr": float(np.min(cagrs)),
            "max_cagr": float(np.max(cagrs)),
            "avg_mdd": float(np.mean(mdds)),
            "median_mdd": float(np.median(mdds)),
            "worst_mdd": float(np.min(mdds)),
            "avg_win_rate": float(np.mean(win_rates)),
            "num_windows": len(all_results),
        }
