"""Combinatorially Purged Cross-Validation (CPCV).

Advanced cross-validation method that handles time-series data
with purging and embargo to prevent leakage.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from bt.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

logger = get_logger(__name__)


class CombinatorialPurgedCV:
    """Combinatorially Purged Cross-Validation.

    Advanced CV method for time series that:
    1. Creates multiple non-contiguous splits
    2. Purges data between train/test to prevent leakage
    3. Embargoes data after test to prevent forward-looking bias

    Why CPCV:
    - Prevents information leakage in time series
    - Provides more robust performance estimates
    - Handles overlapping observations
    """

    def __init__(
        self,
        num_splits: int = 5,
        test_size: float = 0.2,
        purge_pct: float = 0.02,
        embargo_pct: float = 0.01,
    ) -> None:
        """Initialize CPCV.

        Args:
            num_splits: Number of cross-validation splits
            test_size: Fraction of data for testing (0.0-1.0)
            purge_pct: Percentage of data to purge between train/test
            embargo_pct: Percentage of data to embargo after test set
        """
        self.num_splits = num_splits
        self.test_size = test_size
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

        logger.info(
            "CPCV initialized",
            extra={
                "num_splits": num_splits,
                "test_size": test_size,
                "purge_pct": purge_pct,
                "embargo_pct": embargo_pct,
            },
        )

    def create_splits(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Create train/test splits with purging and embargo.

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_indices, test_indices) tuples
        """
        test_samples = int(n_samples * self.test_size)
        purge_samples = int(n_samples * self.purge_pct)
        embargo_samples = int(n_samples * self.embargo_pct)

        splits = []

        # Create equally spaced test sets
        test_starts = np.linspace(0, n_samples - test_samples, self.num_splits, dtype=int)

        for test_start in test_starts:
            test_end = test_start + test_samples

            # Create test indices
            test_indices = np.arange(test_start, test_end)

            # Create train indices with purging
            train_indices_list: list[int] = []

            # Add samples before test set (with purge)
            if test_start > purge_samples:
                train_indices_list.extend(range(0, test_start - purge_samples))

            # Add samples after test set (with embargo)
            if test_end + embargo_samples < n_samples:
                train_indices_list.extend(range(test_end + embargo_samples, n_samples))

            train_indices = np.array(train_indices_list)

            if len(train_indices) > 0:
                splits.append((train_indices, test_indices))

        logger.debug("Splits created", extra={"num_splits": len(splits)})
        return splits

    def run(
        self,
        data: dict[str, pd.DataFrame],
        backtest_func: Callable[..., Any],
    ) -> dict[str, Any]:
        """Run Combinatorially Purged Cross-Validation.

        Args:
            data: Dictionary of symbol -> DataFrame
            backtest_func: Function to run backtest on data

        Returns:
            Dictionary with CPCV results and summary
        """
        # Get number of samples (assuming all symbols have same length)
        first_symbol = next(iter(data.keys()))
        n_samples = len(data[first_symbol])

        # Create splits
        splits = self.create_splits(n_samples)

        logger.info(
            "Starting CPCV",
            extra={"num_folds": len(splits), "symbols": list(data.keys())},
        )

        all_results = []

        for i, (train_indices, test_indices) in enumerate(splits):
            logger.info(
                f"Processing fold {i + 1}/{len(splits)}",
                extra={
                    "train_samples": len(train_indices),
                    "test_samples": len(test_indices),
                },
            )

            # Prepare train/test data for all symbols
            train_data = {}
            test_data = {}

            for symbol, df in data.items():
                train_data[symbol] = df.iloc[train_indices].copy()
                test_data[symbol] = df.iloc[test_indices].copy()

            # Run backtest on test set
            test_results = backtest_func(test_data, {})

            fold_result = {
                "fold": i + 1,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "results": test_results,
            }

            all_results.append(fold_result)

            logger.info(
                f"Fold {i + 1} complete",
                extra={
                    "test_cagr": test_results.get("cagr", 0),
                    "test_mdd": test_results.get("mdd", 0),
                },
            )

        # Aggregate results
        summary = self._summarize_results(all_results)

        logger.info("CPCV complete", extra={"summary": summary})

        return {"fold_results": all_results, "summary": summary}

    def _summarize_results(self, all_results: list[dict[str, Any]]) -> dict[str, float]:
        """Summarize CPCV results across all folds.

        Args:
            all_results: List of fold results

        Returns:
            Summary statistics
        """
        cagrs = [r["results"].get("cagr", 0) for r in all_results]
        mdds = [r["results"].get("mdd", 0) for r in all_results]
        win_rates = [r["results"].get("win_rate", 0) for r in all_results]
        sortinos = [r["results"].get("sortino_ratio", 0) for r in all_results]

        return {
            "avg_cagr": float(np.mean(cagrs)),
            "std_cagr": float(np.std(cagrs)),
            "min_cagr": float(np.min(cagrs)),
            "max_cagr": float(np.max(cagrs)),
            "avg_mdd": float(np.mean(mdds)),
            "worst_mdd": float(np.min(mdds)),
            "avg_win_rate": float(np.mean(win_rates)),
            "avg_sortino": float(np.mean(sortinos)),
            "consistency": float(len([c for c in cagrs if c > 0]) / len(cagrs) * 100),
            "num_folds": len(all_results),
        }
