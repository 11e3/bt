"""Test CPCV (Combinatorially Purged Cross-Validation)."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from bt.validation.cpcv import CombinatorialPurgedCV


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Provide sample OHLCV DataFrame."""
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dates = [base_date + timedelta(days=i) for i in range(100)]

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [100 + i for i in range(100)],
            "high": [105 + i for i in range(100)],
            "low": [95 + i for i in range(100)],
            "close": [102 + i for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        }
    )


class TestCombinatorialPurgedCVInit:
    """Test CombinatorialPurgedCV initialization."""

    def test_default_parameters(self) -> None:
        """Test default initialization."""
        cpcv = CombinatorialPurgedCV()

        assert cpcv.num_splits == 5
        assert cpcv.test_size == 0.2
        assert cpcv.purge_pct == 0.02
        assert cpcv.embargo_pct == 0.01

    def test_custom_parameters(self) -> None:
        """Test custom initialization."""
        cpcv = CombinatorialPurgedCV(
            num_splits=10,
            test_size=0.3,
            purge_pct=0.05,
            embargo_pct=0.02,
        )

        assert cpcv.num_splits == 10
        assert cpcv.test_size == 0.3
        assert cpcv.purge_pct == 0.05
        assert cpcv.embargo_pct == 0.02


class TestCreateSplits:
    """Test create_splits method."""

    def test_basic_splits(self) -> None:
        """Test basic split creation."""
        cpcv = CombinatorialPurgedCV(num_splits=5, test_size=0.2, purge_pct=0.02, embargo_pct=0.01)

        splits = cpcv.create_splits(n_samples=100)

        assert len(splits) > 0
        for train_indices, test_indices in splits:
            assert len(train_indices) > 0
            assert len(test_indices) > 0

    def test_purge_applied(self) -> None:
        """Test that purge is applied between train and test."""
        cpcv = CombinatorialPurgedCV(num_splits=3, test_size=0.2, purge_pct=0.1, embargo_pct=0.0)

        splits = cpcv.create_splits(n_samples=100)

        for train_indices, test_indices in splits:
            test_start = test_indices.min()
            test_indices.max()

            # Check no training samples are within purge distance of test
            purge_samples = int(100 * 0.1)
            if test_start > purge_samples:
                # Train before test should have gap
                train_before = train_indices[train_indices < test_start]
                if len(train_before) > 0:
                    assert train_before.max() < test_start - purge_samples

    def test_embargo_applied(self) -> None:
        """Test that embargo is applied after test."""
        cpcv = CombinatorialPurgedCV(num_splits=3, test_size=0.2, purge_pct=0.0, embargo_pct=0.1)

        splits = cpcv.create_splits(n_samples=100)

        for train_indices, test_indices in splits:
            test_end = test_indices.max()

            # Check no training samples are within embargo distance after test
            embargo_samples = int(100 * 0.1)
            train_after = train_indices[train_indices > test_end]
            if len(train_after) > 0:
                assert train_after.min() > test_end + embargo_samples

    def test_test_size_proportion(self) -> None:
        """Test that test size is approximately correct."""
        cpcv = CombinatorialPurgedCV(num_splits=3, test_size=0.25, purge_pct=0.0, embargo_pct=0.0)

        splits = cpcv.create_splits(n_samples=100)

        for _train_indices, test_indices in splits:
            # Test size should be ~25 samples
            assert len(test_indices) == pytest.approx(25, abs=1)

    def test_no_overlap(self) -> None:
        """Test train and test indices don't overlap."""
        cpcv = CombinatorialPurgedCV(num_splits=5)

        splits = cpcv.create_splits(n_samples=100)

        for train_indices, test_indices in splits:
            overlap = set(train_indices) & set(test_indices)
            assert len(overlap) == 0


class TestCPCVRun:
    """Test run method."""

    def test_run_basic(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test basic CPCV run."""
        cpcv = CombinatorialPurgedCV(num_splits=3, test_size=0.2)

        def simple_backtest(
            _data: dict[str, pd.DataFrame], _params: dict[str, Any]
        ) -> dict[str, float]:
            return {"cagr": 10.0, "mdd": -5.0, "win_rate": 60.0, "sortino_ratio": 1.5}

        data = {"BTC": sample_ohlcv_df}
        results = cpcv.run(data, simple_backtest)

        assert "fold_results" in results
        assert "summary" in results
        assert len(results["fold_results"]) > 0

    def test_run_multiple_symbols(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test CPCV with multiple symbols."""
        cpcv = CombinatorialPurgedCV(num_splits=3)

        def simple_backtest(
            data: dict[str, pd.DataFrame], _params: dict[str, Any]
        ) -> dict[str, float]:
            return {"cagr": 10.0, "mdd": -5.0}

        data = {"BTC": sample_ohlcv_df, "ETH": sample_ohlcv_df.copy()}
        results = cpcv.run(data, simple_backtest)

        assert len(results["fold_results"]) > 0

    def test_summary_statistics(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test summary statistics are calculated."""
        cpcv = CombinatorialPurgedCV(num_splits=3)

        def simple_backtest(
            _data: dict[str, pd.DataFrame], _params: dict[str, Any]
        ) -> dict[str, float]:
            return {"cagr": 10.0, "mdd": -5.0, "win_rate": 60.0, "sortino_ratio": 1.5}

        data = {"BTC": sample_ohlcv_df}
        results = cpcv.run(data, simple_backtest)

        summary = results["summary"]
        assert "avg_cagr" in summary
        assert "std_cagr" in summary
        assert "min_cagr" in summary
        assert "max_cagr" in summary
        assert "avg_mdd" in summary
        assert "worst_mdd" in summary

    def test_fold_info(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test fold information is recorded."""
        cpcv = CombinatorialPurgedCV(num_splits=3)

        def simple_backtest(
            _data: dict[str, pd.DataFrame], _params: dict[str, Any]
        ) -> dict[str, float]:
            return {"cagr": 10.0, "mdd": -5.0}

        data = {"BTC": sample_ohlcv_df}
        results = cpcv.run(data, simple_backtest)

        for fold in results["fold_results"]:
            assert "fold" in fold
            assert "train_size" in fold
            assert "test_size" in fold
            assert "results" in fold
