"""Test WFA (Walk Forward Analysis) validation."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from bt.validation.wfa import WalkForwardAnalysis


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Provide sample OHLCV DataFrame."""
    base_date = datetime(2024, 1, 1, tzinfo=UTC)
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


class TestWalkForwardAnalysisInit:
    """Test WalkForwardAnalysis initialization."""

    def test_default_parameters(self) -> None:
        """Test default initialization."""
        wfa = WalkForwardAnalysis()

        assert wfa.train_periods == 12
        assert wfa.test_periods == 3
        assert wfa.step_periods == 3
        assert wfa.anchored is False

    def test_custom_parameters(self) -> None:
        """Test custom initialization."""
        wfa = WalkForwardAnalysis(
            train_periods=20,
            test_periods=5,
            step_periods=5,
            anchored=True,
        )

        assert wfa.train_periods == 20
        assert wfa.test_periods == 5
        assert wfa.step_periods == 5
        assert wfa.anchored is True


class TestSplitData:
    """Test split_data method."""

    def test_basic_split(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test basic data splitting."""
        wfa = WalkForwardAnalysis(train_periods=30, test_periods=10, step_periods=10)

        splits = wfa.split_data(sample_ohlcv_df)

        assert len(splits) > 0
        for train_df, test_df in splits:
            assert len(train_df) == 30
            assert len(test_df) == 10

    def test_anchored_split(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test anchored splitting (growing train window)."""
        wfa = WalkForwardAnalysis(train_periods=30, test_periods=10, step_periods=10, anchored=True)

        splits = wfa.split_data(sample_ohlcv_df)

        assert len(splits) > 0

        # Anchored: train window starts from 0 and grows
        for i, (train_df, _test_df) in enumerate(splits):
            # First split has 30 rows, subsequent splits have more
            if i == 0:
                assert len(train_df) == 30
            else:
                assert len(train_df) >= 30

    def test_insufficient_data(self) -> None:
        """Test with insufficient data for splits."""
        wfa = WalkForwardAnalysis(train_periods=50, test_periods=30)

        small_df = pd.DataFrame(
            {
                "datetime": [datetime(2024, 1, i + 1, tzinfo=UTC) for i in range(20)],
                "close": [100 + i for i in range(20)],
            }
        )

        splits = wfa.split_data(small_df)

        assert len(splits) == 0

    def test_step_size(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test step size affects number of splits."""
        wfa_small_step = WalkForwardAnalysis(train_periods=30, test_periods=10, step_periods=5)
        wfa_large_step = WalkForwardAnalysis(train_periods=30, test_periods=10, step_periods=20)

        splits_small = wfa_small_step.split_data(sample_ohlcv_df)
        splits_large = wfa_large_step.split_data(sample_ohlcv_df)

        # Smaller step size should produce more splits
        assert len(splits_small) > len(splits_large)


class TestWFARun:
    """Test run method."""

    def test_run_basic(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test basic WFA run."""
        wfa = WalkForwardAnalysis(train_periods=30, test_periods=10, step_periods=10)

        def simple_backtest(
            _data: dict[str, pd.DataFrame], _params: dict[str, Any]
        ) -> dict[str, float]:
            return {"cagr": 10.0, "mdd": -5.0, "win_rate": 60.0}

        data = {"BTC": sample_ohlcv_df}
        results = wfa.run(data, simple_backtest)

        assert "window_results" in results
        assert "summary" in results
        assert len(results["window_results"]) > 0

    def test_run_with_optimizer(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test WFA run with optimization function."""
        wfa = WalkForwardAnalysis(train_periods=30, test_periods=10, step_periods=20)

        def simple_backtest(
            _data: dict[str, pd.DataFrame], params: dict[str, Any]
        ) -> dict[str, float]:
            return {"cagr": 10.0, "mdd": -5.0, "win_rate": 60.0, **params}

        def simple_optimizer(
            _data: dict[str, pd.DataFrame],
        ) -> dict[str, Any]:
            return {"lookback": 5, "multiplier": 2}

        data = {"BTC": sample_ohlcv_df}
        results = wfa.run(data, simple_backtest, simple_optimizer)

        # Check that optimizer params are passed
        for window in results["window_results"]:
            assert window["params"] == {"lookback": 5, "multiplier": 2}

    def test_run_multiple_symbols(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test WFA with multiple symbols."""
        wfa = WalkForwardAnalysis(train_periods=30, test_periods=10, step_periods=20)

        def simple_backtest(
            _data: dict[str, pd.DataFrame], _params: dict[str, Any]
        ) -> dict[str, float]:
            return {"cagr": 10.0, "mdd": -5.0}

        data = {"BTC": sample_ohlcv_df, "ETH": sample_ohlcv_df.copy()}
        results = wfa.run(data, simple_backtest)

        assert len(results["window_results"]) > 0

    def test_summary_statistics(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test summary statistics are calculated."""
        wfa = WalkForwardAnalysis(train_periods=30, test_periods=10, step_periods=10)

        def simple_backtest(
            _data: dict[str, pd.DataFrame], _params: dict[str, Any]
        ) -> dict[str, float]:
            return {"cagr": 10.0, "mdd": -5.0, "win_rate": 60.0}

        data = {"BTC": sample_ohlcv_df}
        results = wfa.run(data, simple_backtest)

        summary = results["summary"]
        assert "avg_cagr" in summary
        assert "std_cagr" in summary
        assert "avg_mdd" in summary
