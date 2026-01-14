"""Test technical indicators."""

from decimal import Decimal

import pandas as pd
import pytest

from bt.strategies.indicators import (
    calculate_noise_ratio,
    calculate_range,
    calculate_sma,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Provide sample OHLC data."""
    return pd.DataFrame(
        {
            "open": [100, 102, 104, 103, 105],
            "high": [105, 107, 109, 108, 110],
            "low": [95, 97, 99, 98, 100],
            "close": [102, 104, 106, 105, 107],
            "volume": [1000, 1100, 1200, 1150, 1300],
        }
    )


class TestCalculateNoiseRatio:
    """Test calculate_noise_ratio function."""

    def test_basic_calculation(self, sample_df: pd.DataFrame) -> None:
        """Test basic noise ratio calculation."""
        noise = calculate_noise_ratio(sample_df)

        assert len(noise) == len(sample_df)
        assert (noise >= 0).all()  # Noise should be non-negative
        assert (noise <= 1).all()  # Noise should be <= 1

    def test_zero_range(self) -> None:
        """Test noise when high == low (no range)."""
        df = pd.DataFrame(
            {
                "open": [100, 100],
                "high": [100, 100],
                "low": [100, 100],
                "close": [100, 100],
                "volume": [1000, 1000],
            }
        )

        noise = calculate_noise_ratio(df)

        # When range is zero, noise should be 0
        assert (noise == 0).all()

    def test_full_range_noise(self) -> None:
        """Test noise when price moves full range."""
        df = pd.DataFrame(
            {
                "open": [100, 95],
                "high": [105, 100],
                "low": [95, 90],
                "close": [105, 100],
                "volume": [1000, 1000],
            }
        )

        noise = calculate_noise_ratio(df)

        # First bar: |100-105|/(105-95) = 5/10 = 0.5
        assert noise.iloc[0] == Decimal("0.5")
        # Second bar: |95-100|/(100-90) = 5/10 = 0.5
        assert noise.iloc[1] == Decimal("0.5")

    def test_high_noise(self) -> None:
        """Test noise when price is very volatile."""
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [110],
                "low": [90],
                "close": [108],
                "volume": [1000],
            }
        )

        noise = calculate_noise_ratio(df)

        # |100-108|/(110-90) = 8/20 = 0.4
        assert abs(float(noise.iloc[0]) - 0.4) < 0.001

    def test_low_noise(self) -> None:
        """Test noise when price is stable."""
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [100.5],
                "low": [99.5],
                "close": [100.1],
                "volume": [1000],
            }
        )

        noise = calculate_noise_ratio(df)

        # |100-100.1|/(100.5-99.5) = 0.1/1 = 0.1
        assert abs(float(noise.iloc[0]) - 0.1) < 0.001


class TestCalculateSMA:
    """Test calculate_sma function."""

    def test_basic_sma(self) -> None:
        """Test basic SMA calculation."""
        series = pd.Series([100, 102, 104, 106, 108])
        sma = calculate_sma(series, window=3)

        # SMA should have same length as input
        assert len(sma) == len(series)
        # First 2 values should be NaN
        assert pd.isna(sma.iloc[0])
        assert pd.isna(sma.iloc[1])
        # Third value: (100+102+104)/3 = 102
        assert sma.iloc[2] == 102
        # Fourth value: (102+104+106)/3 = 104
        assert sma.iloc[3] == 104

    def test_sma_window_1(self) -> None:
        """Test SMA with window=1 (should equal original series)."""
        series = pd.Series([100.0, 102.0, 104.0])
        sma = calculate_sma(series, window=1)

        pd.testing.assert_series_equal(series, sma)

    def test_sma_full_window(self) -> None:
        """Test SMA with window equal to series length."""
        series = pd.Series([100, 102, 104, 106])
        sma = calculate_sma(series, window=4)

        # Only last value should not be NaN
        assert pd.isna(sma.iloc[0])
        assert pd.isna(sma.iloc[1])
        assert pd.isna(sma.iloc[2])
        # Last value: (100+102+104+106)/4 = 103
        assert sma.iloc[3] == 103

    def test_sma_with_floats(self) -> None:
        """Test SMA with float values."""
        series = pd.Series([100.5, 102.5, 104.5])
        sma = calculate_sma(series, window=2)

        # (100.5+102.5)/2 = 101.5
        assert sma.iloc[1] == 101.5
        # (102.5+104.5)/2 = 103.5
        assert sma.iloc[2] == 103.5

    def test_sma_constant_values(self) -> None:
        """Test SMA with constant values."""
        series = pd.Series([100, 100, 100, 100])
        sma = calculate_sma(series, window=2)

        # All SMA values should be 100
        assert sma.iloc[1] == 100
        assert sma.iloc[2] == 100
        assert sma.iloc[3] == 100


class TestCalculateRange:
    """Test calculate_range function."""

    def test_basic_range(self, sample_df: pd.DataFrame) -> None:
        """Test basic range calculation."""
        range_series = calculate_range(sample_df)

        # Range should have same length
        assert len(range_series) == len(sample_df)
        # Each range value should be high - low
        assert range_series.iloc[0] == 10  # 105 - 95
        assert range_series.iloc[1] == 10  # 107 - 97

    def test_zero_range(self) -> None:
        """Test range when high == low."""
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [100],
                "low": [100],
                "close": [100],
                "volume": [1000],
            }
        )

        range_series = calculate_range(df)

        assert range_series.iloc[0] == 0

    def test_large_range(self) -> None:
        """Test range with large difference."""
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [200],
                "low": [50],
                "close": [150],
                "volume": [1000],
            }
        )

        range_series = calculate_range(df)

        assert range_series.iloc[0] == 150  # 200 - 50

    def test_range_with_floats(self) -> None:
        """Test range with float values."""
        df = pd.DataFrame(
            {
                "open": [100.5],
                "high": [105.5],
                "low": [99.5],
                "close": [103.2],
                "volume": [1000],
            }
        )

        range_series = calculate_range(df)

        assert range_series.iloc[0] == 6.0  # 105.5 - 99.5
