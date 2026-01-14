"""Test DataProvider module."""

from datetime import datetime

import pandas as pd
import pytest

from bt.engine.data_provider import DataProvider


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Provide sample OHLCV DataFrame."""
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
            "open": [100.0, 102.0, 101.0, 103.0, 104.0],
            "high": [105.0, 106.0, 104.0, 107.0, 108.0],
            "low": [99.0, 100.0, 99.0, 101.0, 102.0],
            "close": [102.0, 101.0, 103.0, 104.0, 105.0],
            "volume": [1000, 1200, 1100, 1300, 1400],
        }
    )


class TestDataProvider:
    """Test DataProvider class."""

    def test_initialization(self) -> None:
        """Test DataProvider initialization."""
        provider = DataProvider()

        assert provider.symbols == []
        assert not provider.has_more_data()

    def test_load_data_success(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test loading data successfully."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)

        assert "BTC" in provider.symbols
        assert len(provider.symbols) == 1

    def test_load_data_missing_columns(self) -> None:
        """Test loading data with missing columns."""
        provider = DataProvider()
        df = pd.DataFrame({"datetime": [1, 2, 3], "close": [100, 101, 102]})

        with pytest.raises(ValueError, match="Missing required columns"):
            provider.load_data("BTC", df)

    def test_get_bar_current(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test getting current bar."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)

        bar = provider.get_bar("BTC")
        assert bar is not None
        assert bar["close"] == 102.0

    def test_get_bar_with_offset(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test getting bar with offset."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)

        # Move to index 2
        provider.set_current_bar("BTC", 2)

        # Get previous bar (offset -1)
        prev_bar = provider.get_bar("BTC", offset=-1)
        assert prev_bar is not None
        assert prev_bar["close"] == 101.0

        # Get current bar
        current_bar = provider.get_bar("BTC", offset=0)
        assert current_bar is not None
        assert current_bar["close"] == 103.0

    def test_get_bar_out_of_bounds(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test getting bar with out-of-bounds offset."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)

        # Negative offset beyond data
        bar = provider.get_bar("BTC", offset=-10)
        assert bar is None

        # Positive offset beyond data
        bar = provider.get_bar("BTC", offset=100)
        assert bar is None

    def test_get_bar_unknown_symbol(self) -> None:
        """Test getting bar for unknown symbol."""
        provider = DataProvider()

        bar = provider.get_bar("UNKNOWN")
        assert bar is None

    def test_get_bars(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test getting multiple bars."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)
        provider.set_current_bar("BTC", 4)

        bars = provider.get_bars("BTC", 3)
        assert bars is not None
        assert len(bars) == 3
        assert bars.iloc[-1]["close"] == 105.0
        assert bars.iloc[0]["close"] == 103.0

    def test_get_bars_insufficient_data(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test getting bars when insufficient data."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)

        # At index 0, requesting 3 bars
        bars = provider.get_bars("BTC", 3)
        assert bars is not None
        assert len(bars) == 1

    def test_get_bars_unknown_symbol(self) -> None:
        """Test getting bars for unknown symbol."""
        provider = DataProvider()

        bars = provider.get_bars("UNKNOWN", 5)
        assert bars is None

    def test_has_more_data(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test has_more_data check."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)

        assert provider.has_more_data()

        # Move to last bar
        provider.set_current_bar("BTC", 4)
        assert not provider.has_more_data()

    def test_next_bar(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test advancing to next bar."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)

        bar1 = provider.get_bar("BTC")
        assert bar1 is not None
        assert bar1["close"] == 102.0

        provider.next_bar()

        bar2 = provider.get_bar("BTC")
        assert bar2 is not None
        assert bar2["close"] == 101.0

    def test_set_current_bar(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test setting current bar position."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)

        provider.set_current_bar("BTC", 3)

        bar = provider.get_bar("BTC")
        assert bar is not None
        assert bar["close"] == 104.0

    def test_set_current_bar_invalid_symbol(self) -> None:
        """Test setting bar for unknown symbol."""
        provider = DataProvider()

        with pytest.raises(ValueError, match="Symbol.*not loaded"):
            provider.set_current_bar("UNKNOWN", 0)

    def test_set_current_bar_invalid_index(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test setting invalid bar index."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)

        with pytest.raises(ValueError, match="Index.*out of bounds"):
            provider.set_current_bar("BTC", 100)

        with pytest.raises(ValueError, match="Index.*out of bounds"):
            provider.set_current_bar("BTC", -1)

    def test_get_current_datetime(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test getting current bar datetime."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)

        dt = provider.get_current_datetime("BTC")
        assert dt is not None
        assert isinstance(dt, datetime)

    def test_get_current_datetime_no_data(self) -> None:
        """Test getting datetime for unknown symbol."""
        provider = DataProvider()

        dt = provider.get_current_datetime("UNKNOWN")
        assert dt is None

    def test_multiple_symbols(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test loading multiple symbols."""
        provider = DataProvider()
        provider.load_data("BTC", sample_ohlcv_df)
        provider.load_data("ETH", sample_ohlcv_df.copy())

        assert len(provider.symbols) == 2
        assert "BTC" in provider.symbols
        assert "ETH" in provider.symbols

    def test_data_sorting(self) -> None:
        """Test that data is sorted by datetime."""
        provider = DataProvider()
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
                "open": [101, 100, 102],
                "high": [102, 101, 103],
                "low": [100, 99, 101],
                "close": [101, 100, 102],
                "volume": [1000, 1100, 1200],
            }
        )

        provider.load_data("BTC", df)

        # First bar should be earliest date
        bar = provider.get_bar("BTC")
        assert bar is not None
        assert bar["close"] == 100
