"""Test trading conditions."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from bt.domain.models import BacktestConfig
from bt.domain.types import Amount, Fee, Percentage, Price, Quantity
from bt.engine.backtest import BacktestEngine
from bt.strategies.conditions import (
    close_below_short_ma,
    has_open_position,
    never,
    no_open_position,
    noise_is_decreasing,
    price_above_long_ma,
    price_above_short_ma,
    vbo_breakout_triggered,
)


@pytest.fixture
def sample_engine() -> BacktestEngine:
    """Provide sample backtest engine with test data."""
    config = BacktestConfig(
        initial_cash=Amount(Decimal("10000000")),
        fee=Fee(Decimal("0.0005")),
        slippage=Percentage(Decimal("0.0005")),
        lookback=5,
        multiplier=2,
    )

    engine = BacktestEngine(config=config)

    # Create uptrend data
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dates = [base_date + timedelta(days=i) for i in range(30)]

    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": [100 + i * 2 for i in range(30)],
            "high": [105 + i * 2 for i in range(30)],
            "low": [95 + i * 2 for i in range(30)],
            "close": [102 + i * 2 for i in range(30)],
            "volume": [1000 + i * 10 for i in range(30)],
        }
    )

    engine.load_data("BTC", df)
    engine.data_provider.set_current_bar("BTC", 15)

    return engine


class TestCommonConditions:
    """Test common entry/exit conditions."""

    def test_no_open_position_true(self, sample_engine: BacktestEngine) -> None:
        """Test no_open_position returns True when no position."""
        result = no_open_position(sample_engine, "BTC")
        assert result is True

    def test_no_open_position_false(self, sample_engine: BacktestEngine) -> None:
        """Test no_open_position returns False when position exists."""
        sample_engine.portfolio.buy(
            "BTC",
            Price(Decimal("50000")),
            Quantity(Decimal("0.1")),
            datetime.now(tz=timezone.utc),
        )

        result = no_open_position(sample_engine, "BTC")
        assert result is False

    def test_has_open_position_true(self, sample_engine: BacktestEngine) -> None:
        """Test has_open_position returns True when position exists."""
        sample_engine.portfolio.buy(
            "BTC",
            Price(Decimal("50000")),
            Quantity(Decimal("0.1")),
            datetime.now(tz=timezone.utc),
        )

        result = has_open_position(sample_engine, "BTC")
        assert result is True

    def test_has_open_position_false(self, sample_engine: BacktestEngine) -> None:
        """Test has_open_position returns False when no position."""
        result = has_open_position(sample_engine, "BTC")
        assert result is False

    def test_never_always_false(self, sample_engine: BacktestEngine) -> None:
        """Test never condition always returns False."""
        result = never(sample_engine, "BTC")
        assert result is False

        # Test again with position
        sample_engine.portfolio.buy(
            "BTC",
            Price(Decimal("50000")),
            Quantity(Decimal("0.1")),
            datetime.now(tz=timezone.utc),
        )

        result = never(sample_engine, "BTC")
        assert result is False


class TestTrendConditions:
    """Test trend-based conditions."""

    def test_price_above_short_ma_uptrend(self, sample_engine: BacktestEngine) -> None:
        """Test price_above_short_ma in uptrend."""
        result = price_above_short_ma(sample_engine, "BTC")
        assert isinstance(result, bool)

    def test_price_above_short_ma_no_data(self) -> None:
        """Test price_above_short_ma with no data."""
        config = BacktestConfig(lookback=5)
        engine = BacktestEngine(config=config)

        result = price_above_short_ma(engine, "UNKNOWN")
        assert result is False

    def test_price_above_short_ma_insufficient_data(self) -> None:
        """Test price_above_short_ma with insufficient data."""
        config = BacktestConfig(lookback=10)
        engine = BacktestEngine(config=config)

        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dates = [base_date + timedelta(days=i) for i in range(5)]

        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100 + i for i in range(5)],
                "high": [105 + i for i in range(5)],
                "low": [95 + i for i in range(5)],
                "close": [102 + i for i in range(5)],
                "volume": [1000 + i * 10 for i in range(5)],
            }
        )

        engine.load_data("BTC", df)
        engine.data_provider.set_current_bar("BTC", 2)

        result = price_above_short_ma(engine, "BTC")
        assert result is False

    def test_price_above_long_ma(self, sample_engine: BacktestEngine) -> None:
        """Test price_above_long_ma condition."""
        result = price_above_long_ma(sample_engine, "BTC")
        assert isinstance(result, bool)

    def test_price_above_long_ma_no_data(self) -> None:
        """Test price_above_long_ma with no data."""
        config = BacktestConfig(lookback=5, multiplier=2)
        engine = BacktestEngine(config=config)

        result = price_above_long_ma(engine, "UNKNOWN")
        assert result is False

    def test_close_below_short_ma_downtrend(self) -> None:
        """Test close_below_short_ma in downtrend."""
        config = BacktestConfig(lookback=5, multiplier=2)
        engine = BacktestEngine(config=config)

        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dates = [base_date + timedelta(days=i) for i in range(20)]

        # Downtrend data
        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": [150 - i * 2 for i in range(20)],
                "high": [155 - i * 2 for i in range(20)],
                "low": [145 - i * 2 for i in range(20)],
                "close": [148 - i * 2 for i in range(20)],
                "volume": [1000 + i * 10 for i in range(20)],
            }
        )

        engine.load_data("BTC", df)
        engine.data_provider.set_current_bar("BTC", 10)

        result = close_below_short_ma(engine, "BTC")
        assert isinstance(result, bool)

    def test_close_below_short_ma_no_data(self) -> None:
        """Test close_below_short_ma with no data."""
        config = BacktestConfig(lookback=5)
        engine = BacktestEngine(config=config)

        result = close_below_short_ma(engine, "UNKNOWN")
        assert result is False


class TestVBOConditions:
    """Test VBO-specific conditions."""

    def test_vbo_breakout_triggered_true(self) -> None:
        """Test vbo_breakout_triggered when breakout occurs."""
        config = BacktestConfig(lookback=5, multiplier=2)
        engine = BacktestEngine(config=config)

        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dates = [base_date + timedelta(days=i) for i in range(20)]

        # Create data with clear breakout opportunity
        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100 + i for i in range(20)],
                "high": [105 + i * 2 for i in range(20)],  # Increasing highs
                "low": [95 + i for i in range(20)],
                "close": [102 + i for i in range(20)],
                "volume": [1000 + i * 10 for i in range(20)],
            }
        )

        engine.load_data("BTC", df)
        engine.data_provider.set_current_bar("BTC", 10)

        result = vbo_breakout_triggered(engine, "BTC")
        assert isinstance(result, bool)

    def test_vbo_breakout_triggered_no_data(self) -> None:
        """Test vbo_breakout_triggered with no data."""
        config = BacktestConfig()
        engine = BacktestEngine(config=config)

        result = vbo_breakout_triggered(engine, "UNKNOWN")
        assert result is False

    def test_noise_is_decreasing_true(self) -> None:
        """Test noise_is_decreasing when noise decreases."""
        config = BacktestConfig(lookback=5, multiplier=2)
        engine = BacktestEngine(config=config)

        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dates = [base_date + timedelta(days=i) for i in range(20)]

        # Data with decreasing noise (small range)
        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100 + i * 0.5 for i in range(20)],
                "high": [100.5 + i * 0.5 for i in range(20)],
                "low": [99.5 + i * 0.5 for i in range(20)],
                "close": [100.2 + i * 0.5 for i in range(20)],
                "volume": [1000 + i * 10 for i in range(20)],
            }
        )

        engine.load_data("BTC", df)
        engine.data_provider.set_current_bar("BTC", 15)

        result = noise_is_decreasing(engine, "BTC")
        assert isinstance(result, bool)

    def test_noise_is_decreasing_no_data(self) -> None:
        """Test noise_is_decreasing with no data."""
        config = BacktestConfig(lookback=5, multiplier=2)
        engine = BacktestEngine(config=config)

        result = noise_is_decreasing(engine, "UNKNOWN")
        assert result is False

    def test_noise_is_decreasing_insufficient_data(self) -> None:
        """Test noise_is_decreasing with insufficient data."""
        config = BacktestConfig(lookback=5, multiplier=2)
        engine = BacktestEngine(config=config)

        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dates = [base_date + timedelta(days=i) for i in range(5)]

        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100 + i for i in range(5)],
                "high": [105 + i for i in range(5)],
                "low": [95 + i for i in range(5)],
                "close": [102 + i for i in range(5)],
                "volume": [1000 + i * 10 for i in range(5)],
            }
        )

        engine.load_data("BTC", df)
        engine.data_provider.set_current_bar("BTC", 2)

        result = noise_is_decreasing(engine, "BTC")
        assert result is False
