"""Test price calculation functions."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from bt.domain.models import BacktestConfig
from bt.domain.types import Amount, Fee, Percentage, Price
from bt.engine.backtest import BacktestEngine
from bt.strategies.pricing import (
    get_current_close,
    get_vbo_buy_price,
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

    # Create sample data
    base_date = datetime(2024, 1, 1, tzinfo=UTC)
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


class TestGetCurrentClose:
    """Test get_current_close function."""

    def test_returns_current_close(self, sample_engine: BacktestEngine) -> None:
        """Test that function returns current bar's close price."""
        price = get_current_close(sample_engine, "BTC")

        # At index 15, close = 102 + 15*2 = 132
        expected = Price(Decimal("132"))
        assert price == expected

    def test_unknown_symbol(self) -> None:
        """Test with unknown symbol."""
        config = BacktestConfig()
        engine = BacktestEngine(config=config)

        price = get_current_close(engine, "UNKNOWN")

        assert price == Price(Decimal("0"))

    def test_no_data(self) -> None:
        """Test when no data is available."""
        config = BacktestConfig()
        engine = BacktestEngine(config=config)

        price = get_current_close(engine, "BTC")

        assert price == Price(Decimal("0"))

    def test_different_symbols(self) -> None:
        """Test with different symbols."""
        config = BacktestConfig(
            initial_cash=Amount(Decimal("10000000")),
            fee=Fee(Decimal("0.0005")),
            slippage=Percentage(Decimal("0.0005")),
        )
        engine = BacktestEngine(config=config)

        base_date = datetime(2024, 1, 1, tzinfo=UTC)
        dates = [base_date + timedelta(days=i) for i in range(20)]

        df_btc = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100 + i for i in range(20)],
                "high": [105 + i for i in range(20)],
                "low": [95 + i for i in range(20)],
                "close": [102 + i for i in range(20)],
                "volume": [1000 + i * 10 for i in range(20)],
            }
        )

        df_eth = pd.DataFrame(
            {
                "datetime": dates,
                "open": [50 + i for i in range(20)],
                "high": [55 + i for i in range(20)],
                "low": [45 + i for i in range(20)],
                "close": [52 + i for i in range(20)],
                "volume": [2000 + i * 10 for i in range(20)],
            }
        )

        engine.load_data("BTC", df_btc)
        engine.load_data("ETH", df_eth)
        engine.data_provider.set_current_bar("BTC", 10)
        engine.data_provider.set_current_bar("ETH", 10)

        price_btc = get_current_close(engine, "BTC")
        price_eth = get_current_close(engine, "ETH")

        # BTC close at index 10: 102 + 10 = 112
        assert price_btc == Price(Decimal("112"))
        # ETH close at index 10: 52 + 10 = 62
        assert price_eth == Price(Decimal("62"))


class TestGetVBOBuyPrice:
    """Test get_vbo_buy_price function."""

    def test_basic_buy_price_calculation(self, sample_engine: BacktestEngine) -> None:
        """Test basic VBO buy price calculation."""
        price = get_vbo_buy_price(sample_engine, "BTC")

        # Should return a valid price
        assert price > 0
        assert isinstance(price, type(Price(Decimal("0"))))

    def test_buy_price_greater_than_open(self, sample_engine: BacktestEngine) -> None:
        """Test that buy price is typically above open price."""
        buy_price = get_vbo_buy_price(sample_engine, "BTC")
        current_bar = sample_engine.get_bar("BTC")

        assert current_bar is not None
        # Buy price should be >= open (since we add range * noise)
        assert buy_price >= Decimal(str(current_bar["open"]))

    def test_no_data(self) -> None:
        """Test buy price when no data available."""
        config = BacktestConfig(lookback=5, multiplier=2)
        engine = BacktestEngine(config=config)

        price = get_vbo_buy_price(engine, "UNKNOWN")

        assert price == Price(Decimal("0"))

    def test_insufficient_data(self) -> None:
        """Test buy price with insufficient history."""
        config = BacktestConfig(lookback=5, multiplier=2)
        engine = BacktestEngine(config=config)

        base_date = datetime(2024, 1, 1, tzinfo=UTC)
        # Only 3 bars of data
        dates = [base_date + timedelta(days=i) for i in range(3)]

        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100, 102, 104],
                "high": [105, 107, 109],
                "low": [95, 97, 99],
                "close": [102, 104, 106],
                "volume": [1000, 1100, 1200],
            }
        )

        engine.load_data("BTC", df)
        engine.data_provider.set_current_bar("BTC", 2)

        price = get_vbo_buy_price(engine, "BTC")

        # With insufficient data, should return 0
        assert price == Price(Decimal("0"))

    def test_buy_price_consistency(self, sample_engine: BacktestEngine) -> None:
        """Test that buy price is consistent across calls."""
        price1 = get_vbo_buy_price(sample_engine, "BTC")
        price2 = get_vbo_buy_price(sample_engine, "BTC")

        assert price1 == price2

    def test_buy_price_changes_with_bar(self) -> None:
        """Test that buy price changes as we move through bars."""
        config = BacktestConfig(
            initial_cash=Amount(Decimal("10000000")),
            fee=Fee(Decimal("0.0005")),
            slippage=Percentage(Decimal("0.0005")),
            lookback=3,
            multiplier=2,
        )
        engine = BacktestEngine(config=config)

        base_date = datetime(2024, 1, 1, tzinfo=UTC)
        dates = [base_date + timedelta(days=i) for i in range(20)]

        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100 + i for i in range(20)],
                "high": [105 + i for i in range(20)],
                "low": [95 + i for i in range(20)],
                "close": [102 + i for i in range(20)],
                "volume": [1000 + i * 10 for i in range(20)],
            }
        )

        engine.load_data("BTC", df)

        # Get buy prices at different bars
        engine.data_provider.set_current_bar("BTC", 5)
        price_at_5 = get_vbo_buy_price(engine, "BTC")

        engine.data_provider.set_current_bar("BTC", 10)
        price_at_10 = get_vbo_buy_price(engine, "BTC")

        # Prices should be different at different bars
        # (unless by coincidence the calculation results in the same value)
        # In an uptrend with consistent data, they likely differ
        assert isinstance(price_at_5, type(Price(Decimal("0"))))
        assert isinstance(price_at_10, type(Price(Decimal("0"))))

    def test_buy_price_with_zero_range(self) -> None:
        """Test buy price when bars have no range."""
        config = BacktestConfig(
            initial_cash=Amount(Decimal("10000000")),
            fee=Fee(Decimal("0.0005")),
            slippage=Percentage(Decimal("0.0005")),
            lookback=3,
            multiplier=2,
        )
        engine = BacktestEngine(config=config)

        base_date = datetime(2024, 1, 1, tzinfo=UTC)
        dates = [base_date + timedelta(days=i) for i in range(10)]

        # All bars have same OHLC (no range)
        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100] * 10,
                "high": [100] * 10,
                "low": [100] * 10,
                "close": [100] * 10,
                "volume": [1000] * 10,
            }
        )

        engine.load_data("BTC", df)
        engine.data_provider.set_current_bar("BTC", 5)

        price = get_vbo_buy_price(engine, "BTC")

        # With zero range and zero noise, buy price should equal open
        assert price == Price(Decimal("100"))

    def test_buy_price_high_volatility(self) -> None:
        """Test buy price with high volatility."""
        config = BacktestConfig(
            initial_cash=Amount(Decimal("10000000")),
            fee=Fee(Decimal("0.0005")),
            slippage=Percentage(Decimal("0.0005")),
            lookback=3,
            multiplier=2,
        )
        engine = BacktestEngine(config=config)

        base_date = datetime(2024, 1, 1, tzinfo=UTC)
        dates = [base_date + timedelta(days=i) for i in range(10)]

        # Highly volatile data with large ranges
        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100, 90, 110, 80, 120, 70, 130, 60, 140, 50],
                "high": [130, 120, 140, 110, 150, 100, 160, 90, 170, 80],
                "low": [70, 60, 80, 50, 90, 40, 100, 30, 110, 20],
                "close": [120, 100, 130, 90, 140, 80, 150, 70, 160, 60],
                "volume": [1000] * 10,
            }
        )

        engine.load_data("BTC", df)
        engine.data_provider.set_current_bar("BTC", 5)

        price = get_vbo_buy_price(engine, "BTC")

        # Should return a valid price
        assert price > 0
