"""Test allocation strategies."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from bt.domain.models import BacktestConfig
from bt.domain.types import Amount, Fee, Percentage, Price, Quantity
from bt.engine.backtest import BacktestEngine
from bt.strategies.allocation import (
    all_in_allocation,
    cash_partition_allocation,
    create_cash_partition_allocator,
    equal_weight_allocation,
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

    # Add sample data
    base_date = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [base_date + timedelta(days=i) for i in range(30)]

    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": [100 + i for i in range(30)],
            "high": [105 + i for i in range(30)],
            "low": [95 + i for i in range(30)],
            "close": [102 + i for i in range(30)],
            "volume": [1000 + i * 10 for i in range(30)],
        }
    )

    engine.load_data("BTC", df)
    engine.data_provider.set_current_bar("BTC", 10)

    return engine


class TestAllInAllocation:
    """Test all_in_allocation function."""

    def test_normal_allocation(self, sample_engine: BacktestEngine) -> None:
        """Test normal all-in allocation."""
        price = Price(Decimal("50000"))
        quantity = all_in_allocation(sample_engine, "BTC", price)

        assert quantity > 0
        assert isinstance(quantity, type(Quantity(Decimal("0"))))

    def test_zero_price(self, sample_engine: BacktestEngine) -> None:
        """Test allocation with zero price."""
        price = Price(Decimal("0"))
        quantity = all_in_allocation(sample_engine, "BTC", price)

        assert quantity == Quantity(Decimal("0"))

    def test_negative_price(self, sample_engine: BacktestEngine) -> None:
        """Test allocation with negative price."""
        price = Price(Decimal("-100"))
        quantity = all_in_allocation(sample_engine, "BTC", price)

        assert quantity == Quantity(Decimal("0"))

    def test_zero_cash(self) -> None:
        """Test allocation with very low cash."""
        config = BacktestConfig(initial_cash=Amount(Decimal("1")))  # Minimum valid
        engine = BacktestEngine(config=config)

        price = Price(Decimal("50000"))
        quantity = all_in_allocation(engine, "BTC", price)

        # With minimal cash, should get minimal quantity
        assert quantity >= 0

    def test_high_price_vs_cash(self, sample_engine: BacktestEngine) -> None:
        """Test with very high price."""
        price = Price(Decimal("10000000000"))  # Extremely high
        quantity = all_in_allocation(sample_engine, "BTC", price)

        # Should return very small quantity
        assert quantity >= 0
        assert quantity < Quantity(Decimal("1"))


class TestEqualWeightAllocation:
    """Test equal_weight_allocation function."""

    def test_single_symbol(self, sample_engine: BacktestEngine) -> None:
        """Test with single symbol."""
        price = Price(Decimal("50000"))
        quantity = equal_weight_allocation(sample_engine, "BTC", price)

        assert quantity > 0

    def test_multiple_symbols(self) -> None:
        """Test with multiple symbols."""
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
        engine.data_provider.set_current_bar("BTC", 5)

        # With 2 symbols, allocation should be half per symbol
        price = Price(Decimal("50000"))
        qty_btc = equal_weight_allocation(engine, "BTC", price)

        assert qty_btc > 0

    def test_zero_symbols(self, sample_engine: BacktestEngine) -> None:
        """Test with no symbols loaded."""
        new_engine = BacktestEngine(config=BacktestConfig())

        price = Price(Decimal("50000"))
        quantity = equal_weight_allocation(new_engine, "BTC", price)

        assert quantity == Quantity(Decimal("0"))

    def test_zero_price(self, sample_engine: BacktestEngine) -> None:
        """Test with zero price."""
        price = Price(Decimal("0"))

        # Should raise ZeroDivisionError due to division by zero
        with pytest.raises(ZeroDivisionError):
            equal_weight_allocation(sample_engine, "BTC", price)


class TestCashPartitionAllocation:
    """Test cash_partition_allocation function."""

    def test_all_assets_available(self, sample_engine: BacktestEngine) -> None:
        """Test when all assets in pool are available."""
        pool = ["BTC", "ETH", "XRP"]
        price = Price(Decimal("50000"))

        quantity = cash_partition_allocation(sample_engine, "BTC", price, pool)

        # With 3 assets available, allocation = cash / 3 / price
        assert quantity > 0

    def test_some_assets_have_positions(self, sample_engine: BacktestEngine) -> None:
        """Test when some assets already have open positions."""
        pool = ["BTC", "ETH", "XRP"]
        price = Price(Decimal("50000"))

        # Open a position in BTC
        from datetime import datetime

        sample_engine.portfolio.buy(
            "BTC",
            Price(Decimal("50000")),
            Quantity(Decimal("0.1")),
            datetime.now(tz=UTC),
        )

        quantity = cash_partition_allocation(sample_engine, "ETH", price, pool)

        # ETH should still have positive allocation
        assert quantity > 0

    def test_no_remaining_assets(self, sample_engine: BacktestEngine) -> None:
        """Test when all assets in pool have positions."""
        pool = ["BTC"]
        price = Price(Decimal("50000"))

        # Open a position in BTC
        from datetime import datetime

        sample_engine.portfolio.buy(
            "BTC",
            Price(Decimal("50000")),
            Quantity(Decimal("0.1")),
            datetime.now(tz=UTC),
        )

        quantity = cash_partition_allocation(sample_engine, "BTC", price, pool)

        # No remaining assets, allocation should be 0
        assert quantity == Quantity(Decimal("0"))

    def test_empty_pool(self, sample_engine: BacktestEngine) -> None:
        """Test with empty pool."""
        pool: list[str] = []
        price = Price(Decimal("50000"))

        quantity = cash_partition_allocation(sample_engine, "BTC", price, pool)

        assert quantity == Quantity(Decimal("0"))


class TestCreateCashPartitionAllocator:
    """Test create_cash_partition_allocator factory function."""

    def test_returns_callable(self) -> None:
        """Test that factory returns a callable."""
        pool = ["BTC", "ETH"]
        allocator = create_cash_partition_allocator(pool)

        assert callable(allocator)

    def test_allocator_function(self, sample_engine: BacktestEngine) -> None:
        """Test the returned allocator function."""
        pool = ["BTC", "ETH", "XRP"]
        allocator = create_cash_partition_allocator(pool)

        price = Price(Decimal("50000"))
        quantity = allocator(sample_engine, "BTC", price)

        assert quantity > 0

    def test_multiple_allocators_independent(self, sample_engine: BacktestEngine) -> None:
        """Test that multiple allocators don't interfere."""
        pool1 = ["BTC", "ETH"]
        pool2 = ["BTC", "ETH", "XRP", "ADA"]

        allocator1 = create_cash_partition_allocator(pool1)
        allocator2 = create_cash_partition_allocator(pool2)

        price = Price(Decimal("50000"))
        qty1 = allocator1(sample_engine, "BTC", price)
        qty2 = allocator2(sample_engine, "BTC", price)

        # pool2 has more assets, so per-asset allocation should be smaller
        assert qty2 < qty1
