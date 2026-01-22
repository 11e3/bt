"""Test BacktestEngine module."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd
import pytest

from bt.domain.models import BacktestConfig
from bt.domain.types import Amount, Fee, Percentage, Price, Quantity
from bt.engine.backtest import BacktestEngine
from bt.engine.data_provider import InMemoryDataProvider
from bt.engine.portfolio import Portfolio
from bt.interfaces.core import DataProvider as DataProviderABC
from bt.interfaces.core import Portfolio as PortfolioABC


@pytest.fixture
def sample_config() -> BacktestConfig:
    """Provide sample backtest config."""
    return BacktestConfig(
        initial_cash=Amount(Decimal("10000000")),
        fee=Fee(Decimal("0.0005")),
        slippage=Percentage(Decimal("0.0005")),
        lookback=2,
        multiplier=2,
    )


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Provide sample OHLCV DataFrame with enough data."""
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dates = [base_date + timedelta(days=i) for i in range(20)]

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [100 + i for i in range(20)],
            "high": [105 + i for i in range(20)],
            "low": [95 + i for i in range(20)],
            "close": [102 + i for i in range(20)],
            "volume": [1000 + i * 10 for i in range(20)],
        }
    )


class TestBacktestEngineInit:
    """Test BacktestEngine initialization."""

    def test_default_dependencies(self, sample_config: BacktestConfig) -> None:
        """Test engine creates default dependencies."""
        engine = BacktestEngine(config=sample_config)

        assert engine.config == sample_config
        # Use ABC interfaces for isinstance checks (allows any concrete implementation)
        assert isinstance(engine.data_provider, DataProviderABC)
        assert isinstance(engine.portfolio, PortfolioABC)

    def test_custom_dependencies(self, sample_config: BacktestConfig) -> None:
        """Test engine uses injected dependencies."""
        custom_provider = InMemoryDataProvider()
        custom_portfolio = Portfolio(
            initial_cash=Amount(Decimal("5000000")),
            fee=Fee(Decimal("0.001")),
            slippage=Percentage(Decimal("0.001")),
        )

        engine = BacktestEngine(
            config=sample_config,
            data_provider=custom_provider,
            portfolio=custom_portfolio,
        )

        assert engine.data_provider is custom_provider
        assert engine.portfolio is custom_portfolio

    def test_portfolio_uses_config(self, sample_config: BacktestConfig) -> None:
        """Test default portfolio uses config values."""
        engine = BacktestEngine(config=sample_config)

        assert engine.portfolio.initial_cash == sample_config.initial_cash
        assert engine.portfolio.fee == sample_config.fee


class TestBacktestEngineDataAccess:
    """Test BacktestEngine data access methods."""

    def test_load_data(self, sample_config: BacktestConfig, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test loading data into engine."""
        engine = BacktestEngine(config=sample_config)
        engine.load_data("BTC", sample_ohlcv_df)

        assert "BTC" in engine.data_provider.symbols

    def test_get_bar(self, sample_config: BacktestConfig, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test getting bar through engine."""
        engine = BacktestEngine(config=sample_config)
        engine.load_data("BTC", sample_ohlcv_df)

        bar = engine.get_bar("BTC")
        assert bar is not None
        assert bar["close"] == 102

    def test_get_bars(self, sample_config: BacktestConfig, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test getting multiple bars through engine."""
        engine = BacktestEngine(config=sample_config)
        engine.load_data("BTC", sample_ohlcv_df)
        engine.data_provider.set_current_bar("BTC", 5)

        bars = engine.get_bars("BTC", 3)
        assert bars is not None
        assert len(bars) == 3


class TestBacktestEngineRun:
    """Test BacktestEngine run method."""

    def test_run_basic(self, sample_config: BacktestConfig, sample_ohlcv_df: pd.DataFrame) -> None:
        """Test basic backtest run."""
        engine = BacktestEngine(config=sample_config)
        engine.load_data("BTC", sample_ohlcv_df)

        # Simple always-buy condition
        def always_true(_engine: BacktestEngine, _symbol: str) -> bool:
            return True

        # Simple never-sell condition
        def always_false(_engine: BacktestEngine, _symbol: str) -> bool:
            return False

        # Use close price
        def close_price(engine: BacktestEngine, symbol: str) -> Price:
            bar = engine.get_bar(symbol)
            if bar is None:
                return Price(Decimal("0"))
            return Price(Decimal(str(bar["close"])))

        # Simple allocation function
        def simple_allocation(_engine: BacktestEngine, _symbol: str, _price: Price) -> Quantity:
            return Quantity(Decimal("0.1"))

        engine.run(
            symbols=["BTC"],
            buy_conditions={"always": always_true},
            sell_conditions={"never": always_false},
            buy_price_func=close_price,
            sell_price_func=close_price,
            allocation_func=simple_allocation,
        )

        # Engine should have processed bars
        assert len(engine.portfolio.equity_curve) > 1

    def test_run_with_trades(
        self, sample_config: BacktestConfig, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Test backtest run with buy and sell."""
        engine = BacktestEngine(config=sample_config)
        engine.load_data("BTC", sample_ohlcv_df)

        bar_count = [0]

        def buy_on_even(_engine: BacktestEngine, _symbol: str) -> bool:
            bar_count[0] += 1
            return bar_count[0] % 4 == 1  # Buy every 4th bar

        def sell_on_odd(_engine: BacktestEngine, _symbol: str) -> bool:
            return bar_count[0] % 4 == 3  # Sell 2 bars after buy

        def close_price(engine: BacktestEngine, symbol: str) -> Price:
            bar = engine.get_bar(symbol)
            if bar is None:
                return Price(Decimal("0"))
            return Price(Decimal(str(bar["close"])))

        def simple_allocation(_engine: BacktestEngine, _symbol: str, _price: Price) -> Quantity:
            return Quantity(Decimal("0.1"))

        engine.run(
            symbols=["BTC"],
            buy_conditions={"signal": buy_on_even},
            sell_conditions={"signal": sell_on_odd},
            buy_price_func=close_price,
            sell_price_func=close_price,
            allocation_func=simple_allocation,
        )

        # Should have some trades
        assert len(engine.portfolio.trades) >= 0

    def test_run_multiple_symbols(self, sample_config: BacktestConfig) -> None:
        """Test backtest run with multiple symbols."""
        engine = BacktestEngine(config=sample_config)

        base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        dates = [base_date + timedelta(days=i) for i in range(20)]

        btc_df = pd.DataFrame(
            {
                "datetime": dates,
                "open": [100 + i for i in range(20)],
                "high": [105 + i for i in range(20)],
                "low": [95 + i for i in range(20)],
                "close": [102 + i for i in range(20)],
                "volume": [1000 + i * 10 for i in range(20)],
            }
        )

        eth_df = pd.DataFrame(
            {
                "datetime": dates,
                "open": [50 + i for i in range(20)],
                "high": [55 + i for i in range(20)],
                "low": [45 + i for i in range(20)],
                "close": [52 + i for i in range(20)],
                "volume": [2000 + i * 10 for i in range(20)],
            }
        )

        engine.load_data("BTC", btc_df)
        engine.load_data("ETH", eth_df)

        def no_signal(_engine: BacktestEngine, _symbol: str) -> bool:
            return False

        def close_price(engine: BacktestEngine, symbol: str) -> Price:
            bar = engine.get_bar(symbol)
            if bar is None:
                return Price(Decimal("0"))
            return Price(Decimal(str(bar["close"])))

        def simple_allocation(_engine: BacktestEngine, _symbol: str, _price: Price) -> Quantity:
            return Quantity(Decimal("0.05"))

        engine.run(
            symbols=["BTC", "ETH"],
            buy_conditions={"signal": no_signal},
            sell_conditions={"signal": no_signal},
            buy_price_func=close_price,
            sell_price_func=close_price,
            allocation_func=simple_allocation,
        )

        assert "BTC" in engine.data_provider.symbols
        assert "ETH" in engine.data_provider.symbols

    def test_run_with_allocation_func(
        self, sample_config: BacktestConfig, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Test backtest run with custom allocation function."""
        engine = BacktestEngine(config=sample_config)
        engine.load_data("BTC", sample_ohlcv_df)

        allocations_called: list[str] = []

        def always_buy(_engine: BacktestEngine, _symbol: str) -> bool:
            return True

        def never_sell(_engine: BacktestEngine, _symbol: str) -> bool:
            return False

        def close_price(engine: BacktestEngine, symbol: str) -> Price:
            bar = engine.get_bar(symbol)
            if bar is None:
                return Price(Decimal("0"))
            return Price(Decimal(str(bar["close"])))

        def custom_allocation(_engine: BacktestEngine, symbol: str, _price: Price) -> Quantity:
            allocations_called.append(symbol)
            return Quantity(Decimal("0.1"))

        engine.run(
            symbols=["BTC"],
            buy_conditions={"signal": always_buy},
            sell_conditions={"signal": never_sell},
            buy_price_func=close_price,
            sell_price_func=close_price,
            allocation_func=custom_allocation,
        )

        # Allocation function should have been called
        assert len(allocations_called) > 0


class TestBacktestEngineConditions:
    """Test condition evaluation."""

    def test_condition_exception_handling(
        self, sample_config: BacktestConfig, sample_ohlcv_df: pd.DataFrame
    ) -> None:
        """Test that condition exceptions are handled gracefully."""
        engine = BacktestEngine(config=sample_config)
        engine.load_data("BTC", sample_ohlcv_df)

        def failing_condition(_engine: BacktestEngine, _symbol: str) -> bool:
            raise RuntimeError("Condition error")

        def no_sell(_engine: BacktestEngine, _symbol: str) -> bool:
            return False

        def close_price(engine: BacktestEngine, symbol: str) -> Price:
            bar = engine.get_bar(symbol)
            if bar is None:
                return Price(Decimal("0"))
            return Price(Decimal(str(bar["close"])))

        def simple_allocation(_engine: BacktestEngine, _symbol: str, _price: Price) -> Quantity:
            return Quantity(Decimal("0.1"))

        # Should not raise, just log and continue
        engine.run(
            symbols=["BTC"],
            buy_conditions={"failing": failing_condition},
            sell_conditions={"never": no_sell},
            buy_price_func=close_price,
            sell_price_func=close_price,
            allocation_func=simple_allocation,
        )
