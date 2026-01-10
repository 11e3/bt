"""Test performance metrics calculation."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from bt.domain.models import Trade
from bt.domain.types import Amount, Percentage, Price, Quantity
from bt.reporting.metrics import (
    calculate_performance_metrics,
    calculate_yearly_returns,
    print_performance_report,
    print_sample_trades,
)


@pytest.fixture
def sample_equity_curve() -> list[Decimal]:
    """Provide sample equity curve."""
    return [
        Decimal("10000000"),
        Decimal("10100000"),
        Decimal("10200000"),
        Decimal("10150000"),
        Decimal("10300000"),
        Decimal("10400000"),
        Decimal("10350000"),
        Decimal("10500000"),
        Decimal("10600000"),
        Decimal("10700000"),
    ]


@pytest.fixture
def sample_dates() -> list[datetime]:
    """Provide sample dates."""
    base_date = datetime(2024, 1, 1, tzinfo=UTC)
    return [base_date + timedelta(days=i * 30) for i in range(10)]


@pytest.fixture
def sample_trades() -> list[Trade]:
    """Provide sample trades."""
    base_date = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Trade(
            symbol="BTC",
            entry_date=base_date,
            exit_date=base_date + timedelta(days=5),
            entry_price=Price(Decimal("50000000")),
            exit_price=Price(Decimal("52000000")),
            quantity=Quantity(Decimal("0.1")),
            pnl=Amount(Decimal("200000")),
            return_pct=Percentage(Decimal("4.0")),
        ),
        Trade(
            symbol="ETH",
            entry_date=base_date + timedelta(days=10),
            exit_date=base_date + timedelta(days=15),
            entry_price=Price(Decimal("3000000")),
            exit_price=Price(Decimal("2900000")),
            quantity=Quantity(Decimal("1.0")),
            pnl=Amount(Decimal("-100000")),
            return_pct=Percentage(Decimal("-3.33")),
        ),
        Trade(
            symbol="BTC",
            entry_date=base_date + timedelta(days=20),
            exit_date=base_date + timedelta(days=25),
            entry_price=Price(Decimal("51000000")),
            exit_price=Price(Decimal("54000000")),
            quantity=Quantity(Decimal("0.1")),
            pnl=Amount(Decimal("300000")),
            return_pct=Percentage(Decimal("5.88")),
        ),
    ]


class TestCalculatePerformanceMetrics:
    """Test calculate_performance_metrics function."""

    def test_basic_metrics(
        self,
        sample_equity_curve: list[Decimal],
        sample_dates: list[datetime],
        sample_trades: list[Trade],
    ) -> None:
        """Test basic metrics calculation."""
        metrics = calculate_performance_metrics(
            equity_curve=sample_equity_curve,
            dates=sample_dates,
            trades=sample_trades,
            _initial_cash=Amount(Decimal("10000000")),
        )

        assert metrics.total_return > 0
        assert metrics.num_trades == 3
        assert metrics.final_equity > Amount(Decimal("10000000"))

    def test_cagr_calculation(
        self,
        sample_equity_curve: list[Decimal],
        sample_dates: list[datetime],
        sample_trades: list[Trade],
    ) -> None:
        """Test CAGR calculation."""
        metrics = calculate_performance_metrics(
            equity_curve=sample_equity_curve,
            dates=sample_dates,
            trades=sample_trades,
            _initial_cash=Amount(Decimal("10000000")),
        )

        # CAGR should be positive for growing equity
        assert metrics.cagr > 0

    def test_mdd_calculation(
        self,
        sample_equity_curve: list[Decimal],
        sample_dates: list[datetime],
        sample_trades: list[Trade],
    ) -> None:
        """Test MDD calculation."""
        metrics = calculate_performance_metrics(
            equity_curve=sample_equity_curve,
            dates=sample_dates,
            trades=sample_trades,
            _initial_cash=Amount(Decimal("10000000")),
        )

        # MDD should be negative (drawdown)
        assert metrics.mdd < 0

    def test_win_rate_calculation(
        self,
        sample_equity_curve: list[Decimal],
        sample_dates: list[datetime],
        sample_trades: list[Trade],
    ) -> None:
        """Test win rate calculation."""
        metrics = calculate_performance_metrics(
            equity_curve=sample_equity_curve,
            dates=sample_dates,
            trades=sample_trades,
            _initial_cash=Amount(Decimal("10000000")),
        )

        # 2 winning trades out of 3
        expected_win_rate = Percentage(Decimal("66.66666666666666666666666667"))
        assert abs(float(metrics.win_rate) - float(expected_win_rate)) < 1

    def test_profit_factor_calculation(
        self,
        sample_equity_curve: list[Decimal],
        sample_dates: list[datetime],
        sample_trades: list[Trade],
    ) -> None:
        """Test profit factor calculation."""
        metrics = calculate_performance_metrics(
            equity_curve=sample_equity_curve,
            dates=sample_dates,
            trades=sample_trades,
            _initial_cash=Amount(Decimal("10000000")),
        )

        # Profit factor: (200000 + 300000) / 100000 = 5.0
        assert float(metrics.profit_factor) == pytest.approx(5.0, rel=0.01)

    def test_no_trades(
        self,
        sample_equity_curve: list[Decimal],
        sample_dates: list[datetime],
    ) -> None:
        """Test with no trades."""
        metrics = calculate_performance_metrics(
            equity_curve=sample_equity_curve,
            dates=sample_dates,
            trades=[],
            _initial_cash=Amount(Decimal("10000000")),
        )

        assert metrics.num_trades == 0
        assert metrics.win_rate == Percentage(Decimal("0"))
        assert metrics.profit_factor == Decimal("0")

    def test_all_winning_trades(
        self,
        sample_equity_curve: list[Decimal],
        sample_dates: list[datetime],
    ) -> None:
        """Test with all winning trades."""
        base_date = datetime(2024, 1, 1, tzinfo=UTC)
        winning_trades = [
            Trade(
                symbol="BTC",
                entry_date=base_date,
                exit_date=base_date + timedelta(days=5),
                entry_price=Price(Decimal("50000000")),
                exit_price=Price(Decimal("52000000")),
                quantity=Quantity(Decimal("0.1")),
                pnl=Amount(Decimal("200000")),
                return_pct=Percentage(Decimal("4.0")),
            ),
            Trade(
                symbol="ETH",
                entry_date=base_date + timedelta(days=10),
                exit_date=base_date + timedelta(days=15),
                entry_price=Price(Decimal("3000000")),
                exit_price=Price(Decimal("3100000")),
                quantity=Quantity(Decimal("1.0")),
                pnl=Amount(Decimal("100000")),
                return_pct=Percentage(Decimal("3.33")),
            ),
        ]

        metrics = calculate_performance_metrics(
            equity_curve=sample_equity_curve,
            dates=sample_dates,
            trades=winning_trades,
            _initial_cash=Amount(Decimal("10000000")),
        )

        assert metrics.win_rate == Percentage(Decimal("100"))
        # With all winning trades, profit_factor is very large
        assert float(metrics.profit_factor) > 0

    def test_all_losing_trades(
        self,
        sample_equity_curve: list[Decimal],
        sample_dates: list[datetime],
    ) -> None:
        """Test with all losing trades."""
        base_date = datetime(2024, 1, 1, tzinfo=UTC)
        losing_trades = [
            Trade(
                symbol="BTC",
                entry_date=base_date,
                exit_date=base_date + timedelta(days=5),
                entry_price=Price(Decimal("50000000")),
                exit_price=Price(Decimal("48000000")),
                quantity=Quantity(Decimal("0.1")),
                pnl=Amount(Decimal("-200000")),
                return_pct=Percentage(Decimal("-4.0")),
            ),
        ]

        metrics = calculate_performance_metrics(
            equity_curve=sample_equity_curve,
            dates=sample_dates,
            trades=losing_trades,
            _initial_cash=Amount(Decimal("10000000")),
        )

        assert metrics.win_rate == Percentage(Decimal("0"))


class TestCalculateYearlyReturns:
    """Test calculate_yearly_returns function."""

    def test_single_year(self) -> None:
        """Test yearly returns for single year."""
        import numpy as np

        equity = np.array([10000, 10500, 11000, 11500, 12000])
        dates = [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 3, 1, tzinfo=UTC),
            datetime(2024, 6, 1, tzinfo=UTC),
            datetime(2024, 9, 1, tzinfo=UTC),
            datetime(2024, 12, 1, tzinfo=UTC),
        ]

        yearly_returns = calculate_yearly_returns(equity, dates)

        assert 2024 in yearly_returns
        assert float(yearly_returns[2024]) == pytest.approx(20.0, rel=0.01)

    def test_multiple_years(self) -> None:
        """Test yearly returns for multiple years."""
        import numpy as np

        equity = np.array([10000, 11000, 12000, 11500, 12500])
        dates = [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 12, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 6, 1, tzinfo=UTC),
            datetime(2024, 12, 1, tzinfo=UTC),
        ]

        yearly_returns = calculate_yearly_returns(equity, dates)

        assert 2023 in yearly_returns
        assert 2024 in yearly_returns


class TestPrintFunctions:
    """Test printing functions using capsys to capture stdout."""

    def test_print_performance_report(
        self,
        sample_equity_curve: list[Decimal],
        sample_dates: list[datetime],
        sample_trades: list[Trade],
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that report prints correctly."""
        # Calculate metrics first to get a valid object
        metrics = calculate_performance_metrics(
            equity_curve=sample_equity_curve,
            dates=sample_dates,
            trades=sample_trades,
            _initial_cash=Amount(Decimal("10000000")),
        )

        # Run the print function
        print_performance_report(metrics)

        # Capture output
        captured = capsys.readouterr()

        # Verify key parts of the report exist
        assert "BACKTEST RESULTS" in captured.out
        assert "Total Return" in captured.out
        assert "CAGR" in captured.out
        assert "Win Rate" in captured.out
        assert "Final Equity" in captured.out

    def test_print_sample_trades(
        self,
        sample_trades: list[Trade],
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that trade table prints correctly."""
        print_sample_trades(sample_trades)

        captured = capsys.readouterr()

        assert "SAMPLE TRADES" in captured.out
        assert "Symbol" in captured.out
        assert "BTC" in captured.out  # Check if symbol exists
        assert "P&L" in captured.out

    def test_print_sample_trades_empty(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test printing empty trades list."""
        print_sample_trades([])

        captured = capsys.readouterr()

        # Should print nothing
        assert captured.out == ""

    def test_print_sample_trades_limit(
        self,
        sample_trades: list[Trade],
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test limiting the number of trades displayed."""
        # Limit to 1 trade
        print_sample_trades(sample_trades, max_trades=1)

        captured = capsys.readouterr()

        assert "SAMPLE TRADES" in captured.out
        # Should show "... and X more trades"
        assert "more trades" in captured.out
