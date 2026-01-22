"""Test portfolio management."""

from datetime import datetime, timezone
from decimal import Decimal

from bt.domain.types import Amount, Fee, Percentage, Price, Quantity
from bt.engine.portfolio import Portfolio


class TestPortfolio:
    """Test Portfolio class."""

    def test_portfolio_initialization(
        self,
        sample_initial_cash: Decimal,
        sample_fee: Decimal,
        sample_slippage: Decimal,
    ) -> None:
        """Test portfolio initialization."""
        portfolio = Portfolio(
            initial_cash=Amount(sample_initial_cash),
            fee=Fee(sample_fee),
            slippage=Percentage(sample_slippage),
        )

        assert portfolio.cash == Amount(sample_initial_cash)
        assert portfolio.initial_cash == Amount(sample_initial_cash)
        assert len(portfolio.equity_curve) == 1

    def test_buy_order_success(
        self,
        sample_initial_cash: Decimal,
        sample_fee: Decimal,
        sample_slippage: Decimal,
    ) -> None:
        """Test successful buy order."""
        portfolio = Portfolio(
            initial_cash=Amount(sample_initial_cash),
            fee=Fee(sample_fee),
            slippage=Percentage(sample_slippage),
        )

        price = Price(Decimal("50000000"))
        quantity = Quantity(Decimal("0.1"))

        success = portfolio.buy("BTC", price, quantity, datetime.now(tz=timezone.utc))

        assert success
        position = portfolio.get_position("BTC")
        assert position.is_open
        assert position.quantity == quantity

    def test_buy_order_insufficient_funds(
        self,
        sample_fee: Decimal,
        sample_slippage: Decimal,
    ) -> None:
        """Test buy order with insufficient funds."""
        portfolio = Portfolio(
            initial_cash=Amount(Decimal("1000")),  # Very small amount
            fee=Fee(sample_fee),
            slippage=Percentage(sample_slippage),
        )

        price = Price(Decimal("50000000"))
        quantity = Quantity(Decimal("1"))

        success = portfolio.buy("BTC", price, quantity, datetime.now(tz=timezone.utc))

        assert not success
        position = portfolio.get_position("BTC")
        assert not position.is_open

    def test_sell_order(
        self,
        sample_initial_cash: Decimal,
        sample_fee: Decimal,
        sample_slippage: Decimal,
    ) -> None:
        """Test sell order."""
        portfolio = Portfolio(
            initial_cash=Amount(sample_initial_cash),
            fee=Fee(sample_fee),
            slippage=Percentage(sample_slippage),
        )

        # Buy first
        buy_price = Price(Decimal("50000000"))
        quantity = Quantity(Decimal("0.1"))
        portfolio.buy("BTC", buy_price, quantity, datetime.now(tz=timezone.utc))

        # Then sell (full position)
        sell_price = Price(Decimal("55000000"))
        success = portfolio.sell("BTC", sell_price, quantity, datetime.now(tz=timezone.utc))

        assert success
        position = portfolio.get_position("BTC")
        assert not position.is_open
        assert len(portfolio.trades) == 1

        # Check trade
        trade = portfolio.trades[0]
        assert trade.symbol == "BTC"
        assert float(trade.pnl) > 0  # Profitable trade

    def test_sell_without_position(
        self,
        sample_initial_cash: Decimal,
        sample_fee: Decimal,
        sample_slippage: Decimal,
    ) -> None:
        """Test sell order without open position."""
        portfolio = Portfolio(
            initial_cash=Amount(sample_initial_cash),
            fee=Fee(sample_fee),
            slippage=Percentage(sample_slippage),
        )

        success = portfolio.sell(
            "BTC", Price(Decimal("50000000")), Quantity(Decimal("1")), datetime.now(tz=timezone.utc)
        )

        assert not success
        assert len(portfolio.trades) == 0
