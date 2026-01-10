"""Test domain models validation."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from bt.domain.models import BacktestConfig, Position
from bt.domain.types import Amount, Fee, Percentage, Price, Quantity


class TestBacktestConfig:
    """Test BacktestConfig validation."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = BacktestConfig()

        assert config.initial_cash == Amount(Decimal("10000000"))
        assert config.fee == Fee(Decimal("0.0005"))
        assert config.slippage == Percentage(Decimal("0.0005"))
        assert config.multiplier == 2
        assert config.lookback == 5
        assert config.interval == "days"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BacktestConfig(
            initial_cash=Amount(Decimal("5000000")),
            fee=Fee(Decimal("0.001")),
            multiplier=3,
        )

        assert config.initial_cash == Amount(Decimal("5000000"))
        assert config.fee == Fee(Decimal("0.001"))
        assert config.multiplier == 3

    def test_invalid_initial_cash(self) -> None:
        """Test validation of negative initial cash."""
        with pytest.raises(ValidationError):
            BacktestConfig(initial_cash=Amount(Decimal("-1000")))

    def test_invalid_fee(self) -> None:
        """Test validation of invalid fee."""
        with pytest.raises(ValidationError):
            BacktestConfig(fee=Fee(Decimal("1.5")))

    def test_config_immutability(self) -> None:
        """Test that config is frozen."""
        config = BacktestConfig()

        with pytest.raises(ValidationError):
            config.fee = Fee(Decimal("0.002"))  # type: ignore


class TestPosition:
    """Test Position model."""

    def test_empty_position(self) -> None:
        """Test empty position."""
        position = Position(symbol="BTC")

        assert position.symbol == "BTC"
        assert position.quantity == Quantity(Decimal("0"))
        assert position.entry_price == Price(Decimal("0"))
        assert not position.is_open

    def test_open_position(self) -> None:
        """Test open position."""
        position = Position(
            symbol="BTC",
            quantity=Quantity(Decimal("0.5")),
            entry_price=Price(Decimal("50000000")),
        )

        assert position.is_open
        assert position.quantity == Quantity(Decimal("0.5"))

    def test_position_value(self) -> None:
        """Test position value calculation."""
        position = Position(
            symbol="BTC",
            quantity=Quantity(Decimal("0.5")),
            entry_price=Price(Decimal("50000000")),
        )

        current_price = Price(Decimal("55000000"))
        value = position.value(current_price)

        assert value == Amount(Decimal("27500000"))

    def test_position_pnl(self) -> None:
        """Test P&L calculation."""
        position = Position(
            symbol="BTC",
            quantity=Quantity(Decimal("0.5")),
            entry_price=Price(Decimal("50000000")),
        )

        current_price = Price(Decimal("55000000"))
        pnl = position.pnl(current_price)

        # (55M - 50M) * 0.5 = 2.5M
        assert pnl == Amount(Decimal("2500000"))
