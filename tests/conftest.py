"""Test configuration and fixtures for pytest."""

from decimal import Decimal
from pathlib import Path

import pytest

from bt.config import Settings


@pytest.fixture
def test_settings() -> Settings:
    """Provide test settings."""
    return Settings(
        data_dir=Path("test_data"),
        log_level="DEBUG",
        log_format="text",
        api_timeout=10,
        api_rate_limit=0.01,
    )


@pytest.fixture
def sample_initial_cash() -> Decimal:
    """Provide sample initial cash."""
    return Decimal("10000000")


@pytest.fixture
def sample_fee() -> Decimal:
    """Provide sample trading fee."""
    return Decimal("0.0005")


@pytest.fixture
def sample_slippage() -> Decimal:
    """Provide sample slippage."""
    return Decimal("0.0005")
