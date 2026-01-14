"""Shared financial calculation utilities."""

from decimal import Decimal
from typing import TYPE_CHECKING

from .constants import ONE, SAFETY_BUFFER

if TYPE_CHECKING:
    from bt.domain.models import BacktestConfig


def calculate_cost_multiplier(config: "BacktestConfig") -> Decimal:
    """Calculate cost multiplier for trading (1 + fee + slippage)."""
    return ONE + Decimal(config.fee) + Decimal(config.slippage)


def apply_safety_buffer(amount: Decimal) -> Decimal:
    """Apply safety buffer to prevent precision issues."""
    return amount * SAFETY_BUFFER


def calculate_max_affordable_quantity(
    cash: Decimal, price: Decimal, cost_multiplier: Decimal
) -> Decimal:
    """Calculate maximum quantity that can be purchased with available cash."""
    available_cash = apply_safety_buffer(cash)
    return available_cash / (price * cost_multiplier)


def calculate_execution_price(price: Decimal, slippage: float) -> Decimal:
    """Calculate execution price with slippage applied."""
    return price * (ONE + Decimal(str(slippage)))
