"""Simplified public API facade for the backtesting framework.

Provides a single entry point for common backtesting operations
with streamlined imports and configuration.
"""

from bt.framework.framework import BacktestFramework
from bt.framework.shortcuts import (
    buy_and_hold_backtest,
    momentum_backtest,
    quick_backtest,
)

__all__ = [
    # Main framework
    "BacktestFramework",
    # Convenience shortcuts
    "quick_backtest",
    "momentum_backtest",
    "buy_and_hold_backtest",
]
