"""Convenience functions for common backtest operations."""

from __future__ import annotations

from typing import Any

from bt.framework.framework import BacktestFramework


def quick_backtest(
    strategy: str = "volatility_breakout", symbols: list[str] | None = None, **kwargs
) -> dict[str, Any]:
    """Quick backtest with VBO strategy and defaults."""
    framework = BacktestFramework()
    return framework.run_backtest(strategy, symbols, **kwargs)


def momentum_backtest(
    strategy: str = "momentum", symbols: list[str] | None = None, **kwargs
) -> dict[str, Any]:
    """Quick backtest with momentum strategy and defaults."""
    framework = BacktestFramework()
    return framework.run_backtest(strategy, symbols, **kwargs)


def buy_and_hold_backtest(symbols: list[str] | None = None, **kwargs) -> dict[str, Any]:
    """Quick buy and hold backtest with defaults."""
    framework = BacktestFramework()
    return framework.run_backtest("buy_and_hold", symbols, **kwargs)
