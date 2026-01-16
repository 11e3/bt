"""Test data generators for consistent synthetic data in tests."""

import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


class MarketDataGenerator:
    """Generate synthetic market data for testing."""

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible data."""
        self.rng = np.random.RandomState(seed)
        random.seed(seed)

    def generate_ohlcv_data(
        self,
        symbol: str,
        start_date: datetime,
        periods: int = 252,
        frequency: str = "D",
        base_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.0001,
        include_volume: bool = True,
    ) -> pd.DataFrame:
        """Generate OHLCV data with realistic price movements.

        Args:
            symbol: Trading symbol
            start_date: Start date for data
            periods: Number of periods to generate
            frequency: Data frequency ('D', 'H', 'T', etc.)
            base_price: Starting price
            volatility: Daily volatility (as fraction)
            trend: Daily trend (as fraction)
            include_volume: Whether to include volume data

        Returns:
            DataFrame with OHLCV data
        """
        # Generate timestamps
        dates = pd.date_range(start_date, periods=periods, freq=frequency)

        # Generate price movements
        returns = self.rng.normal(trend, volatility, periods)
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC data
        highs = prices * (1 + self.rng.uniform(0, 0.02, periods))
        lows = prices * (1 - self.rng.uniform(0, 0.02, periods))
        opens = prices * (1 + self.rng.normal(0, 0.005, periods))
        closes = prices

        # Ensure OHLC relationships are valid
        opens = np.clip(opens, lows, highs)
        closes = np.clip(closes, lows, highs)

        # Create DataFrame
        data = {
            "datetime": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }

        if include_volume:
            # Generate realistic volume
            base_volume = 1000000
            volume_noise = self.rng.lognormal(0, 0.5, periods)
            volumes = (base_volume * volume_noise).astype(int)
            data["volume"] = volumes

        df = pd.DataFrame(data)
        df["symbol"] = symbol

        return df

    def generate_multi_symbol_data(
        self,
        symbols: list[str],
        start_date: datetime,
        periods: int = 252,
        correlation_matrix: np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """Generate correlated multi-symbol data.

        Args:
            symbols: List of symbols to generate
            start_date: Start date
            periods: Number of periods
            correlation_matrix: Correlation matrix for symbols
            **kwargs: Additional arguments for generate_ohlcv_data

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if correlation_matrix is None:
            # Default uncorrelated
            correlation_matrix = np.eye(len(symbols))

        # Generate correlated returns
        uncorrelated_returns = self.rng.normal(0, 0.02, (periods, len(symbols)))
        cholesky = np.linalg.cholesky(correlation_matrix)
        correlated_returns = uncorrelated_returns @ cholesky.T

        data = {}
        for i, symbol in enumerate(symbols):
            df = self.generate_ohlcv_data(
                symbol=symbol, start_date=start_date, periods=periods, **kwargs
            )

            # Replace generated returns with correlated ones
            base_price = df["close"].iloc[0]
            trend = kwargs.get("trend", 0.0001)
            adjusted_returns = correlated_returns[:, i] + trend
            prices = base_price * np.exp(np.cumsum(adjusted_returns))

            # Update OHLC data
            df["close"] = prices
            df["high"] = prices * (1 + self.rng.uniform(0, 0.02, periods))
            df["low"] = prices * (1 - self.rng.uniform(0, 0.02, periods))
            df["open"] = prices * (1 + self.rng.normal(0, 0.005, periods))

            # Ensure OHLC relationships
            df["open"] = np.clip(df["open"], df["low"], df["high"])
            df["close"] = np.clip(df["close"], df["low"], df["high"])

            data[symbol] = df

        return data


class BacktestResultGenerator:
    """Generate synthetic backtest results for testing."""

    def __init__(self, seed: int = 42):
        """Initialize with random seed."""
        self.rng = np.random.RandomState(seed)

    def generate_backtest_result(
        self,
        strategy_name: str = "test_strategy",
        total_return: float = 0.15,
        sharpe_ratio: float = 1.5,
        max_drawdown: float = -0.12,
        win_rate: float = 0.55,
        num_trades: int = 50,
        periods: int = 252,
    ) -> dict[str, Any]:
        """Generate a complete synthetic backtest result.

        Args:
            strategy_name: Name of the strategy
            total_return: Total portfolio return
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown
            win_rate: Win rate percentage
            num_trades: Number of trades
            periods: Number of periods

        Returns:
            Dictionary with complete backtest result
        """
        # Generate equity curve
        dates = pd.date_range("2020-01-01", periods=periods, freq="D")
        initial_value = 1000000

        # Generate realistic equity curve with drawdowns
        returns = self.rng.normal(total_return / periods, 0.02, periods)
        equity_values = initial_value * np.exp(np.cumsum(returns))

        # Add some drawdown periods
        for _ in range(3):
            start_idx = self.rng.randint(0, periods - 20)
            end_idx = min(start_idx + self.rng.randint(10, 20), periods)
            # Create a drawdown
            dd_magnitude = abs(max_drawdown) * self.rng.uniform(0.3, 1.0)
            dd_returns = self.rng.normal(
                -dd_magnitude / (end_idx - start_idx), 0.03, end_idx - start_idx
            )
            equity_values[start_idx:end_idx] = equity_values[start_idx] * np.exp(
                np.cumsum(dd_returns)
            )

        # Generate trades
        trades = []
        # Convert dates to list to ensure proper datetime handling
        dates_list = list(dates)
        trade_indices = self.rng.choice(len(dates_list), num_trades, replace=False)
        trade_dates = sorted([dates_list[i] for i in trade_indices])

        for i, trade_date in enumerate(trade_dates):
            is_win = self.rng.random() < win_rate
            pnl = self.rng.lognormal(0, 1) * (1 if is_win else -1) * 1000
            quantity = self.rng.randint(10, 100)
            price = self.rng.uniform(50, 200)

            trades.append(
                {
                    "symbol": f"SYMBOL_{i % 5}",
                    "entry_date": trade_date,
                    "exit_date": trade_date + timedelta(days=self.rng.randint(1, 30)),
                    "entry_price": price,
                    "exit_price": price * (1 + pnl / (price * quantity)),
                    "quantity": quantity,
                    "pnl": pnl,
                    "return_pct": pnl / (price * quantity) * 100,
                }
            )

        return {
            "strategy": strategy_name,
            "configuration": {
                "initial_cash": initial_value,
                "fee_rate": 0.0005,
                "slippage_rate": 0.0005,
            },
            "performance": {
                "total_return": total_return * 100,
                "cagr": ((1 + total_return) ** (252 / periods) - 1) * 100,
                "sharpe": sharpe_ratio,
                "sortino": sharpe_ratio * 0.8,
                "mdd": max_drawdown * 100,
                "calmar": (total_return * 100) / abs(max_drawdown * 100),
                "var_95": -2.5,
                "cvar_95": -3.2,
                "win_rate": win_rate * 100,
                "profit_factor": 1.3,
                "avg_win": 1200,
                "avg_loss": -800,
                "best_trade": 2500,
                "worst_trade": -1500,
                "num_trades": num_trades,
            },
            "equity_curve": {
                "dates": dates.tolist(),
                "values": equity_values.tolist(),
            },
            "trades": trades,
        }


class PerformanceScenarioGenerator:
    """Generate specific performance scenarios for testing edge cases."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_crash_scenario(self, periods: int = 252) -> dict[str, Any]:
        """Generate a market crash scenario."""
        dates = pd.date_range("2020-01-01", periods=periods, freq="D")
        initial_value = 1000000

        # Normal market for first 150 days
        normal_returns = self.rng.normal(0.0001, 0.015, 150)
        # Crash for 30 days
        crash_returns = self.rng.normal(-0.05, 0.03, 30)
        # Recovery for remaining days
        recovery_returns = self.rng.normal(0.0005, 0.02, periods - 180)

        all_returns = np.concatenate([normal_returns, crash_returns, recovery_returns])
        equity_values = initial_value * np.exp(np.cumsum(all_returns))

        return {
            "scenario": "market_crash",
            "description": "Normal market, severe crash, then recovery",
            "equity_curve": {
                "dates": dates.tolist(),
                "values": equity_values.tolist(),
            },
            "expected_max_dd": -0.4,  # Expect ~40% drawdown
            "expected_total_return": -0.1,  # Expect ~10% loss
        }

    def generate_bull_market_scenario(self, periods: int = 252) -> dict[str, Any]:
        """Generate a strong bull market scenario."""
        dates = pd.date_range("2020-01-01", periods=periods, freq="D")
        initial_value = 1000000

        # Strong upward trend with volatility
        returns = self.rng.normal(0.001, 0.025, periods)
        equity_values = initial_value * np.exp(np.cumsum(returns))

        return {
            "scenario": "bull_market",
            "description": "Strong upward trending market",
            "equity_curve": {
                "dates": dates.tolist(),
                "values": equity_values.tolist(),
            },
            "expected_total_return": 0.8,  # Expect ~80% return
            "expected_max_dd": -0.15,  # Expect ~15% max drawdown
        }

    def generate_sideways_market_scenario(self, periods: int = 252) -> dict[str, Any]:
        """Generate a sideways/choppy market scenario."""
        dates = pd.date_range("2020-01-01", periods=periods, freq="D")
        initial_value = 1000000

        # Mean-reverting with high volatility but no trend
        returns = self.rng.normal(0, 0.03, periods)
        equity_values = initial_value * np.exp(np.cumsum(returns))

        return {
            "scenario": "sideways_market",
            "description": "High volatility with no clear trend",
            "equity_curve": {
                "dates": dates.tolist(),
                "values": equity_values.tolist(),
            },
            "expected_total_return": 0.0,  # Expect ~0% return
            "expected_max_dd": -0.25,  # Expect ~25% drawdown
        }


# Convenience functions for common test data needs


def create_test_market_data(
    symbols: list[str] = None, periods: int = 100, start_date: str = "2020-01-01"
) -> dict[str, pd.DataFrame]:
    """Create standard test market data."""
    if symbols is None:
        symbols = ["BTC", "ETH", "ADA"]

    generator = MarketDataGenerator()
    start = pd.to_datetime(start_date)

    return generator.generate_multi_symbol_data(symbols, start, periods)


def create_test_backtest_result(
    strategy_name: str = "test_strategy", **overrides
) -> dict[str, Any]:
    """Create a standard test backtest result."""
    generator = BacktestResultGenerator()
    defaults = {
        "strategy_name": strategy_name,
        "total_return": 0.15,
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.12,
        "win_rate": 0.55,
        "num_trades": 25,
        "periods": 100,
    }
    defaults.update(overrides)

    return generator.generate_backtest_result(**defaults)


def create_performance_test_scenarios() -> dict[str, dict[str, Any]]:
    """Create all performance test scenarios."""
    generator = PerformanceScenarioGenerator()
    return {
        "crash": generator.generate_crash_scenario(),
        "bull": generator.generate_bull_market_scenario(),
        "sideways": generator.generate_sideways_market_scenario(),
    }
