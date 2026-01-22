"""Strategy factory for creating strategy instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bt.strategies.implementations.strategies import (
    BaseStrategy,
    BuyAndHoldStrategy,
    MomentumStrategy,
    VBOPortfolioStrategy,
    VBORegimeStrategy,
    VBOSingleCoinStrategy,
    VolatilityBreakoutStrategy,
)

if TYPE_CHECKING:
    from bt.interfaces.strategy_types import IStrategy


class StrategyFactory:
    """Factory for creating strategy instances."""

    @staticmethod
    def create_strategy(strategy_type: str, **config) -> IStrategy:
        """Create strategy instance by type.

        Args:
            strategy_type: Type of strategy to create
            **config: Strategy configuration parameters

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy type is unknown
        """
        strategies: dict[str, type[BaseStrategy]] = {
            "volatility_breakout": VolatilityBreakoutStrategy,
            "vbo": VolatilityBreakoutStrategy,  # Alias for VBO
            "momentum": MomentumStrategy,
            "buy_and_hold": BuyAndHoldStrategy,
            "vbo_single_coin": VBOSingleCoinStrategy,
            "vbo_portfolio": VBOPortfolioStrategy,
            "vbo_regime": VBORegimeStrategy,
        }

        if strategy_type not in strategies:
            available = ", ".join(strategies.keys())
            raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {available}")

        strategy_class = strategies[strategy_type]
        return strategy_class(**config)

    @staticmethod
    def list_strategies() -> list[str]:
        """List all available strategies."""
        return [
            "volatility_breakout",
            "vbo",
            "momentum",
            "buy_and_hold",
            "vbo_single_coin",
            "vbo_portfolio",
            "vbo_regime",
        ]

    @staticmethod
    def get_strategy_info(strategy_type: str) -> dict:
        """Get information about a strategy type.

        Args:
            strategy_type: Strategy type to get info for

        Returns:
            Dictionary with strategy information
        """
        info = {
            "volatility_breakout": {
                "name": "Volatility Breakout",
                "description": "Breakout strategy with trend confirmation and momentum allocation",
                "parameters": ["lookback", "multiplier", "k_factor", "top_n", "mom_lookback"],
                "default_config": {
                    "lookback": 5,
                    "multiplier": 2,
                    "k_factor": 0.5,
                    "top_n": 3,
                    "mom_lookback": 20,
                },
            },
            "vbo": {
                "name": "VBO (Volatility Breakout)",
                "description": "Alias for volatility breakout strategy",
                "parameters": ["lookback", "multiplier", "k_factor", "top_n", "mom_lookback"],
                "default_config": {
                    "lookback": 5,
                    "multiplier": 2,
                    "k_factor": 0.5,
                    "top_n": 3,
                    "mom_lookback": 20,
                },
            },
            "momentum": {
                "name": "Momentum",
                "description": "Pure momentum strategy with equal-weight allocation",
                "parameters": ["lookback"],
                "default_config": {"lookback": 20},
            },
            "buy_and_hold": {
                "name": "Buy and Hold",
                "description": "Simple buy and hold strategy",
                "parameters": [],
                "default_config": {},
            },
            "vbo_single_coin": {
                "name": "VBO Single Coin",
                "description": "Single-asset VBO strategy with BTC market filter and all-in allocation",
                "parameters": ["ma_short", "btc_ma", "noise_ratio", "btc_symbol"],
                "default_config": {
                    "ma_short": 5,
                    "btc_ma": 20,
                    "noise_ratio": 0.5,
                    "btc_symbol": "BTC",
                },
            },
            "vbo_portfolio": {
                "name": "VBO Portfolio",
                "description": "Multi-asset VBO strategy with BTC market filter and 1/N allocation",
                "parameters": ["ma_short", "btc_ma", "noise_ratio", "btc_symbol"],
                "default_config": {
                    "ma_short": 5,
                    "btc_ma": 20,
                    "noise_ratio": 0.5,
                    "btc_symbol": "BTC",
                },
            },
            "vbo_regime": {
                "name": "VBO Regime",
                "description": "Multi-asset VBO strategy with ML regime model filter and 1/N allocation",
                "parameters": ["ma_short", "noise_ratio", "btc_symbol", "model_path"],
                "default_config": {
                    "ma_short": 5,
                    "noise_ratio": 0.5,
                    "btc_symbol": "BTC",
                    "model_path": "",  # Required
                },
            },
        }

        return info.get(strategy_type, {})
