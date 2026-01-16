"""Consolidated strategy implementations.

Unified strategy definitions using the new component system.
Replaces scattered VBO and other strategy files.
"""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from bt.interfaces.strategy_types import (
    ConditionDict,
    IAllocation,
    ICondition,
    IPricing,
    IStrategy,
    StrategyConfig,
)
from bt.strategies.components import create_allocation, create_condition, create_pricing
from bt.utils.decimal_cache import get_decimal

if TYPE_CHECKING:
    from bt.interfaces.protocols import IBacktestEngine


class BaseStrategy(IStrategy):
    """Base class for all trading strategies."""

    def __init__(self, **config):
        self.config = config
        self.validate()

    def validate(self) -> list[str]:
        """Validate strategy configuration.

        Returns:
            List of validation errors
        """
        return []

    def get_name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__

    def get_buy_conditions(self) -> ConditionDict:
        """Get buy condition functions."""
        return {}

    def get_sell_conditions(self) -> ConditionDict:
        """Get sell condition functions."""
        return {}

    def get_buy_price_func(self) -> IPricing:
        """Get buy price function."""
        from bt.strategies.components import CurrentClosePricing

        return CurrentClosePricing()

    def get_sell_price_func(self) -> IPricing:
        """Get sell price function."""
        from bt.strategies.components import CurrentClosePricing

        return CurrentClosePricing()

    def get_allocation_func(self) -> IAllocation:
        """Get allocation function."""
        from bt.strategies.components import AllInAllocation

        return AllInAllocation()


class VolatilityBreakoutStrategy(BaseStrategy):
    """Consolidated VBO (Volatility Breakout) strategy.

    Combines:
    - Volatility breakout entry signal
    - Trend confirmation with moving averages
    - Momentum-based allocation among top assets
    """

    def validate(self) -> list[str]:
        """Validate VBO configuration."""
        errors = []

        lookback = self.config.get("lookback", 5)
        if not isinstance(lookback, int) or lookback < 1 or lookback > 100:
            errors.append("lookback must be integer between 1-100")

        multiplier = self.config.get("multiplier", 2)
        if not isinstance(multiplier, int) or multiplier < 1 or multiplier > 10:
            errors.append("multiplier must be integer between 1-10")

        k_factor = self.config.get("k_factor", 0.5)
        if not isinstance(k_factor, (int, float, Decimal)) or k_factor <= 0 or k_factor > 3:
            errors.append("k_factor must be number between 0-3")

        top_n = self.config.get("top_n", 3)
        if not isinstance(top_n, int) or top_n < 1 or top_n > 20:
            errors.append("top_n must be integer between 1-20")

        mom_lookback = self.config.get("mom_lookback", 20)
        if not isinstance(mom_lookback, int) or mom_lookback < 5 or mom_lookback > 252:
            errors.append("mom_lookback must be integer between 5-252")

        return errors

    def get_name(self) -> str:
        """Get strategy name."""
        return f"VolatilityBreakout({self.config})"

    def get_buy_conditions(self) -> ConditionDict:
        """Get VBO buy conditions."""
        lookback = self.config.get("lookback", 5)
        multiplier = self.config.get("multiplier", 2)
        k_factor = self.config.get("k_factor", 0.5)

        return {
            "no_position": create_condition("no_open_position"),
            "breakout": create_condition(
                "volatility_breakout", k_factor=k_factor, lookback=lookback
            ),
            "trend_short": create_condition("price_above_sma", lookback=lookback),
            "trend_long": create_condition("price_above_sma", lookback=lookback * multiplier),
        }

    def get_sell_conditions(self) -> ConditionDict:
        """Get VBO sell conditions."""
        lookback = self.config.get("lookback", 5)

        return {
            "trend_exit": create_condition(
                "price_above_sma", lookback=lookback, use_current_bar=True
            ),
        }

    def get_buy_price_func(self) -> IPricing:
        """Get VBO buy price function."""
        return create_pricing(
            "volatility_breakout",
            lookback=self.config.get("lookback", 5),
            k_factor=self.config.get("k_factor", 0.5),
        )

    def get_allocation_func(self) -> IAllocation:
        """Get momentum allocation function."""
        return create_allocation(
            "volatility_breakout",
            top_n=self.config.get("top_n", 3),
            mom_lookback=self.config.get("mom_lookback", 20),
        )

    def get_sell_price_func(self) -> IPricing:
        """Get sell price function (open price for next-day exit)."""
        from bt.strategies.components import CurrentOpenPricing

        return CurrentOpenPricing()


class MomentumStrategy(BaseStrategy):
    """Pure momentum strategy with equal-weight allocation."""

    def validate(self) -> list[str]:
        """Validate momentum strategy configuration."""
        errors = []

        lookback = self.config.get("lookback", 20)
        if not isinstance(lookback, int) or lookback < 5 or lookback > 252:
            errors.append("lookback must be integer between 5-252")

        return errors

    def get_name(self) -> str:
        """Get strategy name."""
        return f"Momentum({self.config})"

    def get_buy_conditions(self) -> ConditionDict:
        """Get momentum buy conditions."""
        return {
            "no_position": create_condition("no_open_position"),
        }

    def get_sell_conditions(self) -> ConditionDict:
        """Get momentum sell conditions."""
        return {}

    def get_allocation_func(self) -> IAllocation:
        """Get momentum allocation function."""
        return create_allocation(
            "equal_weight_momentum", mom_lookback=self.config.get("lookback", 20)
        )


class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold strategy."""

    def validate(self) -> list[str]:
        """Validate buy and hold configuration."""
        return []  # No configuration required

    def get_name(self) -> str:
        """Get strategy name."""
        return "BuyAndHold"

    def get_buy_conditions(self) -> ConditionDict:
        """Get buy and hold conditions."""
        return {
            "no_position": create_condition("no_open_position"),
        }

    def get_sell_conditions(self) -> ConditionDict:
        """Get buy and hold conditions (no selling)."""
        return {}

    def get_allocation_func(self) -> IAllocation:
        """Get all-in allocation function."""
        return create_allocation("all_in")


class VBOPortfolioStrategy(BaseStrategy):
    """VBO Portfolio Strategy with BTC market filter.

    A multi-asset volatility breakout strategy that:
    - Uses BTC MA20 as a market filter
    - Uses individual coin MA5 for trend confirmation
    - Allocates 1/N of total equity to each coin
    - Sells when trend exits (MA5 or BTC MA20 breakdown)

    Parameters:
        ma_short: Short MA period for individual coins (default: 5)
        btc_ma: BTC MA period for market filter (default: 20)
        noise_ratio: Volatility breakout multiplier (default: 0.5)
        btc_symbol: Symbol for BTC market filter (default: "BTC")
    """

    def validate(self) -> list[str]:
        """Validate VBO Portfolio configuration."""
        errors = []

        ma_short = self.config.get("ma_short", 5)
        if not isinstance(ma_short, int) or ma_short < 1 or ma_short > 50:
            errors.append("ma_short must be integer between 1-50")

        btc_ma = self.config.get("btc_ma", 20)
        if not isinstance(btc_ma, int) or btc_ma < 5 or btc_ma > 100:
            errors.append("btc_ma must be integer between 5-100")

        noise_ratio = self.config.get("noise_ratio", 0.5)
        if not isinstance(noise_ratio, (int, float)) or noise_ratio <= 0 or noise_ratio > 2:
            errors.append("noise_ratio must be number between 0-2")

        return errors

    def get_name(self) -> str:
        """Get strategy name."""
        ma_short = self.config.get("ma_short", 5)
        btc_ma = self.config.get("btc_ma", 20)
        noise_ratio = self.config.get("noise_ratio", 0.5)
        return f"VBOPortfolio(MA{ma_short}, BTC_MA{btc_ma}, k={noise_ratio})"

    def get_buy_conditions(self) -> ConditionDict:
        """Get VBO Portfolio buy conditions."""
        return {
            "no_position": create_condition("no_open_position"),
            "vbo_buy_signal": create_condition(
                "vbo_portfolio_buy",
                ma_short=self.config.get("ma_short", 5),
                btc_ma=self.config.get("btc_ma", 20),
                noise_ratio=self.config.get("noise_ratio", 0.5),
                btc_symbol=self.config.get("btc_symbol", "BTC"),
            ),
        }

    def get_sell_conditions(self) -> ConditionDict:
        """Get VBO Portfolio sell conditions."""
        return {
            "vbo_sell_signal": create_condition(
                "vbo_portfolio_sell",
                ma_short=self.config.get("ma_short", 5),
                btc_ma=self.config.get("btc_ma", 20),
                btc_symbol=self.config.get("btc_symbol", "BTC"),
            ),
        }

    def get_buy_price_func(self) -> IPricing:
        """Get VBO Portfolio buy price function."""
        return create_pricing(
            "vbo_portfolio",
            noise_ratio=self.config.get("noise_ratio", 0.5),
        )

    def get_sell_price_func(self) -> IPricing:
        """Get sell price function (open price)."""
        from bt.strategies.components import CurrentOpenPricing

        return CurrentOpenPricing()

    def get_allocation_func(self) -> IAllocation:
        """Get VBO Portfolio allocation function (1/N equal weight)."""
        return create_allocation("vbo_portfolio")


# === STRATEGY FACTORY ===


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
        strategies = {
            "volatility_breakout": VolatilityBreakoutStrategy,
            "vbo": VolatilityBreakoutStrategy,  # Alias for VBO
            "momentum": MomentumStrategy,
            "buy_and_hold": BuyAndHoldStrategy,
            "vbo_portfolio": VBOPortfolioStrategy,
        }

        if strategy_type not in strategies:
            available = ", ".join(strategies.keys())
            raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {available}")

        strategy_class = strategies[strategy_type]
        return strategy_class(**config)

    @staticmethod
    def list_strategies() -> list[str]:
        """List all available strategies."""
        return ["volatility_breakout", "vbo", "momentum", "buy_and_hold", "vbo_portfolio"]

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
        }

        return info.get(strategy_type, {})
