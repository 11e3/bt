"""Example custom strategy plugin demonstrating the plugin system."""

from bt.interfaces.strategy_types import AllocationFunc, ConditionFunc, PriceFunc
from bt.plugins import StrategyPlugin
from bt.strategies.implementations import BaseStrategy
from bt.utils.indicator_cache import get_indicator_cache


class MeanReversionStrategy(BaseStrategy):
    """Example mean reversion strategy plugin."""

    def validate(self) -> list[str]:
        """Validate strategy configuration."""
        errors = []

        # Validate lookback periods
        lookback = self.config.get("lookback", 20)
        if not isinstance(lookback, int) or lookback < 5 or lookback > 100:
            errors.append("lookback must be integer between 5-100")

        # Validate thresholds
        oversold_threshold = self.config.get("oversold_threshold", -2.0)
        overbought_threshold = self.config.get("overbought_threshold", 2.0)

        if oversold_threshold >= overbought_threshold:
            errors.append("oversold_threshold must be less than overbought_threshold")

        return errors

    def get_name(self) -> str:
        """Get strategy name."""
        return f"MeanReversion({self.config})"

    def get_buy_conditions(self) -> dict[str, ConditionFunc]:
        """Define buy conditions for mean reversion."""
        return {
            "no_position": lambda engine, symbol: not engine.portfolio.get_position(symbol).is_open,
            "oversold": self._create_oversold_condition(),
            "trend_filter": self._create_trend_filter_condition(),
        }

    def get_sell_conditions(self) -> dict[str, ConditionFunc]:
        """Define sell conditions for mean reversion."""
        return {
            "overbought": self._create_overbought_condition(),
            "profit_target": self._create_profit_target_condition(),
        }

    def get_buy_price_func(self) -> PriceFunc:
        """Get buy price function."""
        return lambda engine, symbol: self._calculate_buy_price(engine, symbol)

    def get_sell_price_func(self) -> PriceFunc:
        """Get sell price function."""
        return lambda engine, symbol: self._calculate_sell_price(engine, symbol)

    def get_allocation_func(self) -> AllocationFunc:
        """Get allocation function."""
        return lambda engine, symbol, price: self._calculate_allocation(engine, symbol, price)

    def _create_oversold_condition(self) -> ConditionFunc:
        """Create oversold condition based on z-score."""
        lookback = self.config.get("lookback", 20)
        threshold = self.config.get("oversold_threshold", -2.0)

        def oversold_condition(engine, symbol: str) -> bool:
            bars = engine.get_bars(symbol, lookback + 1)
            if bars is None or len(bars) < lookback + 1:
                return False

            # Calculate z-score of current price vs mean
            prices = bars["close"].iloc[:-1]  # Exclude current bar
            current_price = bars["close"].iloc[-1]

            if len(prices) < lookback:
                return False

            mean_price = prices.tail(lookback).mean()
            std_price = prices.tail(lookback).std()

            if std_price == 0:
                return False

            z_score = (current_price - mean_price) / std_price
            return z_score <= threshold

        return oversold_condition

    def _create_overbought_condition(self) -> ConditionFunc:
        """Create overbought condition based on z-score."""
        threshold = self.config.get("overbought_threshold", 2.0)

        def overbought_condition(engine, symbol: str) -> bool:
            position = engine.portfolio.get_position(symbol)
            if not position.is_open:
                return False

            # Simple overbought condition: current price > entry price * (1 + threshold)
            current_bar = engine.get_bar(symbol)
            if current_bar is None:
                return False

            current_price = float(current_bar["close"])
            entry_price = float(position.entry_price)

            return current_price >= entry_price * (1 + threshold * 0.01)  # Convert to percentage

        return overbought_condition

    def _create_trend_filter_condition(self) -> ConditionFunc:
        """Create trend filter to avoid counter-trend trades."""
        trend_lookback = self.config.get("trend_lookback", 50)

        def trend_filter(engine, symbol: str) -> bool:
            bars = engine.get_bars(symbol, trend_lookback + 1)
            if bars is None or len(bars) < trend_lookback + 1:
                return True  # Allow trade if insufficient data

            # Simple trend filter: price above SMA
            cache = get_indicator_cache()
            close_prices = bars["close"].iloc[:-1]
            sma = cache.calculate_indicator(symbol, "sma", trend_lookback, close_prices)
            current_price = bars["close"].iloc[-1]

            return current_price > sma

        return trend_filter

    def _create_profit_target_condition(self) -> ConditionFunc:
        """Create profit target condition."""
        profit_target_pct = self.config.get("profit_target_pct", 5.0)

        def profit_target(engine, symbol: str) -> bool:
            position = engine.portfolio.get_position(symbol)
            if not position.is_open:
                return False

            current_bar = engine.get_bar(symbol)
            if current_bar is None:
                return False

            current_price = float(current_bar["close"])
            entry_price = float(position.entry_price)

            return current_price >= entry_price * (1 + profit_target_pct / 100)

        return profit_target

    def _calculate_buy_price(self, engine, symbol: str) -> float:
        """Calculate buy price (limit order below current price)."""
        current_bar = engine.get_bar(symbol)
        if current_bar is None:
            return 0.0

        current_price = float(current_bar["close"])
        slippage = float(engine.config.slippage)

        return current_price * (1 + slippage)  # Pay premium for immediate execution

    def _calculate_sell_price(self, engine, symbol: str) -> float:
        """Calculate sell price (limit order above current price)."""
        current_bar = engine.get_bar(symbol)
        if current_bar is None:
            return 0.0

        current_price = float(current_bar["close"])
        slippage = float(engine.config.slippage)

        return current_price * (1 - slippage)  # Accept discount for immediate execution

    def _calculate_allocation(self, engine, symbol: str, price: float) -> float:
        """Calculate position size based on risk management."""
        if price <= 0:
            return 0.0

        # Risk-based allocation: 2% of portfolio per trade
        risk_per_trade_pct = self.config.get("risk_per_trade_pct", 2.0)
        portfolio_value = float(engine.portfolio.value)
        risk_amount = portfolio_value * (risk_per_trade_pct / 100)

        # Assume 10% stop loss from entry
        stop_loss_pct = self.config.get("stop_loss_pct", 10.0)
        risk_per_share = price * (stop_loss_pct / 100)

        if risk_per_share <= 0:
            return 0.0

        quantity = risk_amount / risk_per_share

        # Apply cost adjustments
        fee_rate = float(engine.config.fee)
        slippage_rate = float(engine.config.slippage)
        total_cost_multiplier = 1 + fee_rate + slippage_rate

        adjusted_quantity = quantity / total_cost_multiplier

        return max(0, adjusted_quantity)


class MeanReversionStrategyPlugin(StrategyPlugin):
    """Plugin wrapper for Mean Reversion Strategy."""

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Mean reversion strategy that buys oversold assets and sells overbought positions"

    @property
    def author(self) -> str:
        return "BT Framework Example"

    def get_strategy_class(self):
        return MeanReversionStrategy

    def get_strategy_config_schema(self) -> dict:
        """Return JSON schema for strategy configuration."""
        return {
            "type": "object",
            "properties": {
                "lookback": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 100,
                    "default": 20,
                    "description": "Lookback period for mean calculation",
                },
                "oversold_threshold": {
                    "type": "number",
                    "minimum": -5.0,
                    "maximum": 0.0,
                    "default": -2.0,
                    "description": "Z-score threshold for oversold condition",
                },
                "overbought_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 5.0,
                    "default": 2.0,
                    "description": "Percentage threshold for overbought condition",
                },
                "trend_lookback": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 200,
                    "default": 50,
                    "description": "Lookback period for trend filter",
                },
                "profit_target_pct": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 20.0,
                    "default": 5.0,
                    "description": "Profit target percentage",
                },
                "risk_per_trade_pct": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 10.0,
                    "default": 2.0,
                    "description": "Risk per trade as percentage of portfolio",
                },
                "stop_loss_pct": {
                    "type": "number",
                    "minimum": 1.0,
                    "maximum": 50.0,
                    "default": 10.0,
                    "description": "Stop loss percentage from entry price",
                },
            },
            "required": ["lookback"],
        }

    def initialize(self, config: dict) -> None:
        """Initialize the plugin."""
        self._config = config
        # Plugin-specific initialization could go here

    def shutdown(self) -> None:
        """Shutdown the plugin."""
        # Clean up resources if needed
        pass
