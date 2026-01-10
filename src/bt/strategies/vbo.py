"""Volatility Breakout (VBO) Strategy.

A trend-following strategy that enters when price breaks out of a defined range
(volatility) and exits when the trend reverses.
"""

from typing import Any

from bt.strategies import conditions, pricing


def get_vbo_strategy() -> dict[str, Any]:
    """Return VBO strategy components.

    Returns:
        Dictionary containing:
        - buy_conditions
        - sell_conditions
        - buy_price_func
        - sell_price_func
    """
    return {
        "buy_conditions": {
            "no_pos": conditions.no_open_position,
            "breakout": conditions.vbo_breakout_triggered,
            "trend_short": conditions.price_above_short_ma,
            "trend_long": conditions.price_above_long_ma,
        },
        "sell_conditions": {
            "has_pos": conditions.has_open_position,
            "stop_trend": conditions.close_below_short_ma,
        },
        "buy_price_func": pricing.get_vbo_buy_price,
        "sell_price_func": pricing.get_current_close,
    }
