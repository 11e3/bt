"""Domain types and type aliases."""

from decimal import Decimal
from typing import TYPE_CHECKING, NewType

# 순환 참조 방지를 위해 TYPE_CHECKING 블록 사용
if TYPE_CHECKING:
    from collections.abc import Callable

    from bt.engine.backtest import BacktestEngine

# --- Existing Types ---
Price = NewType("Price", Decimal)
Quantity = NewType("Quantity", Decimal)
Amount = NewType("Amount", Decimal)
Percentage = NewType("Percentage", Decimal)
Fee = NewType("Fee", Decimal)


# --- New: Strategy Function Types (Moved from base.py) ---
if TYPE_CHECKING:
    # 조건 함수: (엔진, 심볼) -> True/False
    ConditionFunc = Callable[["BacktestEngine", str], bool]

    # 가격 함수: (엔진, 심볼) -> Price
    PriceFunc = Callable[["BacktestEngine", str], Price]

    # 비중 함수: (엔진, 심볼, 가격) -> Quantity
    AllocationFunc = Callable[["BacktestEngine", str, Price], Quantity]

    # 전략 딕셔너리 구조 정의 (선택 사항)
    class StrategyConfig(dict[str, "ConditionFunc" | "PriceFunc" | "AllocationFunc"]):
        buy_conditions: dict[str, ConditionFunc]
        sell_conditions: dict[str, ConditionFunc]
        buy_price_func: PriceFunc
        sell_price_func: PriceFunc
        allocation_func: AllocationFunc
