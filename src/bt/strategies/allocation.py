"""Portfolio allocation strategies."""

from decimal import Decimal
from typing import TYPE_CHECKING

from bt.domain.types import Price, Quantity
from bt.logging import get_logger
from bt.strategies.indicators import calculate_noise_ratio

if TYPE_CHECKING:
    from collections.abc import Callable

    from bt.engine.backtest import BacktestEngine

logger = get_logger(__name__)


def all_in_allocation(engine: BacktestEngine, symbol: str, price: Price) -> Quantity:
    """Buy with all available cash accounting for costs.

    Used for Buy & Hold or Single Asset strategies.
    """
    cash = Decimal(str(engine.portfolio.cash))
    current_price = Decimal(str(price))

    if cash <= 0 or current_price <= 0:
        return Quantity(Decimal("0"))

    # Calculate cost multiplier (1 + fee + slippage)
    cost_multiplier = Decimal("1") + Decimal(engine.config.fee) + Decimal(engine.config.slippage)

    # Add a tiny safety buffer (0.1%) to prevent precision issues
    safety_buffer = Decimal("0.999")

    available_cash = cash * safety_buffer
    quantity = available_cash / (current_price * cost_multiplier)

    return Quantity(quantity)


def equal_weight_allocation(engine: BacktestEngine, symbol: str, price: Price) -> Quantity:
    """Equal weight allocation across all symbols."""
    num_symbols = len(engine.data_provider.symbols)
    if num_symbols == 0:
        return Quantity(Decimal("0"))

    target_allocation = Decimal(engine.portfolio.cash) / Decimal(num_symbols)

    cost_multiplier = Decimal("1") + Decimal(engine.config.fee) + Decimal(engine.config.slippage)
    quantity = Quantity(target_allocation / (Decimal(price) * cost_multiplier))

    return Quantity(quantity)


def cash_partition_allocation(
    engine: BacktestEngine,
    symbol: str,
    price: Price,
    pool: list[str],
) -> Quantity:
    """Divides remaining cash by the number of remaining assets."""
    remaining_assets = sum(1 for s in pool if not engine.portfolio.get_position(s).is_open)

    if remaining_assets == 0:
        return Quantity(Decimal("0"))

    target_allocation = Decimal(engine.portfolio.cash) / Decimal(remaining_assets)

    cost_multiplier = Decimal("1") + Decimal(engine.config.fee) + Decimal(engine.config.slippage)
    quantity = Quantity(target_allocation / (Decimal(price) * cost_multiplier))

    return Quantity(quantity)


def create_cash_partition_allocator(
    pool: list[str],
) -> Callable[[BacktestEngine, str, Price], Quantity]:
    """Factory for cash partition allocator."""

    def allocator(engine: BacktestEngine, symbol: str, price: Price) -> Quantity:
        return cash_partition_allocation(engine, symbol, price, pool)

    return allocator


def create_target_weight_allocator(weights: dict[str, float]):
    """
    각 종목별 목표 비중(%)을 설정하여 배분하는 함수
    weights = {"KRW-BTC": 0.1, "KRW-ETH": 0.2, ...}
    """
    def allocator(engine: BacktestEngine, symbol: str, price: Price) -> Quantity:
        # 1. 현재 총 자산 가치(Equity) 계산 (현금 + 보유 주식 평가금)
        total_equity = Decimal(str(engine.portfolio.value))

        # 2. 해당 코인의 목표 매수 금액 계산
        target_pct = weights.get(symbol, 0.0)
        target_amount = total_equity * Decimal(str(target_pct))

        # 3. 현재 보유 현금 확인
        cash = Decimal(str(engine.portfolio.cash))

        # 4. 현금이 목표 금액보다 적으면, 현금 전액 사용 (Slippage 고려 99.9%)
        buy_amount = min(target_amount, cash * Decimal("0.999"))

        # 5. 수량 계산 (금액 / 가격)
        if buy_amount <= 0:
            return Quantity(Decimal("0"))

        quantity = buy_amount / Decimal(str(price))
        return Quantity(quantity)

    return allocator


def create_momentum_allocator(top_n: int = 3, mom_lookback: int = 20) -> Callable[[BacktestEngine, str, Price], Quantity]:
    """
    최근 20일 수익률(Momentum) 상위 N개 종목에만 자산을 배분하는 로직
    
    Note: 모멘텀 계산 시 현재 봉(미확정)이 아닌 전일 봉까지만 사용하여
    Look-ahead bias를 방지합니다.
    """
    def allocator(engine: BacktestEngine, symbol: str, price: Price) -> Quantity:
        # 1. 모든 종목의 모멘텀(20일 수익률) 계산
        # 22개 봉을 가져와서 iloc[-2] (전일)와 iloc[-22] (21일 전) 비교
        # 이렇게 해야 현재 미확정 봉을 사용하지 않음
        momentum_scores: dict[str, float] = {}
        for s in engine.data_provider.symbols:
            bars = engine.get_bars(s, 22)  # 22개 봉 (현재 + 이전 21개)
            if bars is not None and len(bars) >= 22:
                # 전일 종가 / 21일 전 종가 - 1 (현재 봉 제외)
                prev_close = float(bars.iloc[-2]["close"])
                old_close = float(bars.iloc[-(mom_lookback + 2)]["close"])
                if old_close > 0:
                    ret = (prev_close / old_close) - 1
                else:
                    ret = -999.0
                momentum_scores[s] = ret
            else:
                momentum_scores[s] = -999.0  # 데이터 없으면 꼴찌

        # 2. 순위 산정 - 모멘텀 높은 순으로 정렬
        sorted_symbols = sorted(momentum_scores, key=lambda x: momentum_scores[x], reverse=True)
        top_symbols = sorted_symbols[:top_n]

        # 3. 내 종목이 Top N에 들었는지 확인
        if symbol not in top_symbols:
            return Quantity(Decimal("0"))  # 탈락한 종목은 매수 안 함

        # 4. Top N 종목끼리 자산 1/N 등분
        total_equity = Decimal(str(engine.portfolio.value))
        target_amount = total_equity / Decimal(top_n)

        # 현금 한도 내 매수
        cash = Decimal(str(engine.portfolio.cash))
        buy_amount = min(target_amount, cash * Decimal("0.999"))

        if buy_amount <= 0:
            return Quantity(Decimal("0"))

        quantity = buy_amount / Decimal(str(price))
        return Quantity(quantity)

    return allocator