"""SOLID 원칙 마이그레이션 예제.

기존 코드와 새로운 SOLID 구조의 비교 및 사용 예제를 보여줍니다.
"""

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd


def example_1_basic_migration():
    """예제 1: 기본 마이그레이션 (최소 변경)"""
    print("\n" + "=" * 80)
    print("예제 1: 기본 마이그레이션 - Import만 변경")
    print("=" * 80)

    # ============================================================
    # BEFORE: 기존 코드
    # ============================================================
    # from bt.framework import BacktestFramework

    # ============================================================
    # AFTER: 새로운 코드 (alias 사용으로 100% 호환)
    # ============================================================
    from bt.framework.facade import BacktestFacade as BacktestFramework

    # 샘플 데이터 생성
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    prices = 50000 * np.exp(np.cumsum(np.random.normal(0.001, 0.03, len(dates))))
    data = pd.DataFrame(
        {
            "datetime": dates,
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            "high": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            "low": prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            "close": prices,
            "volume": np.random.lognormal(15, 1, len(dates)).astype(int),
        }
    )

    # 기존 코드 그대로 실행!
    framework = BacktestFramework()
    results = framework.run_backtest(
        strategy="volatility_breakout", symbols=["BTC"], data={"BTC": data}
    )

    print("✅ 백테스트 완료!")
    print(f"   총 거래 수: {len(results.get('trades', []))}")
    print(f"   총 수익률: {results.get('performance', {}).get('total_return', 0):.2f}%")
    print("\n💡 Import만 변경했지만 SOLID 아키텍처로 실행됨!")


def example_2_component_access():
    """예제 2: 세부 컴포넌트 접근"""
    print("\n" + "=" * 80)
    print("예제 2: 세부 컴포넌트 접근 - 새로운 기능 활용")
    print("=" * 80)

    from bt.framework.facade import BacktestFacade

    facade = BacktestFacade()

    # ============================================================
    # 새로운 기능: 각 컴포넌트에 직접 접근 가능!
    # ============================================================

    # 1. StrategyManager로 전략 관리
    print("\n1️⃣ StrategyManager 사용:")
    strategies = facade.strategy_manager.list_strategies()
    print(f"   사용 가능한 전략: {strategies}")

    strategy_info = facade.strategy_manager.get_strategy_info("volatility_breakout")
    if strategy_info:
        print(f"   전략 정보: {strategy_info.get('category', 'N/A')}")

    # 2. DataLoader로 데이터 검증
    print("\n2️⃣ DataLoader 사용:")
    # 샘플 데이터
    sample_data = {
        "BTC": pd.DataFrame(
            {
                "datetime": pd.date_range("2020-01-01", periods=10),
                "open": [50000] * 10,
                "high": [51000] * 10,
                "low": [49000] * 10,
                "close": [50500] * 10,
                "volume": [1000000] * 10,
            }
        )
    }

    is_valid, errors = facade.data_loader.validate_data(sample_data)
    print(f"   데이터 유효성: {'✅ 유효' if is_valid else '❌ 오류'}")
    if errors:
        print(f"   오류 목록: {errors}")

    print("\n💡 각 컴포넌트가 명확한 책임을 가지고 독립적으로 동작!")


def example_3_new_order_types():
    """예제 3: 새로운 주문 타입 사용 (OCP)"""
    print("\n" + "=" * 80)
    print("예제 3: 새로운 주문 타입 - Open/Closed Principle")
    print("=" * 80)

    from bt.domain.orders import LimitOrder, MarketOrder, OrderSide, StopLossOrder
    from bt.engine.portfolio_refactored import PortfolioRefactored

    # Portfolio 생성
    PortfolioRefactored(
        initial_cash=Decimal("1000000"), fee=Decimal("0.0005"), slippage=Decimal("0.001")
    )

    print("\n🔧 4가지 주문 타입 사용 가능:")

    # 1. Market Order (기존과 동일)
    print("\n1️⃣ Market Order (시장가):")
    market_order = MarketOrder("BTC", OrderSide.BUY, Decimal("0.1"), datetime.now(UTC))
    print(f"   주문 타입: {market_order.get_order_type().value}")
    print(f"   실행 가능: {market_order.can_execute(Decimal('50000'))}")

    # 2. Limit Order (새로운 기능!)
    print("\n2️⃣ Limit Order (지정가):")
    limit_order = LimitOrder(
        "BTC", OrderSide.BUY, Decimal("0.1"), Decimal("48000"), datetime.now(UTC)
    )
    market_price_high = Decimal("50000")
    market_price_low = Decimal("47000")

    print("   지정가: 48,000")
    print(f"   현재가 50,000에서 실행 가능: {limit_order.can_execute(market_price_high)}")
    print(f"   현재가 47,000에서 실행 가능: {limit_order.can_execute(market_price_low)}")

    # 3. Stop Loss Order (새로운 기능!)
    print("\n3️⃣ Stop Loss Order (손절):")
    stop_loss = StopLossOrder(
        "BTC", OrderSide.SELL, Decimal("0.1"), Decimal("45000"), datetime.now(UTC)
    )
    print("   손절가: 45,000")
    print(f"   현재가 50,000에서 실행 가능: {stop_loss.can_execute(Decimal('50000'))}")
    print(f"   현재가 44,000에서 실행 가능: {stop_loss.can_execute(Decimal('44000'))}")

    print("\n💡 기존 코드 수정 없이 새로운 주문 타입 추가 가능! (OCP)")


def example_4_portfolio_components():
    """예제 4: Portfolio 컴포넌트 활용 (SRP)"""
    print("\n" + "=" * 80)
    print("예제 4: Portfolio 컴포넌트 - Single Responsibility Principle")
    print("=" * 80)

    from bt.engine.portfolio_refactored import PortfolioRefactored

    # Portfolio 생성
    portfolio = PortfolioRefactored(
        initial_cash=Decimal("1000000"), fee=Decimal("0.0005"), slippage=Decimal("0.001")
    )

    # 샘플 거래 시뮬레이션
    portfolio.buy("BTC", Decimal("50000"), Decimal("0.1"), datetime.now(UTC))
    portfolio.sell("BTC", Decimal("55000"), datetime.now(UTC))

    print("\n📊 각 컴포넌트에 독립적으로 접근:")

    # 1. OrderExecutor 접근
    print("\n1️⃣ OrderExecutor (주문 실행):")
    max_qty = portfolio.order_executor.calculate_max_quantity(Decimal("50000"), portfolio.cash)
    print(f"   최대 매수 가능 수량: {float(max_qty):.6f} BTC")

    # 2. TradeRecorder 접근
    print("\n2️⃣ TradeRecorder (거래 기록):")
    portfolio.trade_recorder.get_all_trades()
    print(f"   총 거래 수: {portfolio.trade_recorder.get_trade_count()}")
    print(f"   승률: {portfolio.trade_recorder.get_win_rate():.2f}%")
    print(f"   승리한 거래: {portfolio.trade_recorder.get_win_count()}")
    print(f"   손실 거래: {portfolio.trade_recorder.get_loss_count()}")

    # 3. EquityTracker 접근
    print("\n3️⃣ EquityTracker (자산 추적):")
    print(f"   현재 자산: {float(portfolio.equity_tracker.get_current_equity()):,.0f}")
    print(f"   최대 자산: {float(portfolio.equity_tracker.get_max_equity()):,.0f}")
    print(f"   총 수익률: {float(portfolio.equity_tracker.get_total_return()) * 100:.2f}%")

    print("\n💡 각 컴포넌트가 하나의 책임만 가짐! (SRP)")


def example_5_interface_segregation():
    """예제 5: 인터페이스 분리 (ISP)"""
    print("\n" + "=" * 80)
    print("예제 5: 인터페이스 분리 - Interface Segregation Principle")
    print("=" * 80)

    from bt.engine.portfolio_refactored import PortfolioRefactored
    from bt.interfaces.portfolio_protocols import (
        ICashManager,
        IPositionManager,
        ITradeRecorder,
    )

    portfolio = PortfolioRefactored(
        initial_cash=Decimal("1000000"), fee=Decimal("0.0005"), slippage=Decimal("0.001")
    )

    # ============================================================
    # 함수들이 필요한 인터페이스만 의존
    # ============================================================

    def check_liquidity(cash_manager: ICashManager) -> bool:
        """현금 관리만 필요한 함수"""
        return cash_manager.cash > 10000

    def analyze_positions(position_manager: IPositionManager) -> int:
        """포지션 관리만 필요한 함수"""
        open_positions = [p for p in position_manager.positions.values() if p.is_open]
        return len(open_positions)

    def get_trade_summary(trade_recorder: ITradeRecorder) -> dict:
        """거래 기록만 필요한 함수"""
        return {
            "total_trades": len(trade_recorder.trades),
            "win_rate": trade_recorder.get_win_rate()
            if hasattr(trade_recorder, "get_win_rate")
            else 0,
        }

    # 각 함수는 필요한 인터페이스만 받음
    print("\n📋 인터페이스별 함수 호출:")
    print(f"   유동성 충분: {check_liquidity(portfolio)}")
    print(f"   열린 포지션: {analyze_positions(portfolio)}")
    trade_summary = get_trade_summary(portfolio.trade_recorder)
    print(f"   거래 요약: {trade_summary}")

    print("\n💡 클라이언트가 필요한 메서드만 의존! (ISP)")


def example_6_full_comparison():
    """예제 6: 전체 비교 - 기존 vs SOLID"""
    print("\n" + "=" * 80)
    print("예제 6: 전체 비교 - Legacy vs SOLID")
    print("=" * 80)

    # 샘플 데이터
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = 50000 * np.exp(np.cumsum(np.random.normal(0.001, 0.03, len(dates))))
    {
        "BTC": pd.DataFrame(
            {
                "datetime": dates,
                "open": prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
                "high": prices * (1 + np.random.uniform(0, 0.02, len(dates))),
                "low": prices * (1 - np.random.uniform(0, 0.02, len(dates))),
                "close": prices,
                "volume": np.random.lognormal(15, 1, len(dates)).astype(int),
            }
        )
    }

    print("\n📊 기존 방식 vs SOLID 방식 비교:")

    # 기존 방식 시뮬레이션
    print("\n🔴 Legacy (모놀리식):")
    print("   - BacktestFramework: 255 lines, 6가지 책임")
    print("   - Portfolio: 285 lines, 4가지 책임")
    print("   - 총 2개의 큰 클래스")

    # SOLID 방식
    print("\n🟢 SOLID (분리됨):")
    print("   - BacktestFacade: 조정만")
    print("     └─ BacktestRunner: 실행만")
    print("     └─ StrategyManager: 전략 관리만")
    print("     └─ DataLoader: 데이터 로딩만")
    print("     └─ ReportGenerator: 보고서만")
    print("   - PortfolioRefactored: 상태 관리만")
    print("     └─ OrderExecutor: 주문 실행만")
    print("     └─ TradeRecorder: 거래 기록만")
    print("     └─ EquityTracker: 자산 추적만")
    print("   - 총 15개의 작은 클래스")

    print("\n✨ 개선 효과:")
    print("   ✅ 각 클래스가 명확한 책임")
    print("   ✅ 테스트 용이 (모킹 쉬움)")
    print("   ✅ 확장 가능 (새 기능 추가 쉬움)")
    print("   ✅ 유지보수 용이 (변경 범위 제한)")
    print("   ✅ 100% 호환 (기존 코드 그대로 작동)")


def main():
    """모든 예제 실행"""
    print("\n" + "=" * 80)
    print("SOLID 원칙 리팩토링 - 마이그레이션 예제")
    print("=" * 80)

    try:
        example_1_basic_migration()
        example_2_component_access()
        example_3_new_order_types()
        example_4_portfolio_components()
        example_5_interface_segregation()
        example_6_full_comparison()

        print("\n" + "=" * 80)
        print("✅ 모든 예제 완료!")
        print("=" * 80)
        print("\n📘 더 많은 정보:")
        print("   - SOLID_REFACTORING.md: 종합 가이드")
        print("   - MIGRATION_GUIDE.md: 마이그레이션 가이드")
        print("\n")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
