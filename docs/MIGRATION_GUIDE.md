# 마이그레이션 가이드: Legacy → SOLID

## 빠른 시작

### 기존 코드를 그대로 사용하고 싶다면

**좋은 소식**: 기존 API와 100% 호환됩니다! 파일만 바꾸면 됩니다.

```python
# 변경 전
from bt.framework import BacktestFramework

# 변경 후 (동일한 인터페이스!)
from bt.framework.facade import BacktestFacade as BacktestFramework

# 나머지 코드는 그대로!
framework = BacktestFramework()
data = framework.load_market_data("data", ["BTC"])
results = framework.run_backtest("volatility_breakout", ["BTC"], data)
```

### 새로운 기능을 사용하고 싶다면

점진적으로 마이그레이션하세요:

1. **Step 1**: Facade 사용 (기존 코드 변경 없음)
2. **Step 2**: 세부 컴포넌트 활용 (선택적)
3. **Step 3**: 새로운 주문 타입 사용 (확장)

---

## 상세 마이그레이션 단계

### Step 1: BacktestFramework → BacktestFacade

#### Before (기존)

```python
from bt.framework import BacktestFramework

framework = BacktestFramework(config={"initial_cash": 1000000})

# 전략 목록
strategies = framework.list_available_strategies()

# 데이터 로딩
data = framework.load_market_data("data", ["BTC", "ETH"])

# 백테스트 실행
results = framework.run_backtest(
    strategy="volatility_breakout",
    symbols=["BTC"],
    data=data
)

# 보고서 생성
framework.create_performance_report(results)
```

#### After (SOLID)

**Option 1: 최소 변경 (alias 사용)**

```python
from bt.framework.facade import BacktestFacade as BacktestFramework

# 기존 코드 그대로!
framework = BacktestFramework(config={"initial_cash": 1000000})

strategies = framework.list_available_strategies()
data = framework.load_market_data("data", ["BTC", "ETH"])
results = framework.run_backtest("volatility_breakout", ["BTC"], data)
framework.create_performance_report(results)
```

**Option 2: 명확한 이름 사용**

```python
from bt.framework.facade import BacktestFacade

# Facade라는 이름 사용
facade = BacktestFacade(config={"initial_cash": 1000000})

strategies = facade.list_available_strategies()
data = facade.load_market_data("data", ["BTC", "ETH"])
results = facade.run_backtest("volatility_breakout", ["BTC"], data)
facade.create_performance_report(results)
```

**Option 3: 세부 컴포넌트 활용 (고급)**

```python
from bt.framework.facade import BacktestFacade

facade = BacktestFacade()

# 각 컴포넌트에 직접 접근 가능!
# 전략 관리는 StrategyManager에
strategies = facade.strategy_manager.list_strategies()
strategy_info = facade.strategy_manager.get_strategy_info("volatility_breakout")

# 데이터 로딩은 DataLoader에
data = facade.data_loader.load_from_directory("data", ["BTC"])
is_valid, errors = facade.data_loader.validate_data(data)

# 실행은 BacktestRunner에
strategy_instance = facade.strategy_manager.create_strategy("volatility_breakout")
results = facade.runner.run(strategy_instance, ["BTC"], data)

# 보고서는 ReportGenerator에
facade.report_generator.generate_full_report(results)
facade.report_generator.print_summary(results)
```

---

### Step 2: Portfolio → PortfolioRefactored

#### Before (기존)

```python
from bt.engine.portfolio import Portfolio

portfolio = Portfolio(
    initial_cash=1000000,
    fee=0.0005,
    slippage=0.001
)

# 매수
portfolio.buy("BTC", price=50000, quantity=0.1, date=datetime.now())

# 매도
portfolio.sell("BTC", price=55000, date=datetime.now())

# 데이터 접근
trades = portfolio.trades
equity_curve = portfolio.equity_curve
current_cash = portfolio.cash
```

#### After (SOLID)

**Option 1: 드롭인 교체 (100% 호환)**

```python
from bt.engine.portfolio_refactored import PortfolioRefactored as Portfolio

# 기존 코드 그대로!
portfolio = Portfolio(
    initial_cash=1000000,
    fee=0.0005,
    slippage=0.001
)

portfolio.buy("BTC", price=50000, quantity=0.1, date=datetime.now())
portfolio.sell("BTC", price=55000, date=datetime.now())

trades = portfolio.trades
equity_curve = portfolio.equity_curve
current_cash = portfolio.cash
```

**Option 2: 세부 컴포넌트 활용**

```python
from bt.engine.portfolio_refactored import PortfolioRefactored

portfolio = PortfolioRefactored(
    initial_cash=1000000,
    fee=0.0005,
    slippage=0.001
)

# 기본 기능은 동일
portfolio.buy("BTC", price=50000, quantity=0.1, date=datetime.now())
portfolio.sell("BTC", price=55000, date=datetime.now())

# ✨ 새로운 기능: 세부 컴포넌트 접근 가능!

# OrderExecutor를 통한 고급 기능
max_qty = portfolio.order_executor.calculate_max_quantity(
    price=50000,
    available_cash=portfolio.cash
)

# TradeRecorder를 통한 상세 분석
win_rate = portfolio.trade_recorder.get_win_rate()
winning_trades = portfolio.trade_recorder.get_winning_trades()
losing_trades = portfolio.trade_recorder.get_losing_trades()

# EquityTracker를 통한 성과 분석
total_return = portfolio.equity_tracker.get_total_return()
max_drawdown = portfolio.equity_tracker.get_max_drawdown()
max_equity = portfolio.equity_tracker.get_max_equity()
```

---

### Step 3: 새로운 주문 타입 사용

#### 기본 사용 (기존과 동일)

```python
# MarketOrder는 기존 buy/sell과 동일하게 동작
portfolio.buy("BTC", price=50000, quantity=0.1, date=datetime.now())
```

#### 고급 사용 (새로운 기능)

```python
from bt.domain.orders import (
    MarketOrder,
    LimitOrder,
    StopLossOrder,
    StopLimitOrder,
    OrderSide
)
from datetime import datetime

# 1. Limit Order (지정가 주문)
limit_buy = LimitOrder(
    symbol="BTC",
    side=OrderSide.BUY,
    quantity=0.1,
    limit_price=48000,  # 48000 이하에서만 매수
    timestamp=datetime.now()
)

# 주문 실행
success, price, cost = portfolio.order_executor.execute_order(
    limit_buy,
    market_price=50000,  # 현재가
    current_cash=portfolio.cash
)

if success:
    # 실제 매수 가격: 48000 (limit_price)
    print(f"Bought at {price}")

# 2. Stop Loss Order (손절 주문)
stop_loss = StopLossOrder(
    symbol="BTC",
    side=OrderSide.SELL,
    quantity=0.1,
    stop_price=45000,  # 45000 이하로 떨어지면 매도
    timestamp=datetime.now()
)

# 가격이 stop_price에 도달할 때 실행됨
if stop_loss.can_execute(market_price=44000):
    success, price, proceeds = portfolio.order_executor.execute_order(
        stop_loss,
        market_price=44000,
        current_cash=portfolio.cash
    )

# 3. Stop Limit Order (손절 지정가 주문)
stop_limit = StopLimitOrder(
    symbol="BTC",
    side=OrderSide.SELL,
    quantity=0.1,
    stop_price=45000,  # 손절 트리거 가격
    limit_price=44500,  # 최소 매도 가격
    timestamp=datetime.now()
)

# 가격이 45000 이하로 떨어지면 활성화되고,
# 44500 이상에서만 매도됨
```

---

### Step 4: 새로운 인터페이스 활용 (ISP)

#### Before (큰 인터페이스)

```python
from bt.interfaces.protocols import IPortfolio

def my_function(portfolio: IPortfolio):
    # IPortfolio는 너무 많은 메서드를 가짐
    # 실제로는 trades만 필요한데...
    trades = portfolio.trades
```

#### After (작은 인터페이스)

```python
from bt.interfaces.portfolio_protocols import ITradeRecorder

def my_function(trade_recorder: ITradeRecorder):
    # 필요한 인터페이스만 의존!
    trades = trade_recorder.trades
    win_rate = trade_recorder.get_win_rate()

# 호출 시
my_function(portfolio.trade_recorder)
```

**더 많은 예제**:

```python
from bt.interfaces.portfolio_protocols import (
    IPositionManager,
    ICashManager,
    IOrderExecutor,
    IEquityTracker
)

# 포지션 관리만 필요한 함수
def analyze_positions(position_manager: IPositionManager):
    for symbol, position in position_manager.positions.items():
        if position.is_open:
            print(f"{symbol}: {position.quantity}")

# 현금 관리만 필요한 함수
def check_liquidity(cash_manager: ICashManager):
    return cash_manager.cash > 10000

# 주문 실행만 필요한 함수
def place_order(executor: IOrderExecutor, symbol: str):
    executor.buy(symbol, price=50000, quantity=0.1, date=datetime.now())

# 사용
analyze_positions(portfolio)  # IPositionManager 인터페이스 사용
check_liquidity(portfolio)    # ICashManager 인터페이스 사용
place_order(portfolio, "BTC") # IOrderExecutor 인터페이스 사용
```

---

## 점진적 마이그레이션 계획

### Phase 1: 호환성 유지 (즉시 가능)

```python
# 기존 import에 alias만 추가
from bt.framework.facade import BacktestFacade as BacktestFramework
from bt.engine.portfolio_refactored import PortfolioRefactored as Portfolio

# 기존 코드 그대로 실행!
```

**작업량**: 1-2줄 변경
**리스크**: 없음 (100% 호환)
**효과**: SOLID 아키텍처로 전환

### Phase 2: 세부 컴포넌트 활용 (선택적)

```python
from bt.framework.facade import BacktestFacade

facade = BacktestFacade()

# 필요한 부분만 세부 컴포넌트 사용
win_rate = facade.runner.orchestrator.portfolio.trade_recorder.get_win_rate()
max_dd = facade.runner.orchestrator.portfolio.equity_tracker.get_max_drawdown()
```

**작업량**: 필요한 부분만 점진적으로 변경
**리스크**: 낮음
**효과**: 더 나은 성능 분석, 디버깅

### Phase 3: 새로운 기능 활용 (확장)

```python
from bt.domain.orders import LimitOrder, StopLossOrder, OrderSide

# 새로운 주문 타입 사용
limit_order = LimitOrder(...)
stop_loss = StopLossOrder(...)
```

**작업량**: 새로운 기능 추가 시
**리스크**: 없음 (기존 코드에 영향 없음)
**효과**: 더 정교한 거래 전략

---

## 체크리스트

### ✅ 마이그레이션 완료 확인

- [ ] `BacktestFacade`로 import 변경
- [ ] `PortfolioRefactored`로 import 변경 (선택)
- [ ] 모든 테스트 통과 확인
- [ ] 백테스트 결과가 기존과 동일한지 확인
- [ ] 새로운 기능 테스트 (선택)

### 🔍 확인 방법

```python
# 1. 기존 코드 실행
from bt.framework import BacktestFramework as OldFramework
old_results = OldFramework().run_simple_backtest("volatility_breakout", ["BTC"])

# 2. 새로운 코드 실행
from bt.framework.facade import BacktestFacade as NewFramework
new_results = NewFramework().run_simple_backtest("volatility_breakout", ["BTC"])

# 3. 결과 비교
assert old_results["performance"]["total_return"] == new_results["performance"]["total_return"]
assert len(old_results["trades"]) == len(new_results["trades"])
```

---

## FAQ

### Q1: 기존 코드가 깨지나요?
**A**: 아니요! 100% 호환되도록 설계했습니다.

### Q2: 성능이 느려지나요?
**A**: 아니요! 동일한 알고리즘을 사용하며 오버헤드는 무시할 수 있는 수준입니다.

### Q3: 언제 마이그레이션해야 하나요?
**A**: 원하는 시점에 언제든지. 기존 코드를 바로 바꿀 필요는 없습니다.

### Q4: 새로운 기능을 꼭 사용해야 하나요?
**A**: 아니요! 기존 방식대로 사용해도 됩니다. 필요할 때만 새로운 기능을 활용하세요.

### Q5: 테스트 코드도 변경해야 하나요?
**A**: 기존 테스트는 그대로 사용 가능합니다. 새로운 기능을 테스트하려면 추가 테스트를 작성하세요.

---

## 문제 해결

### Import 오류

```python
# 오류
from bt.framework import BacktestFacade
# ImportError: cannot import name 'BacktestFacade'

# 해결
from bt.framework.facade import BacktestFacade
```

### 타입 힌팅 오류

```python
# 오류 (Python 3.9 이하)
def func(portfolio: IFullPortfolio):
    pass

# 해결 (타입 체크 무시 또는 Python 3.10+ 사용)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from bt.interfaces.portfolio_protocols import IFullPortfolio

def func(portfolio: "IFullPortfolio"):
    pass
```

---

## 추가 리소스

- [SOLID 리팩토링 가이드](./SOLID_REFACTORING.md)
- [새로운 API 문서](./docs/api/)
- [예제 코드](./examples/solid_examples/)

---

**마지막 업데이트**: 2026-01-16
