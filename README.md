# BT - 암호화폐 백테스팅 프레임워크 (Cryptocurrency Backtesting Framework)

최신 Python 3.14+ 기능을 활용한 이벤트 기반(Event-Driven) 암호화폐 퀀트 트레이딩 백테스팅 엔진입니다. 클린 아키텍처를 지향하며 타입 안정성, 확장성, 그리고 검증(Validation)에 중점을 두었습니다.

## 📈 백테스트 결과 (Backtest Results)

### VBO (Volatility Breakout) 전략 - 최적화 버전
| Metric | Value |
|--------|-------|
| **CAGR** | 121.34% |
| **MDD** | -24.45% |
| **Sortino Ratio** | 3.28 |
| **Win Rate** | 35.34% |
| **Profit Factor** | 1.69 |
| **기간** | 2017-2026 (9년) |

### 시각화 (Visualizations)

<details>
<summary>📊 수익 곡선 (Equity Curve)</summary>

![Equity Curve](output/equity_curve.png)
</details>

<details>
<summary>📅 연도별 수익률 (Yearly Returns)</summary>

![Yearly Returns](output/yearly_returns.png)
</details>

<details>
<summary>🔄 WFA 검증 결과 (Walk Forward Analysis)</summary>

![WFA Results](output/wfa_results.png)
</details>

<details>
<summary>📉 시장 국면 분석 (Market Regime Analysis)</summary>

![Market Regime](output/market_regime.png)
</details>

### 주요 발견 (Key Findings)
1. **노이즈 필터 제거**: CAGR ~70% → ~120% 개선 (MDD 유지)
2. **시장 국면 민감도**: 상승장에서 탁월 (2017: +514%), 하락장에서 손실 (2022: -11%)
3. **WFA 검증**: 27개 윈도우 중 67%가 양수 수익률

## ✨ 주요 기능 (Features)

- **이벤트 기반 아키텍처 (Event-Driven Architecture)**: 실제 거래 환경과 유사하게 캔들(Bar) 단위로 데이터를 처리하여 정밀한 시뮬레이션을 제공합니다.
- **조립 가능한 전략 (Composable Strategies)**: 매수/매도 조건, 가격 결정 로직, 자산 배분(Allocation) 로직을 레고 블록처럼 조립하여 전략을 구성할 수 있습니다.
- **타입 안정성 (Type Safety)**: `Pydantic`과 `Type Hints`를 전면 도입하여 런타임 오류를 최소화하고 개발 경험을 향상시켰습니다.
- **정밀한 금융 연산**: 부동소수점 오차 방지를 위해 모든 금융 계산에 `Decimal` 타입을 사용합니다.
- **스마트 데이터 수집**: Upbit API를 연동하여 지수형 백오프(Exponential Backoff), 속도 제한(Rate Limiting), 증분 업데이트(Incremental Update) 기능을 갖춘 데이터 수집기를 제공합니다.
- **강력한 검증 도구**: 과적합(Overfitting) 방지를 위한 **Walk Forward Analysis (WFA)** 및 **CPCV (Combinatorial Purged Cross-Validation)** 기능을 내장하고 있습니다.

## 📁 프로젝트 구조 (Project Structure)

```text
bt/
├── pyproject.toml          # 프로젝트 의존성 및 설정 (Python 3.14+)
├── src/bt/                 # 핵심 소스 코드
│   ├── config.py           # 설정 관리 (Pydantic Settings)
│   ├── logging.py          # 구조화된 로깅
│   ├── domain/             # 도메인 모델 및 타입 정의
│   ├── engine/             # 백테스팅 엔진 (BacktestEngine, Portfolio)
│   ├── strategies/         # 전략 컴포넌트 (Allocation, Conditions, Pricing)
│   ├── validation/         # 전략 검증 (WFA, CPCV)
│   ├── data/               # 데이터 수집 (Fetcher)
│   └── reporting/          # 성과 지표 계산 (CAGR, MDD 등)
├── examples/               # 실행 예제 스크립트
│   ├── fetch_data.py       # 데이터 수집 예제
│   ├── run_backtest.py     # 전략 조립 및 백테스팅 실행 예제
│   └── run_wfa_validation.py # WFA 검증 실행 예제
└── tests/                  # 단위 테스트 (Pytest)
```

## 🚀 설치 방법 (Installation)

이 프로젝트는 Python 3.14 이상을 권장합니다.

### `uv` 사용 (권장)

```bash
# uv 설치 (없을 경우)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# 가상환경 생성 및 활성화
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
uv pip install -e ".[dev]"
```

### `pip` 사용

```bash
pip install -r requirements.txt
# 또는
pip install -e .
```

## 💡 사용 방법 (Quick Start)

`examples/` 디렉토리에 포함된 스크립트를 통해 주요 기능을 실행해볼 수 있습니다.

### 1. 데이터 수집 (Data Collection)

Upbit에서 OHLCV 데이터를 수집합니다. 스마트 수집기가 이미 다운로드된 데이터 이후부터 자동으로 이어서 수집합니다.

```bash
python examples/fetch_data.py
```
- 기본 설정: BTC, ETH, XRP, TRX (60분봉, 4시간봉, 일봉, 주봉, 월봉)
- 데이터 저장 경로: `bt/data/{interval}/{symbol}.parquet`

### 2. 백테스팅 실행 (Run Backtest)

변동성 돌파 전략(VBO)과 자산 배분 로직을 조립하여 백테스팅을 수행합니다.

```bash
python examples/run_backtest.py
```
- 실행 결과로 CAGR, MDD, 승률, 샤프 비율 등의 성과 지표와 샘플 매매 로그가 출력됩니다.
- `examples/run_backtest.py` 파일을 수정하여 파라미터나 전략 구성을 변경할 수 있습니다.

### 3. 전략 검증 (Validation)

Walk Forward Analysis (전진 분석)를 통해 전략의 견고성을 검증합니다.

```bash
python examples/run_wfa_validation.py
```
- 기간을 롤링 윈도우로 나누어 학습(Train) 및 테스트(Test)를 반복 수행합니다.
- 과적합 여부와 시장 국면별 성과 안정성을 확인할 수 있습니다.

## 📊 전략 상세 (Strategy Details)

### VBO (Volatility Breakout) 전략 예시
이 프레임워크에서 구현된 VBO 전략은 다음과 같은 구성 요소를 가집니다:

- **매수 조건 (Buy Conditions)**:
    1.  현재 포지션이 없을 것 (`no_open_position`)
    2.  가격이 변동성 돌파 라인을 상향 돌파 (`vbo_breakout_triggered`)
    3.  단기/장기 이동평균선 위에 위치 (추세 추종)
    4.  노이즈 비율이 감소 추세일 것
- **매도 조건 (Sell Conditions)**:
    1.  종가가 단기 이동평균선 아래로 하락 (`stop_trend`)
- **자산 배분 (Allocation)**:
    - **Cash Partition**: 전체 자산을 투자 대상 종목 수(N)로 나누어, 신호가 뜬 종목에 1/N씩 배분합니다.

## 📈 검증 방법론 (Validation Methods)

### 1. Walk Forward Analysis (WFA)
- 데이터를 연속된 구간으로 나누어 최적화와 테스트를 반복합니다.
- 미래의 데이터를 미리 보지 않는(Look-ahead bias 방지) 현실적인 성과를 추정합니다.

### 2. CPCV (Combinatorial Purged Cross-Validation)
- 시계열 데이터의 특성을 고려하여 훈련/테스트 세트 사이를 Purging(제거) 및 Embargo(유예) 처리합니다.
- 데이터 누수(Leakage)를 방지하며 교차 검증을 수행합니다.

## 🛠 지원하는 데이터 주기 (Supported Intervals)

`DataFetcher`는 `pyupbit` 라이브러리를 기반으로 하며 다음 주기를 지원합니다:
- `minute1`, `minute3`, `minute5`, `minute10`, `minute15`, `minute30`, `minute60`, `minute240`
- `day` (일봉)
- `week` (주봉)
- `month` (월봉)

## � 연구 노트 (Research Notes)

### 실험: 노이즈 필터 제거 효과

**가설**: VBO 전략의 노이즈 필터가 오히려 진입 기회를 제한하여 수익률을 저해할 수 있다.

**실험 과정**:
1. Jupyter Notebook에서 벡터 기반 빠른 백테스트로 가설 검증
2. 이벤트 기반 엔진으로 정밀 검증
3. WFA/CPCV로 과적합 여부 확인

**결과**:
| 조건 | CAGR | MDD | Sortino |
|------|------|-----|---------|
| 노이즈 필터 O | ~70% | -25% | ~2.0 |
| 노이즈 필터 X | **121%** | -24% | **3.28** |

**결론**: 노이즈 필터 제거 시 CAGR +73% 개선, 리스크(MDD) 유지. 단, 시장 국면에 따른 성과 편차가 크므로 실전 적용 시 주의 필요.

### 추가 개선 방향
- [ ] 시장 국면 탐지 기반 동적 포지션 사이징
- [ ] 파라미터 최적화 (Grid Search / Optuna)
- [ ] 다중 자산 상관관계 기반 리밸런싱
- [ ] 실시간 트레이딩 봇과 백테스트 전략 통합

## �📝 라이선스 (License)

MIT License