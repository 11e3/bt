# BT Framework

**Backtesting engine for the Crypto Quant Ecosystem.**

Part of: `crypto-quant-system` → **`bt`** → `crypto-bot` → `crypto-regime-classifier-ml`

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SOLID](https://img.shields.io/badge/Architecture-SOLID-brightgreen.svg)](./SOLID_REFACTORING.md)

## Ecosystem Role

```
┌─────────────────────────────────────────────────────────────────┐
│                    Crypto Quant Ecosystem                       │
├─────────────────────────────────────────────────────────────────┤
│  crypto-quant-system     │  Dashboard & data pipeline          │
│    ├── Data download     │  - Fetches OHLCV from exchanges     │
│    ├── Data processing   │  - Imports bt for backtesting       │
│    └── Bot log viewer    │  - Reads logs from GCS              │
├──────────────────────────┼──────────────────────────────────────┤
│  bt (this repo)          │  Backtesting engine                 │
│    ├── Strategy dev      │  - SOLID architecture               │
│    ├── Performance calc  │  - Used by crypto-quant-system      │
│    └── Order simulation  │  - Strategies exported to crypto-bot│
├──────────────────────────┼──────────────────────────────────────┤
│  crypto-bot              │  Live trading bot                   │
│    ├── Auto trading      │  - Runs on GCP e2-micro             │
│    ├── ML integration    │  - Loads models from GCS            │
│    └── Logging           │  - Uploads logs to GCS              │
├──────────────────────────┼──────────────────────────────────────┤
│  crypto-regime-ml        │  Market regime classifier           │
│    └── Model training    │  - Uploads .pkl to GCS              │
└──────────────────────────┴──────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
git clone <repository-url>
cd bt
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev
```

### Run Backtest

```python
from bt.framework.facade import BacktestFacade as BacktestFramework
import pandas as pd

data = pd.read_csv('data/BTC.csv', parse_dates=['datetime'])

framework = BacktestFramework()
result = framework.run_backtest(
    strategy='volatility_breakout',
    symbols=['BTC'],
    data={'BTC': data}
)

print(f"Return: {result['performance']['total_return']:.2f}%")
print(f"Sharpe: {result['performance']['sharpe']:.2f}")
```

### Integration with crypto-quant-system

```python
# In crypto-quant-system dashboard
from bt.framework.facade import BacktestFacade

def run_dashboard_backtest(strategy_name: str, data: dict):
    framework = BacktestFacade()
    return framework.run_backtest(
        strategy=strategy_name,
        symbols=list(data.keys()),
        data=data
    )
```

## Architecture

### SOLID Design (v2.0)

```
src/bt/
├── domain/          # Business models, order types
├── engine/          # Backtest execution, portfolio
├── framework/       # High-level API (BacktestFacade)
├── strategies/      # Trading strategies
├── reporting/       # Performance metrics
├── data/            # Data providers
├── security/        # Input validation
└── monitoring/      # Metrics, profiling
```

### Key Components

| Layer | Component | Description |
|-------|-----------|-------------|
| Framework | `BacktestFacade` | Main entry point for crypto-quant-system |
| Engine | `PortfolioRefactored` | State management with DI |
| Domain | `Order` (4 types) | Market, Limit, StopLoss, StopLimit |
| Reporting | `ReportGenerator` | Performance metrics for dashboard |

### Order Types

```python
from bt.domain.orders import LimitOrder, StopLossOrder, OrderSide
from decimal import Decimal
from datetime import datetime

limit = LimitOrder("BTC", OrderSide.BUY, Decimal("0.1"), Decimal("48000"), datetime.now())
stop = StopLossOrder("BTC", OrderSide.SELL, Decimal("0.1"), Decimal("45000"), datetime.now())
```

## Available Strategies

| Strategy | Description | Status |
|----------|-------------|--------|
| `volatility_breakout` | VBO with MA filters | ✅ Production |
| `momentum` | Trend following | ✅ Ready |
| `mean_reversion` | Mean reversion | ✅ Ready |
| Custom | Plugin system | ✅ Supported |

## Development

```bash
# Quality checks
./scripts/check.sh        # All checks
uv run pytest             # Tests
uv run ruff check src/    # Linting
uv run mypy src/          # Type check

# With coverage
uv run pytest --cov=src/bt --cov-report=html
```

## Documentation

- [SOLID Refactoring Guide](./SOLID_REFACTORING.md)
- [Migration Guide](./MIGRATION_GUIDE.md)
- [API Reference](./docs/api/)

## License

MIT License

## Disclaimer

**For educational and research purposes only.** Past performance does not guarantee future results. Cryptocurrency trading involves substantial risk.

---

**Version**: 2.0.0-SOLID | **Ecosystem**: Crypto Quant System
