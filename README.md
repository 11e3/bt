# BT Framework

**Event-driven cryptocurrency backtesting engine for the Crypto Quant Ecosystem.**

Part of: [crypto-quant-system](https://github.com/11e3/crypto-quant-system) → **[bt](https://github.com/11e3/bt)** → [crypto-bot](https://github.com/11e3/crypto-bot) → [crypto-regime-classifier-ml](https://github.com/11e3/crypto-regime-classifier-ml)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SOLID](https://img.shields.io/badge/Architecture-SOLID-brightgreen.svg)](docs/SOLID_REFACTORING.md)

## Screenshots

| Backtest Results | Parameter Optimization | Monte Carlo Simulation |
|:----------------:|:----------------------:|:----------------------:|
| ![Backtest](img/backtest.png) | ![Optimization](img/optimization.png) | ![Monte Carlo](img/montecarlo.png) |

## Ecosystem Overview

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

## Installation

```bash
git clone https://github.com/11e3/bt.git
cd bt

# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev

# Or using pip
pip install -e ".[dev]"
```

## Quick Start

### Basic Backtest

```python
from bt.framework.facade import BacktestFacade
import pandas as pd

# Load data
data = pd.read_csv('data/BTC.csv', parse_dates=['datetime'])

# Run backtest
framework = BacktestFacade()
result = framework.run_backtest(
    strategy='volatility_breakout',
    symbols=['BTC'],
    data={'BTC': data}
)

print(f"Return: {result['performance']['total_return']:.2f}%")
print(f"Sharpe: {result['performance']['sharpe']:.2f}")
```

### Portfolio Backtest with CLI

```bash
# VBO Portfolio strategy with multiple assets
python scripts/backtest_vbo_portfolio.py --symbols BTC ETH SOL

# With date range
python scripts/backtest_vbo_portfolio.py --symbols BTC ETH --start 2023-01-01 --end 2024-12-31

# Using framework mode
python scripts/backtest_vbo_portfolio.py --symbols BTC ETH --use-framework
```

## Available Strategies

| Strategy | Class | Description |
|----------|-------|-------------|
| `volatility_breakout` | `VolatilityBreakoutStrategy` | VBO with MA trend filters |
| `vbo_portfolio` | `VBOPortfolioStrategy` | Multi-asset VBO with BTC market filter |
| `vbo_regime` | `VBORegimeStrategy` | VBO with ML regime classification |
| `momentum` | `MomentumStrategy` | Pure momentum with equal-weight |
| `buy_and_hold` | `BuyAndHoldStrategy` | Simple buy and hold |

### Strategy Configuration

```python
from bt.strategies.implementations import StrategyFactory

# VBO Portfolio with custom parameters
strategy = StrategyFactory.create(
    'vbo_portfolio',
    ma_short=5,        # Short MA for individual coins
    btc_ma=20,         # BTC MA for market filter
    noise_ratio=0.5,   # Volatility breakout multiplier
)

# VBO with ML Regime Model
strategy = StrategyFactory.create(
    'vbo_regime',
    ma_short=5,
    noise_ratio=0.5,
    model_path='models/regime_classifier.joblib',
)
```

## Architecture

### Project Structure

```
src/bt/
├── framework/       # High-level API (BacktestFacade)
├── strategies/      # Trading strategies
│   ├── components/  # Reusable strategy building blocks
│   └── implementations/  # Strategy implementations
├── core/            # Core services (DI container, registry)
├── engine/          # Backtest execution engine
├── domain/          # Business models, order types
├── data/            # Data providers and loaders
├── interfaces/      # Protocols and abstract interfaces
├── reporting/       # Performance metrics and charts
├── config/          # Configuration management
├── utils/           # Utilities (validation, caching)
├── security/        # Input validation
└── monitoring/      # Metrics and profiling
```

### Key Components

| Component | Description |
|-----------|-------------|
| `BacktestFacade` | Main entry point, coordinates all operations |
| `BacktestOrchestrator` | Executes backtest loop |
| `StrategyFactory` | Creates strategy instances |
| `Container` | Dependency injection container |
| `ReportGenerator` | Performance metrics and visualization |

### SOLID Principles Applied

- **Single Responsibility**: Each class has one job (e.g., `DataLoader` only loads data)
- **Open/Closed**: New strategies via plugin system without modifying core
- **Liskov Substitution**: All strategies implement `IStrategy` protocol
- **Interface Segregation**: Small, focused protocols (`ICondition`, `IAllocation`, `IPricing`)
- **Dependency Inversion**: Core depends on abstractions, not implementations

## Development

### Code Quality

```bash
# Run all checks
uv run ruff check src/    # Linting
uv run ruff format src/   # Formatting
uv run mypy src/bt/       # Type checking
uv run pytest tests/ -v   # Tests

# With coverage
uv run pytest --cov=src/bt --cov-report=html
```

### Creating Custom Strategies

```python
from bt.strategies.implementations import BaseStrategy
from bt.strategies.components import create_condition, create_allocation

class MyStrategy(BaseStrategy):
    def get_buy_conditions(self):
        return {
            "no_position": create_condition("no_open_position"),
            "trend": create_condition("price_above_sma", lookback=20),
        }

    def get_sell_conditions(self):
        return {
            "exit": create_condition("price_above_sma", lookback=10),
        }

    def get_allocation_func(self):
        return create_allocation("equal_weight_momentum", mom_lookback=20)
```

## Configuration

### Environment Variables

```bash
BT_LOG_LEVEL=INFO
BT_DATA_DIR=data/
BT_REPORT_DIR=reports/
```

### Config File

```python
from bt.framework.facade import BacktestFacade

config = {
    'initial_capital': 1_000_000,
    'fee': 0.0005,
    'slippage': 0.0005,
}

framework = BacktestFacade(config=config)
```

## Documentation

- [SOLID Refactoring Guide](docs/SOLID_REFACTORING.md)
- [Migration Guide](docs/MIGRATION_GUIDE.md)
- [Changelog](docs/CHANGELOG_SOLID.md)

## Dependencies

Core:
- pandas, numpy - Data manipulation
- pydantic - Configuration validation
- matplotlib, plotly - Visualization

ML (optional):
- scikit-learn - ML utilities
- xgboost - Gradient boosting
- joblib - Model serialization

## License

MIT License

## Disclaimer

**For educational and research purposes only.** Past performance does not guarantee future results. Cryptocurrency trading involves substantial risk of loss.

---

**Version**: 2.1.0 | **Python**: 3.10+
