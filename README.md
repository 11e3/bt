# BT Framework

**Enterprise-grade cryptocurrency backtesting platform** with SOLID architecture, comprehensive security, and production-ready infrastructure.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SOLID](https://img.shields.io/badge/Architecture-SOLID-brightgreen.svg)](./SOLID_REFACTORING.md)

## 🎯 Version 2.0: SOLID Architecture

**Clean, maintainable design** following SOLID principles.

**Quick Migration (1 line):**
```python
from bt.framework.facade import BacktestFacade as BacktestFramework
```

**New Features:**
- 4 order types (Market, Limit, StopLoss, StopLimit)
- Component-based architecture (15 focused classes)
- Dependency injection for easy testing

📘 [Full Guide](./SOLID_REFACTORING.md) | 🔄 [Migration](./MIGRATION_GUIDE.md) | ⚡ [Quick Ref](./SOLID_SUMMARY.md)

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone <repository-url>
cd bt
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev
```

### Run Backtest

```bash
source .venv/bin/activate
python examples/quickstart.py
```

### Quality Checks

```bash
./scripts/check.sh        # All checks
uv run pytest             # Tests only
uv run ruff check src/    # Linting
```

## 📁 Project Structure

```
bt-framework/
├── src/bt/
│   ├── domain/          # Business models, types, orders
│   ├── engine/          # Backtest execution, portfolio
│   ├── framework/       # High-level API, facade
│   ├── strategies/      # Trading strategies
│   ├── reporting/       # Performance metrics
│   ├── data/           # Data providers
│   ├── security/       # Input validation
│   └── monitoring/     # Metrics, profiling
├── tests/              # Test suite
├── docs/               # Documentation
└── examples/           # Usage examples
```

## 💡 Usage Examples

### Basic Backtest

```python
from bt.framework.facade import BacktestFacade as BacktestFramework
import pandas as pd

# Load data
data = pd.read_csv('data/BTC.csv', parse_dates=['datetime'])

# Run backtest
framework = BacktestFramework()
result = framework.run_backtest(
    strategy='volatility_breakout',
    symbols=['BTC'],
    data={'BTC': data}
)

print(f"Return: {result['performance']['total_return']:.2f}%")
print(f"Sharpe: {result['performance']['sharpe']:.2f}")
```

### Component Access (SOLID)

```python
from bt.engine.portfolio_refactored import PortfolioRefactored
from decimal import Decimal

portfolio = PortfolioRefactored(
    initial_cash=Decimal("1000000"),
    fee=Decimal("0.0005"),
    slippage=Decimal("0.001")
)

# Access individual components
win_rate = portfolio.trade_recorder.get_win_rate()
max_dd = portfolio.equity_tracker.get_max_drawdown()
```

### New Order Types

```python
from bt.domain.orders import LimitOrder, StopLossOrder, OrderSide
from datetime import datetime
from decimal import Decimal

# Limit order
limit = LimitOrder(
    "BTC", OrderSide.BUY,
    Decimal("0.1"), Decimal("48000"),
    datetime.now()
)

# Stop loss
stop = StopLossOrder(
    "BTC", OrderSide.SELL,
    Decimal("0.1"), Decimal("45000"),
    datetime.now()
)
```

## 🏗️ Architecture

### SOLID Principles Applied

- **Single Responsibility**: 15 focused classes (was 2 monoliths)
- **Open/Closed**: Extensible order types via inheritance
- **Liskov Substitution**: Perfect polymorphism in orders
- **Interface Segregation**: 12 small interfaces vs 2 large ones
- **Dependency Inversion**: Container-based DI

### Key Components

**Domain Layer:**
- Order abstraction (4 types)
- Business models (Position, Trade, etc.)
- Type aliases (Price, Quantity, etc.)

**Engine Layer:**
- PortfolioRefactored (state management)
- OrderExecutor (execution logic)
- TradeRecorder (trade history)
- EquityTracker (performance tracking)

**Framework Layer:**
- BacktestFacade (main entry point)
- BacktestRunner (execution)
- StrategyManager (strategy registry)
- DataLoader (data handling)
- ReportGenerator (reporting)

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src/bt --cov-report=html

# Specific test
uv run pytest tests/test_portfolio.py -v
```

## 📊 Available Strategies

- `volatility_breakout` - Range breakout with volatility filtering
- `momentum` - Trend following
- `mean_reversion` - Mean reversion
- Custom strategies via plugin system

## 🔒 Security

- Input validation with Pydantic
- Secure configuration management
- Static analysis (Bandit, Safety)
- No hardcoded credentials
- Comprehensive error handling

## 📈 Performance

- Decimal precision for financial calculations
- NumPy optimization for equity curves
- Efficient data structures
- <3% overhead from SOLID refactoring

## 🛠️ Development

### Setup

```bash
# Install with dev dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

### Code Quality

```bash
# Format code
uv run ruff format src/

# Lint
uv run ruff check src/ --fix

# Type check
uv run mypy src/

# Security scan
uv run bandit -r src/
```

### CI/CD

GitHub Actions runs automatically on push:
- Linting (Ruff)
- Type checking (MyPy)
- Security scanning (Bandit, Safety)
- Tests (pytest)
- Coverage reporting

## 📚 Documentation

- [SOLID Refactoring Guide](./SOLID_REFACTORING.md) - Complete architecture guide
- [Migration Guide](./MIGRATION_GUIDE.md) - Step-by-step migration
- [Quick Summary](./SOLID_SUMMARY.md) - Quick reference
- [Principles Applied](./docs/SOLID_PRINCIPLES_APPLIED.md) - Detailed principles
- [Changelog](./CHANGELOG_SOLID.md) - Version history
- [API Reference](./docs/api/) - API documentation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

Ensure all quality checks pass: `./scripts/check.sh`

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**For educational and research purposes only.**

- Past performance does not guarantee future results
- Cryptocurrency trading involves substantial risk
- Never deploy without comprehensive testing
- Consult financial professionals before investing

## 🙏 Acknowledgments

- Robert C. Martin (Uncle Bob) for SOLID principles
- Python community for excellent tools
- All contributors and testers

---

**Version**: 2.0.0-SOLID
**Status**: Production Ready
**Last Updated**: 2026-01-16
