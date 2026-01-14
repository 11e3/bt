# BT Framework - Enterprise Cryptocurrency Backtesting Platform

**A production-ready, enterprise-grade backtesting platform** for cryptocurrency trading strategies. Built with modern Python architecture, comprehensive security, performance monitoring, and scalable infrastructure.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://img.shields.io/badge/CI/CD-GitHub%20Actions-green.svg)](https://github.com/features/actions)
[![Security](https://img.shields.io/badge/Security-Bandit%20%26%20Safety-red.svg)](https://github.com/PyCQA/bandit)
[![Code Quality](https://img.shields.io/badge/Code%20Quality-Ruff%20%26%20MyPy-blue.svg)](https://github.com/charliermarsh/ruff)

## ✨ Key Highlights

- 🚀 **Production Ready**: Enterprise infrastructure with security, monitoring, and CI/CD
- 🏗️ **Modern Architecture**: Clean separation of concerns with dependency injection
- 🔒 **Security First**: Input validation, secure configuration, and vulnerability scanning
- 📊 **Performance Monitoring**: Real-time metrics, profiling, and alerting
- 🌐 **REST API**: Remote backtest execution and management
- 🗄️ **Data Abstraction**: Flexible storage backends with intelligent caching
- 🧪 **Quality Assurance**: Comprehensive testing, linting, and type checking
- 📈 **Advanced Analytics**: Walk-forward analysis, cross-validation, and reporting

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd bt

# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv
export PATH="$HOME/.local/bin:$PATH"
uv sync --dev
```

### Run a Backtest

```bash
# Activate virtual environment
source .venv/bin/activate

# Run a comprehensive backtest
python -c "
from bt import BacktestFramework
import pandas as pd
import numpy as np

# Create sample data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
prices = 50000 * np.exp(np.cumsum(np.random.normal(0.001, 0.03, len(dates))))
data = pd.DataFrame({
    'datetime': dates,
    'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
    'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
    'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
    'close': prices,
    'volume': np.random.lognormal(15, 1, len(dates)).astype(int)
})

# Run backtest
framework = BacktestFramework()
result = framework.run_backtest(
    strategy='volatility_breakout',
    symbols=['BTC'],
    data={'BTC': data}
)

print(f'Total Return: {result[\"performance\"][\"total_return\"]:.2f}%')
print(f'Sharpe Ratio: {result[\"performance\"][\"sharpe\"]:.2f}')
"
```

### Quality Assurance

```bash
# Run comprehensive quality checks
./scripts/check.sh

# Or run individual tools
uv run ruff format .     # Format code
uv run ruff check .      # Lint code
uv run mypy src/bt       # Type check
uv run pytest            # Run tests
```

## 📊 Sample Results

### VBO Strategy Performance (2017-2026)

| Metric | Value |
|--------|-------|
| **Total Return** | 47,503.10% |
| **CAGR** | 116.12% |
| **MDD** | -29.08% |
| **Sortino Ratio** | 3.01 |
| **Win Rate** | 34.14% |
| **Profit Factor** | 1.70 |
| **Number of Trades** | 1,078 |

### Yearly Performance

| Year | Return |
|------|--------|
| 2018 | 403.80% |
| 2019 | 58.15% |
| 2020 | 218.92% |
| 2021 | 334.71% |
| 2022 | -18.84% |
| 2023 | 35.05% |
| 2024 | 153.99% |
| 2025 | 50.43% |
| 2026 | -2.90% |

## 🏗️ Enterprise Architecture

### Core Components

```
bt-framework/
├── 📁 src/bt/                           # Main package
│   ├── 🏗️ core/                        # Enterprise infrastructure
│   │   ├── container.py                # Dependency injection
│   │   ├── orchestrator.py             # Service orchestration
│   │   ├── registry.py                 # Strategy/component registry
│   │   └── base.py                     # Abstract base classes
│   ├── 🔒 security/                    # Security & validation
│   │   ├── __init__.py                 # Input validation & scanning
│   ├── 📊 monitoring/                  # Performance monitoring
│   ├── 🗄️ data/storage/                # Data abstraction layer
│   ├── 🌐 api/                         # REST API server
│   ├── 🧪 profiling/                   # Performance profiling
│   ├── ⚙️ config/                      # Configuration management
│   ├── 🏛️ domain/                      # Domain models & types
│   ├── 📈 reporting/                   # Performance analytics
│   ├── 🎯 strategies/                  # Trading strategies
│   ├── 🔍 validation/                  # Strategy validation
│   ├── 🛠️ utils/                      # Utilities & helpers
│   └── 🔌 plugins/                     # Plugin system
├── 🧪 tests/                          # Comprehensive test suite
├── 📜 scripts/                        # Quality assurance scripts
├── 📚 docs/                           # Sphinx documentation
├── 🔧 .github/                        # CI/CD workflows
└── 📋 pyproject.toml                  # Modern Python packaging
```

### 🏢 Infrastructure Layers

#### **🔒 Security Layer**
- Input validation and sanitization
- Secure configuration management
- Security scanning integration (Bandit, Safety)
- SQL injection and XSS protection

#### **📊 Monitoring Layer**
- Real-time performance metrics
- Structured logging with correlation IDs
- Alerting system (Slack/Email integration)
- Memory leak detection

#### **🗄️ Data Abstraction Layer**
- Multiple storage backends (Memory, JSON, Parquet, HDF5, SQLite)
- Intelligent caching with TTL and eviction policies
- Data compression and serialization
- Backup and migration utilities

#### **🌐 API Layer**
- FastAPI-based REST endpoints
- Asynchronous job processing
- Health monitoring and metrics
- Result download and reporting

#### **🧪 Quality Assurance**
- Comprehensive test suite (200+ tests)
- Automated linting and formatting (Ruff)
- Type checking (MyPy)
- Performance profiling and analysis

## 🎯 Enterprise Features

### 🚀 Core Capabilities
- **Event-Driven Engine**: High-performance bar-by-bar simulation
- **Composable Strategies**: Modular strategy components with plugin system
- **Type Safety**: Full Pydantic validation and comprehensive type hints
- **Decimal Precision**: Financial calculations with `Decimal` for accuracy
- **Advanced Validation**: Walk-forward analysis and combinatorial cross-validation

### 🔒 Security & Compliance
- **Input Validation**: Comprehensive data sanitization and validation
- **Secure Configuration**: Environment-aware secrets management
- **Security Scanning**: Automated vulnerability detection (Bandit, Safety, pip-audit)
- **Access Control**: Safe file operations and path validation

### 📊 Production Monitoring
- **Performance Profiling**: CPU/memory profiling with detailed reports
- **Real-time Metrics**: Prometheus-compatible monitoring endpoints
- **Structured Logging**: JSON logging with correlation IDs
- **Alerting System**: Configurable alerts for performance issues

### 🗄️ Data Management
- **Multiple Backends**: Memory, file-based (JSON/Parquet/HDF5), and database storage
- **Intelligent Caching**: LRU/LFU/FIFO eviction with TTL support
- **Data Compression**: Efficient storage and retrieval
- **Backup & Migration**: Automated data management utilities

### 🌐 Remote Execution
- **REST API**: FastAPI-based endpoints for remote backtest management
- **Asynchronous Processing**: Background job execution with status tracking
- **Result Management**: Download reports and access historical results
- **Health Monitoring**: System health checks and metrics endpoints

### 🧪 Quality Assurance
- **Comprehensive Testing**: 200+ unit tests with integration and performance tests
- **Code Quality**: Automated linting, formatting, and type checking
- **CI/CD Pipeline**: GitHub Actions with multi-platform testing
- **Documentation**: Sphinx-generated API docs and tutorials

## 💻 Usage Examples

### Modern Backtest Framework

```python
from bt import BacktestFramework
import pandas as pd

# Initialize framework (handles all infrastructure automatically)
framework = BacktestFramework()

# Prepare market data (framework validates and caches automatically)
market_data = {
    "BTC": pd.read_parquet("data/day/BTC.parquet"),
    "ETH": pd.read_parquet("data/day/ETH.parquet")
}

# Run backtest with built-in strategy
result = framework.run_backtest(
    strategy="volatility_breakout",  # Built-in strategy
    symbols=["BTC", "ETH"],
    data=market_data,
    config={
        "initial_cash": 100000,      # $100K starting capital
        "fee_rate": 0.0005,          # 0.05% trading fee
        "slippage_rate": 0.0005,     # 0.05% slippage
        "lookback": 5,               # Strategy parameters
        "multiplier": 2,
        "k_factor": 0.5
    }
)

# Results include comprehensive analysis
print(f"Total Return: {result['performance']['total_return']:.2f}%")
print(f"Sharpe Ratio: {result['performance']['sharpe']:.2f}")
print(f"Max Drawdown: {result['performance']['mdd']:.2f}%")
print(f"Win Rate: {result['performance']['win_rate']:.2f}%")

# Framework automatically generates reports
framework.create_performance_report(result)
```

### REST API Usage

```bash
# Start API server
uv run uvicorn bt.api:app --host 0.0.0.0 --port 8000

# Run backtest remotely
curl -X POST http://localhost:8000/backtests \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "volatility_breakout",
    "symbols": ["BTC"],
    "config": {
      "initial_cash": 100000,
      "fee_rate": 0.0005
    }
  }'

# Check status
curl http://localhost:8000/backtests/{backtest_id}

# Get system health
curl http://localhost:8000/health
```

### Custom Strategy Components

```python
from typing import Dict, Any
from bt.domain.types import Price, Quantity

# Custom buy condition
def custom_buy_condition(engine: BacktestEngine, symbol: str) -> bool:
    """Buy when RSI < 30 and price > 20-day SMA"""
    bars = engine.get_bars(symbol, 20)
    if bars is None or len(bars) < 20:
        return False

    current_price = bars['close'].iloc[-1]
    sma_20 = bars['close'].rolling(20).mean().iloc[-1]

    # Calculate RSI (simplified)
    delta = bars['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1] < 30 and current_price > sma_20

# Custom allocation function
def fixed_allocation(engine: BacktestEngine, symbol: str, price: Price) -> Quantity:
    """Allocate fixed 10% of portfolio per trade"""
    portfolio_value = engine.portfolio.value
    target_amount = portfolio_value * Decimal("0.1")
    return Quantity(target_amount / Decimal(price))
```

## 📈 Strategy Validation

### Walk Forward Analysis (WFA)

```python
from bt.validation.wfa import WalkForwardAnalysis

wfa = WalkForwardAnalysis(
    train_periods=365,   # 1 year training
    test_periods=90,     # 3 months testing
    step_periods=90,     # 3 months step
)

results = wfa.run(data, backtest_func)
print(f"Positive windows: {results.positive_windows}/{results.total_windows}")
print(f"Average CAGR: {results.avg_cagr:.2%}")
```

### Combinatorial Purged Cross-Validation (CPCV)

```python
from bt.validation.cpcv import CombinatorialPurgedCV

cpcv = CombinatorialPurgedCV(
    num_splits=5,
    embargo_pct=0.01,
    purge_pct=0.05,
)

results = cpcv.run(data, backtest_func)
print(f"Mean return: {results.mean_return:.2%}")
print(f"Std deviation: {results.std_return:.2%}")
```

## 📊 Performance Metrics

The framework calculates comprehensive metrics:

- **Return Metrics**: Total Return, CAGR, Yearly Returns
- **Risk Metrics**: Maximum Drawdown, Sortino Ratio, Sharpe Ratio
- **Trade Statistics**: Win Rate, Profit Factor, Average Win/Loss
- **Portfolio Analysis**: Final Equity, Trade Count, Position Analysis

### Visualization

```python
from bt.reporting import plot_equity_curve, plot_yearly_returns, save_all_charts

# Generate visualizations
plot_equity_curve(metrics.equity_curve, metrics.dates)
plot_yearly_returns(metrics.yearly_returns)

# Save all charts
save_all_charts(metrics, output_dir="output/")
```

## 🎛️ Configuration

### Backtest Config

```python
from bt.domain.models import BacktestConfig

config = BacktestConfig(
    initial_cash=Amount(Decimal("10000000")),  # Starting capital
    fee=Fee(Decimal("0.0005")),               # Trading fee (0.05%)
    slippage=Percentage(Decimal("0.0005")),    # Slippage (0.05%)
    multiplier=2,                             # Long-term multiplier
    lookback=5,                               # Short-term lookback
    interval="days",                          # Data interval
)
```

### Logging

```python
from bt.utils.logging import setup_logging, get_logger

# Setup structured logging
setup_logging(level="INFO", format="json")

# Use logger
logger = get_logger(__name__)
logger.info("Backtest started", extra={"symbols": ["BTC", "ETH"]})
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/bt

# Run specific test
pytest tests/test_backtest.py -v
```

## 📁 Advanced Data Management

### Data Abstraction Layer

```python
from bt.data.storage import get_data_manager, StorageBackend
from bt.data.fetcher import DataFetcher

# Initialize data manager (automatic caching and backend management)
data_manager = get_data_manager()

# Store data in different backends
data_manager.store("btc_daily", btc_data, backend=StorageBackend.PARQUET)
data_manager.store("eth_daily", eth_data, backend=StorageBackend.HDF5)

# Retrieve with automatic caching
btc_data = data_manager.retrieve("btc_daily")  # Fast cached access
eth_data = data_manager.retrieve("eth_daily")  # Cached for future use

# Backup data across backends
data_manager.backup_data(
    target_backend=StorageBackend.SQLITE,
    source_backends=[StorageBackend.MEMORY, StorageBackend.PARQUET]
)
```

### Market Data Fetching

```python
from bt.data.fetcher import DataFetcher

# Initialize fetcher with caching
fetcher = DataFetcher()

# Fetch with automatic caching and validation
btc_data = fetcher.fetch_ohlcv(
    symbol="BTC",
    interval="day",
    start_date="2020-01-01",
    end_date="2024-01-01",
    use_cache=True  # Automatic caching
)

# Save to persistent storage
fetcher.save_data(btc_data, "day", "BTC")
```

### Data Format Standards

All data follows standardized OHLCV format with validation:
- `datetime`: timezone-aware timestamp
- `open/high/low/close`: validated price ranges
- `volume`: positive numeric values
- Automatic type conversion and sanitization

### Storage Backend Options

| Backend | Use Case | Features |
|---------|----------|----------|
| **Memory** | Development/Testing | Fast, volatile |
| **JSON** | Configuration/Export | Human-readable |
| **Parquet** | Analytics | Compressed, columnar |
| **HDF5** | Large datasets | Hierarchical, fast I/O |
| **SQLite** | Production | ACID compliant, concurrent |

## 🔧 Available Scripts & Tools

### **Quality Assurance Scripts**
| Script | Purpose |
|--------|---------|
| `scripts/check.sh` | **Comprehensive QA pipeline** (format, lint, test, security) |
| `scripts/code_quality.py` | Code quality analysis and performance profiling |
| `scripts/security_check.py` | Security scanning and vulnerability detection |

### **Development Scripts**
| Script | Purpose |
|--------|---------|
| `scripts/dev.sh` | Development workflow helper (setup, format, test, etc.) |

### **Legacy Scripts** (Maintained for compatibility)
| Script | Purpose |
|--------|---------|
| `scripts/fetch_data.py` | Download market data from exchanges |
| `scripts/run_backtest.py` | Legacy backtest execution |

### **API Server**
```bash
# Start REST API server
uv run python -m bt.api

# Or with uvicorn
uv run uvicorn bt.api:app --host 0.0.0.0 --port 8000
```

### **Infrastructure Commands**
```bash
# View system health
curl http://localhost:8000/health

# Run security scan
python scripts/security_check.py --path src/

# Generate code quality report
python scripts/code_quality.py --path src/ --output report.txt
```

## 🎯 VBO Strategy Details

### Strategy Logic

The VBO (Volatility Breakout) strategy combines:

1. **Volatility Breakout**: Price breaks above recent range
2. **Trend Following**: Moving average filters
3. **Momentum Allocation**: Equal distribution among active signals

### Buy Conditions
- No existing position
- Price > (Open + Range × K) where K=0.5
- Price > 5-day moving average
- Price > 20-day moving average

### Sell Conditions
- Close < 5-day moving average

### Performance Characteristics
- **Bull Markets**: Exceptional performance (2017: +403%, 2021: +334%)
- **Bear Markets**: Moderate drawdowns (2022: -18%)
- **Recovery**: Strong rebound capabilities

## 🛠 Development & Quality Assurance

### 🚀 Quick Start Development

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install all dependencies
uv sync --dev

# Run comprehensive quality checks
./scripts/check.sh

# Start development
uv run python -m bt.api  # Start API server
```

### 🧪 Quality Assurance Pipeline

#### **Comprehensive Quality Checks**
```bash
# Full QA pipeline (recommended for commits)
./scripts/check.sh

# Individual quality tools
uv run ruff format .          # Format code
uv run ruff check . --fix     # Lint and auto-fix
uv run mypy src/bt            # Type checking
python scripts/security_check.py  # Security scanning
python scripts/code_quality.py    # Code quality analysis
uv run pytest                  # Run test suite
```

#### **Performance Profiling**
```python
from bt.profiling import profile_function, get_performance_stats

@profile_function
def my_strategy():
    # Your strategy code
    pass

# Get performance report
stats = get_performance_stats()
print(stats)
```

#### **Security Validation**
```python
from bt.security import validate_input, scan_security

# Validate input data
clean_data = validate_input(user_input, "dataframe")

# Scan codebase for security issues
vulnerabilities = scan_security(Path("src/"))
```

### 🔧 Development Tools

#### **Pre-commit Hooks**
```bash
# Install git hooks
uv run pre-commit install

# Run on all files
uv run pre-commit run --all-files
```

#### **Testing with Coverage**
```bash
# Run tests with coverage
uv run pytest --cov=src/bt --cov-report=html

# Open coverage report
open htmlcov/index.html
```

#### **API Development**
```bash
# Start API server for development
uv run uvicorn bt.api:app --reload --host 0.0.0.0 --port 8000

# API documentation available at http://localhost:8000/docs
```

### 📊 CI/CD Pipeline

The framework includes comprehensive CI/CD:

- **Multi-platform testing**: Linux, macOS, Windows
- **Security scanning**: Bandit, Safety, pip-audit
- **Code quality**: Ruff linting, MyPy type checking
- **Performance profiling**: Automated performance analysis
- **Container builds**: Docker images for deployment

### 🏗️ Infrastructure Components

#### **Dependency Injection**
```python
from bt.core.container import get_default_container

container = get_default_container()
# Services automatically registered and resolved
```

#### **Monitoring & Alerting**
```python
from bt.monitoring import get_monitor

monitor = get_monitor()
monitor.record_metric("backtest_duration", 45.2)
```

#### **Plugin System**
```python
from bt.plugins import PluginManager

manager = PluginManager()
manager.load_plugin("my_custom_strategy")
```

## 🌐 REST API

The framework provides a comprehensive REST API for remote backtest management:

### **Endpoints**
- `POST /backtests` - Create and start backtest
- `GET /backtests` - List all backtests
- `GET /backtests/{id}` - Get backtest details
- `DELETE /backtests/{id}` - Cancel running backtest
- `GET /backtests/{id}/results/download` - Download results
- `GET /strategies` - List available strategies
- `GET /health` - System health check
- `GET /metrics` - Performance metrics

### **API Features**
- **Asynchronous processing** with background job management
- **Result caching** and automatic cleanup
- **Health monitoring** with uptime and active job tracking
- **Interactive documentation** at `/docs`

## 🔌 Plugin System

### **Creating Custom Strategies**

```python
from bt.strategies.base import BaseStrategy
from bt.core.registry import register_strategy

@register_strategy(
    name="my_custom_strategy",
    category="Custom",
    description="My custom trading strategy"
)
class CustomStrategy(BaseStrategy):
    def get_buy_conditions(self) -> dict[str, Callable]:
        return {"custom_signal": self._check_buy_signal}

    def get_sell_conditions(self) -> dict[str, Callable]:
        return {"exit_signal": self._check_sell_signal}

    def _check_buy_signal(self, engine, symbol: str) -> bool:
        # Custom buy logic
        return True

    def _check_sell_signal(self, engine, symbol: str) -> bool:
        # Custom sell logic
        return False
```

### **Plugin Development**

```python
from bt.plugins import BasePlugin

class MyPlugin(BasePlugin):
    def initialize(self, config: dict) -> None:
        self.config = config

    def get_strategies(self) -> list[BaseStrategy]:
        return [CustomStrategy()]

    def get_data_providers(self) -> list:
        return []  # Custom data providers
```

## 📊 Advanced Analytics

### **Performance Profiling**

```python
from bt.profiling import profile_function, profile_context

@profile_function
def complex_calculation():
    # Heavy computation
    return result

with profile_context("data_processing"):
    # Code to profile
    process_large_dataset()

# Generate performance report
from bt.profiling import get_performance_stats
stats = get_performance_stats()
stats.to_csv("performance_report.csv")
```

### **Memory Leak Detection**

```python
from bt.monitoring import get_monitor

monitor = get_monitor()
monitor.start_memory_tracking()

# Your code here

leaks = monitor.detect_memory_leaks()
for leak in leaks:
    print(f"Memory leak: {leak['file']}:{leak['line']} - {leak['size_mb']}MB")
```

### **Strategy Validation**

```python
from bt.validation import WalkForwardAnalysis, CombinatorialPurgedCV

# Walk-forward analysis
wfa = WalkForwardAnalysis(train_periods=365, test_periods=90)
wfa_results = wfa.run(data, backtest_function)

# Cross-validation
cpcv = CombinatorialPurgedCV(num_splits=5, embargo_pct=0.01)
cpcv_results = cpcv.run(data, backtest_function)
```

## 🏢 Enterprise Deployment

### **Docker Deployment**

```dockerfile
FROM python:3.11-slim

# Install uv
RUN pip install uv

# Copy project
COPY . /app
WORKDIR /app

# Install dependencies
RUN uv sync --no-dev

# Expose API port
EXPOSE 8000

# Run API server
CMD ["uv", "run", "python", "-m", "bt.api"]
```

### **Production Configuration**

```python
from bt import BacktestFramework

# Production configuration
framework = BacktestFramework(config={
    "security": {
        "max_data_size": 100_000_000,  # 100MB limit
        "allowed_file_types": [".parquet", ".csv"],
    },
    "monitoring": {
        "alert_email": "admin@company.com",
        "metrics_retention_days": 30,
    },
    "data": {
        "cache_enabled": True,
        "cache_ttl": 3600,
        "default_backend": "sqlite",
    }
})
```

### **Monitoring & Alerting**

```python
from bt.monitoring import AlertRule, AlertChannel

# Configure alerts
alert_rule = AlertRule(
    name="high_memory_usage",
    condition=lambda metrics: metrics.memory_mb > 1000,
    message="Memory usage exceeded 1GB",
    severity="high",
    cooldown_minutes=30
)

# Set up alert channels
email_channel = AlertChannel.email("admin@company.com")
slack_channel = AlertChannel.slack("#alerts", "webhook_url")
```

## 🏆 Infrastructure Achievements

### **Enterprise Production Features**
- ✅ **Security Layer**: Input validation, secure config, vulnerability scanning
- ✅ **Monitoring System**: Performance metrics, alerting, structured logging
- ✅ **Data Abstraction**: Multiple backends with intelligent caching
- ✅ **REST API**: Remote execution with asynchronous processing
- ✅ **Quality Assurance**: Comprehensive testing, linting, type checking
- ✅ **CI/CD Pipeline**: Automated testing and deployment
- ✅ **Containerization**: Docker support for production deployment
- ✅ **Plugin Architecture**: Extensible strategy and data provider system

### **Performance & Scalability**
- 🚀 **High-Performance Engine**: Event-driven backtesting with optimization
- 📊 **Intelligent Caching**: LRU/LFU/FIFO with TTL support
- 🔄 **Asynchronous Processing**: Background job execution
- 📈 **Monitoring**: Real-time performance tracking and alerting

### **Developer Experience**
- 🛠️ **Modern Tooling**: uv, ruff, mypy, pre-commit hooks
- 📚 **Documentation**: Sphinx-generated API docs and tutorials
- 🧪 **Testing**: 200+ tests with comprehensive coverage
- 🔒 **Security**: Automated security scanning and validation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Develop** your changes with tests
4. **Run** quality checks (`./scripts/check.sh`)
5. **Commit** your changes (`git commit -m 'Add amazing feature'`)
6. **Push** to the branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### **Development Guidelines**
- Follow the existing code style (enforced by ruff)
- Add comprehensive tests for new features
- Update documentation for API changes
- Ensure all quality checks pass
- Use type hints for all function signatures

## 📞 Support & Community

- 📖 **Documentation**: [Read the Docs](https://bt-framework.readthedocs.io/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/your-org/bt-framework/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/your-org/bt-framework/discussions)
- 📧 **Security Issues**: security@bt-framework.com

## ⚠️ Important Disclaimer

**This backtesting framework is for educational and research purposes only.**

- Past performance does not guarantee future results
- Always conduct thorough validation and out-of-sample testing
- Never deploy trading strategies without comprehensive risk management
- Cryptocurrency trading involves substantial risk of loss
- Consult with financial professionals before making investment decisions

---

**Built with ❤️ for the quantitative trading community**
