# BT Framework Documentation

## Overview

The BT Framework is a modern, event-driven backtesting engine for cryptocurrency trading strategies. Built with performance, extensibility, and production-readiness in mind.

## Key Features

- **High Performance**: Optimized with numpy arrays and caching
- **Plugin Architecture**: Extensible strategy and data provider system
- **Production Ready**: CI/CD, monitoring, containerization
- **Comprehensive Testing**: 200+ unit tests, integration tests, performance benchmarks
- **Enterprise Monitoring**: Structured logging, metrics collection, alerting

## Quick Start

```python
from bt import BacktestFramework

# Create framework instance
framework = BacktestFramework()

# Run a backtest
result = framework.run_backtest(
    strategy="volatility_breakout",
    symbols=["BTC", "ETH"],
    data=market_data
)

# View results
print(f"Total Return: {result['performance']['total_return']}%")
print(f"Sharpe Ratio: {result['performance']['sharpe']}")
```

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
tutorial
```

```{toctree}
:maxdepth: 2
:caption: User Guide

strategies
data_providers
reporting
configuration
monitoring
```

```{toctree}
:maxdepth: 2
:caption: Developer Guide

architecture
api_reference
plugins
contributing
```

```{toctree}
:maxdepth: 2
:caption: Advanced Topics

performance_optimization
security
deployment
troubleshooting
```

## Indices and Tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
