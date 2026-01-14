# Quick Start Tutorial

This tutorial will guide you through your first backtest using the BT Framework. We'll create a simple volatility breakout strategy and run it on historical data.

## Prerequisites

Make sure you have the BT Framework installed:

```bash
pip install bt-framework
```

Or for development:

```bash
git clone <repository-url>
cd bt-framework
pip install -e .[dev]
```

## Step 1: Import and Initialize

First, let's import the framework and create an instance:

```python
from bt import BacktestFramework

# Create a framework instance
framework = BacktestFramework()

print("BT Framework initialized!")
```

## Step 2: Load Market Data

The framework can load data from various sources. For this tutorial, we'll use synthetic data:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample BTC data
def generate_sample_data():
    dates = pd.date_range("2020-01-01", periods=365, freq="D")

    # Generate realistic price movements
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.03, len(dates))
    prices = 50000 * np.exp(np.cumsum(returns))

    # Create OHLCV data
    highs = prices * (1 + np.random.uniform(0, 0.02, len(dates)))
    lows = prices * (1 - np.random.uniform(0, 0.02, len(dates)))
    opens = prices * (1 + np.random.normal(0, 0.005, len(dates)))
    closes = prices

    # Ensure OHLC relationships
    opens = np.clip(opens, lows, highs)
    closes = np.clip(closes, lows, highs)

    # Add volume
    volumes = np.random.lognormal(15, 1, len(dates)).astype(int)

    return pd.DataFrame({
        "datetime": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })

# Load data
btc_data = generate_sample_data()
market_data = {"BTC": btc_data}

print(f"Loaded {len(btc_data)} days of BTC data")
print(f"Price range: ${btc_data['close'].min():.2f} - ${btc_data['close'].max():.2f}")
```

## Step 3: Choose a Strategy

The framework comes with built-in strategies. Let's use the volatility breakout strategy:

```python
# List available strategies
strategies = framework.list_available_strategies()
print("Available strategies:", strategies)

# Get info about the volatility breakout strategy
strategy_info = framework.get_strategy_info("volatility_breakout")
print("Strategy info:", strategy_info)
```

## Step 4: Configure the Strategy

Let's customize the strategy parameters:

```python
# Configure the strategy
strategy_config = {
    "lookback": 5,          # Lookback period for volatility calculation
    "multiplier": 2,        # Multiplier for long-term MA
    "k_factor": 0.5,        # Breakout threshold factor
    "top_n": 1,            # Number of symbols to allocate to (only BTC here)
    "mom_lookback": 20     # Momentum lookback period
}

# Validate the configuration
errors = framework.validate_strategy_config("volatility_breakout", strategy_config)
if errors:
    print("Configuration errors:", errors)
else:
    print("Configuration is valid!")
```

## Step 5: Run the Backtest

Now let's run the backtest:

```python
# Run the backtest
result = framework.run_backtest(
    strategy="volatility_breakout",
    symbols=["BTC"],
    data=market_data,
    config={
        "initial_cash": 100000,  # $100,000 starting capital
        "fee_rate": 0.0005,      # 0.05% trading fee
        "slippage_rate": 0.0005, # 0.05% slippage
        **strategy_config
    }
)

print("Backtest completed!")
```

## Step 6: Analyze Results

Let's examine the results:

```python
# Basic performance metrics
performance = result["performance"]
print(f"Total Return: {performance['total_return']:.2f}%")
print(f"Sharpe Ratio: {performance['sharpe']:.2f}")
print(f"Max Drawdown: {performance['mdd']:.2f}%")
print(f"Win Rate: {performance['win_rate']:.2f}%")
print(f"Total Trades: {performance['num_trades']}")

# Equity curve
equity = result["equity_curve"]
print(f"Final Portfolio Value: ${equity['values'][-1]:,.2f}")
print(f"Peak Value: ${max(equity['values']):,.2f}")

# Trade details
trades = result["trades"]
profitable_trades = [t for t in trades if t["pnl"] > 0]
print(f"Profitable Trades: {len(profitable_trades)}/{len(trades)}")

if trades:
    print("\\nSample Trade:")
    trade = trades[0]
    print(f"Symbol: {trade['symbol']}")
    print(f"Entry: ${trade['entry_price']:.2f} on {trade['entry_date'].date()}")
    print(f"Exit: ${trade['exit_price']:.2f} on {trade['exit_date'].date()}")
    print(f"P&L: ${trade['pnl']:.2f} ({trade['return_pct']:.2f}%)")
```

## Step 7: Visualize Results

Generate charts and reports:

```python
# Generate performance report
framework.create_performance_report(result)

# The report will be saved as HTML/PDF files
print("Performance report generated!")
```

## Step 8: Compare Strategies

Let's compare different strategies:

```python
strategies_to_test = ["volatility_breakout", "momentum", "buy_and_hold"]
results = {}

for strategy_name in strategies_to_test:
    print(f"Testing {strategy_name}...")

    result = framework.run_backtest(
        strategy=strategy_name,
        symbols=["BTC"],
        data=market_data,
        config={"initial_cash": 100000}
    )

    results[strategy_name] = result
    perf = result["performance"]
    print(".2f")

# Find the best performing strategy
best_strategy = max(results.keys(),
                   key=lambda s: results[s]["performance"]["total_return"])
print(f"\\nBest performing strategy: {best_strategy}")
```

## Complete Example

Here's the complete script you can run:

```python
#!/usr/bin/env python3
"""Complete BT Framework Tutorial Example"""

from bt import BacktestFramework
import pandas as pd
import numpy as np
from datetime import datetime

def generate_sample_data():
    """Generate sample BTC price data"""
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    np.random.seed(42)

    # Generate price movements
    returns = np.random.normal(0.001, 0.03, len(dates))
    prices = 50000 * np.exp(np.cumsum(returns))

    # Create OHLCV data
    highs = prices * (1 + np.random.uniform(0, 0.02, len(dates)))
    lows = prices * (1 - np.random.uniform(0, 0.02, len(dates)))
    opens = prices * (1 + np.random.normal(0, 0.005, len(dates)))
    closes = prices

    # Ensure OHLC relationships
    opens = np.clip(opens, lows, highs)
    closes = np.clip(closes, lows, highs)
    volumes = np.random.lognormal(15, 1, len(dates)).astype(int)

    return pd.DataFrame({
        "datetime": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes
    })

def main():
    """Run the complete tutorial"""
    print("🚀 BT Framework Tutorial")
    print("=" * 50)

    # Initialize framework
    framework = BacktestFramework()
    print("✅ Framework initialized")

    # Load data
    btc_data = generate_sample_data()
    market_data = {"BTC": btc_data}
    print(f"✅ Loaded {len(btc_data)} days of data")

    # Run backtest
    result = framework.run_backtest(
        strategy="volatility_breakout",
        symbols=["BTC"],
        data=market_data,
        config={
            "initial_cash": 100000,
            "fee_rate": 0.0005,
            "slippage_rate": 0.0005,
            "lookback": 5,
            "multiplier": 2,
            "k_factor": 0.5,
            "top_n": 1,
            "mom_lookback": 20
        }
    )

    # Display results
    perf = result["performance"]
    print("
📊 Results:"    print(".2f"    print(".2f"    print(".2f"    print(".2f"    print(f"📈 Total Trades: {perf['num_trades']}")

    final_value = result["equity_curve"]["values"][-1]
    print(",.2f"
    # Generate report
    framework.create_performance_report(result)
    print("\\n✅ Performance report generated!")

    print("\\n🎉 Tutorial completed successfully!")

if __name__ == "__main__":
    main()
```

## Next Steps

Now that you understand the basics, explore:

- **Custom Strategies**: Create your own trading strategies
- **Multi-Asset Portfolios**: Trade across multiple cryptocurrencies
- **Advanced Configuration**: Fine-tune strategy parameters
- **Performance Analysis**: Deep dive into backtest results
- **Plugin Development**: Extend the framework with custom components

Check out the full documentation for detailed guides on each topic!
