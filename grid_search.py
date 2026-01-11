"""Grid search for optimal momentum allocator parameters."""

from decimal import Decimal
from itertools import product

import pandas as pd

from bt.config import settings
from bt.domain.models import BacktestConfig
from bt.domain.types import Amount, Fee, Percentage
from bt.engine.backtest import BacktestEngine
from bt.logging import get_logger, setup_logging
from bt.reporting.metrics import calculate_performance_metrics
from bt.strategies import allocation, conditions, pricing


def load_data(symbol: str, interval: str = "day") -> pd.DataFrame:
    """Load parquet data."""
    interval_dir = settings.data_dir / interval
    file_path = interval_dir / f"{symbol}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_parquet(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def run_backtest(
    symbols: list[str],
    top_n: int,
    mom_lookback: int,
) -> dict:
    """Run a single backtest with given parameters."""
    config = BacktestConfig(
        initial_cash=Amount(Decimal("10000000")),
        fee=Fee(Decimal("0.0005")),
        slippage=Percentage(Decimal("0.0005")),
        multiplier=2,
        lookback=5,
        interval="day",
    )
    
    engine = BacktestEngine(config)
    
    loaded_symbols = []
    for symbol in symbols:
        try:
            df = load_data(symbol, "day")
            engine.load_data(symbol, df)
            loaded_symbols.append(symbol)
        except Exception as e:
            return None

    if not loaded_symbols:
        return None

    strategy = {
        "buy_conditions": {
            "no_pos": conditions.no_open_position,
            "breakout": conditions.vbo_breakout_triggered,
            "trend_short": conditions.price_above_short_ma,
            "trend_long": conditions.price_above_long_ma,
        },
        "sell_conditions": {
            "has_pos": conditions.has_open_position,
            "stop_trend": conditions.close_below_short_ma,
        },
        "buy_price_func": pricing.get_vbo_buy_price,
        "sell_price_func": pricing.get_current_close,
        "allocation_func": allocation.create_momentum_allocator(
            top_n=top_n,
            mom_lookback=mom_lookback
        ),
    }

    try:
        engine.run(
            symbols=loaded_symbols,
            buy_conditions=strategy["buy_conditions"],
            sell_conditions=strategy["sell_conditions"],
            buy_price_func=strategy["buy_price_func"],
            sell_price_func=strategy["sell_price_func"],
            allocation_func=strategy["allocation_func"],
        )

        metrics = calculate_performance_metrics(
            equity_curve=engine.portfolio.equity_curve,
            dates=engine.portfolio.dates,
            trades=engine.portfolio.trades,
            _initial_cash=config.initial_cash,
        )

        return {
            "top_n": top_n,
            "mom_lookback": mom_lookback,
            "cagr": float(metrics.cagr),
            "mdd": float(metrics.mdd),
            "sortino": float(metrics.sortino_ratio),
            "win_rate": float(metrics.win_rate),
            "num_trades": metrics.num_trades,
            "profit_factor": float(metrics.profit_factor),
        }
    except Exception as e:
        print(f"Error with top_n={top_n}, mom_lookback={mom_lookback}: {e}")
        return None


def main() -> None:
    setup_logging(level="WARNING", log_format="text")
    
    symbols = ["BTC", "ETH", "XRP", "TRX", "ADA"]
    
    # Grid parameters
    top_n_values = [1, 2, 3, 4, 5]
    mom_lookback_values = [5, 10, 15, 20, 25]
    
    results = []
    total = len(top_n_values) * len(mom_lookback_values)
    count = 0
    
    print("\n" + "=" * 80)
    print("GRID SEARCH: top_n x mom_lookback")
    print("=" * 80)
    print(f"Total combinations: {total}\n")
    
    for top_n, mom_lookback in product(top_n_values, mom_lookback_values):
        count += 1
        print(f"[{count:2d}/{total}] Testing top_n={top_n}, mom_lookback={mom_lookback:2d}...", end=" ", flush=True)
        
        result = run_backtest(symbols, top_n, mom_lookback)
        
        if result:
            results.append(result)
            print(f"✓ Sortino: {result['sortino']:.2f}, CAGR: {result['cagr']:7.2f}%, MDD: {result['mdd']:7.2f}%")
        else:
            print("✗ Failed")
    
    # Sort by Sortino ratio (descending)
    results_sorted = sorted(results, key=lambda x: x["sortino"], reverse=True)
    
    # Display results
    print("\n" + "=" * 130)
    print("RESULTS (Sorted by Sortino Ratio)")
    print("=" * 130)
    
    df = pd.DataFrame(results_sorted)
    df = df[["top_n", "mom_lookback", "sortino", "cagr", "mdd", "win_rate", "num_trades", "profit_factor"]]
    
    # Format for display
    df_display = df.copy()
    df_display["sortino"] = df_display["sortino"].apply(lambda x: f"{x:.2f}")
    df_display["cagr"] = df_display["cagr"].apply(lambda x: f"{x:7.2f}%")
    df_display["mdd"] = df_display["mdd"].apply(lambda x: f"{x:7.2f}%")
    df_display["win_rate"] = df_display["win_rate"].apply(lambda x: f"{x:6.2f}%")
    df_display["profit_factor"] = df_display["profit_factor"].apply(lambda x: f"{x:6.2f}")
    
    print(df_display.to_string(index=False))
    print("=" * 130)
    
    # Show top 5
    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 80)
    for i, result in enumerate(results_sorted[:5], 1):
        print(f"\n{i}. top_n={result['top_n']}, mom_lookback={result['mom_lookback']}")
        print(f"   Sortino Ratio: {result['sortino']:.2f}")
        print(f"   CAGR:          {result['cagr']:.2f}%")
        print(f"   MDD:           {result['mdd']:.2f}%")
        print(f"   Win Rate:      {result['win_rate']:.2f}%")
        print(f"   Trades:        {result['num_trades']}")
        print(f"   Profit Factor: {result['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
