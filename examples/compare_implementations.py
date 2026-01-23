#!/usr/bin/env python3
"""Compare standalone backtest vs framework implementation.

This script runs the same VBO strategy using both:
1. Standalone implementation (backtest_vbo_comparison.py logic)
2. Framework implementation (bt.strategies.implementations.VBOSingleCoinStrategy)

And compares the results to identify any differences.
"""

from pathlib import Path

import pandas as pd

# =============================================================================
# Configuration (same as standalone)
# =============================================================================
FEE = 0.0005
SLIPPAGE = 0.0005
MA_SHORT = 5
BTC_MA = 20
NOISE_RATIO = 0.5
INITIAL_CAPITAL = 1_000_000


# =============================================================================
# Standalone Implementation (copied from backtest_vbo_comparison.py)
# =============================================================================
def load_data(symbol: str, data_dir: str = "data") -> pd.DataFrame:
    """Load OHLCV data for a single symbol."""
    # Try parquet first, then CSV
    parquet_path = Path(data_dir) / f"{symbol}.parquet"
    csv_path = Path(data_dir) / f"{symbol}.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
    else:
        raise FileNotFoundError(f"No data for {symbol}: {parquet_path} or {csv_path}")

    if "datetime" in df.columns:
        df.set_index("datetime", inplace=True)
    return df.sort_index()


def calculate_indicators(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the strategy."""
    df = df.copy()
    btc_df = btc_df.copy()

    btc_aligned = btc_df.reindex(df.index, method="ffill")

    df["ma5"] = df["close"].rolling(window=MA_SHORT).mean()
    btc_aligned["btc_ma20"] = btc_aligned["close"].rolling(window=BTC_MA).mean()

    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_close"] = df["close"].shift(1)
    df["prev_ma5"] = df["ma5"].shift(1)

    df["prev_btc_close"] = btc_aligned["close"].shift(1)
    df["prev_btc_ma20"] = btc_aligned["btc_ma20"].shift(1)

    df["target_price"] = df["open"] + (df["prev_high"] - df["prev_low"]) * NOISE_RATIO

    return df


def backtest_standalone(symbol: str, start: str = None, end: str = None) -> dict:
    """Run standalone backtest."""
    df = load_data(symbol)
    btc_df = load_data("BTC")

    if start:
        df = df[df.index >= pd.to_datetime(start)]
        btc_df = btc_df[btc_df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
        btc_df = btc_df[btc_df.index <= pd.to_datetime(end)]

    df = calculate_indicators(df, btc_df)

    cash = INITIAL_CAPITAL
    position = 0.0
    trades = []
    equity_curve = []

    for date, row in df.iterrows():
        if pd.isna(row["prev_ma5"]) or pd.isna(row["prev_btc_ma20"]):
            equity = cash + position * row["close"]
            equity_curve.append({"date": date, "equity": equity})
            continue

        # SELL
        if position > 0:
            sell_signal = (row["prev_close"] < row["prev_ma5"]) or (
                row["prev_btc_close"] < row["prev_btc_ma20"]
            )
            if sell_signal:
                sell_price = row["open"] * (1 - SLIPPAGE)
                sell_value = position * sell_price
                sell_fee = sell_value * FEE
                cash += sell_value - sell_fee
                trades.append(
                    {
                        "date": date,
                        "action": "sell",
                        "price": sell_price,
                        "quantity": position,
                        "value": sell_value - sell_fee,
                    }
                )
                position = 0.0

        # BUY
        if position == 0:
            buy_signal = (
                row["high"] >= row["target_price"]
                and row["prev_close"] > row["prev_ma5"]
                and row["prev_btc_close"] > row["prev_btc_ma20"]
            )
            if buy_signal:
                buy_price = max(row["target_price"], row["open"]) * (1 + SLIPPAGE)
                buy_value = cash
                buy_fee = buy_value * FEE
                position = (buy_value - buy_fee) / buy_price
                trades.append(
                    {
                        "date": date,
                        "action": "buy",
                        "price": buy_price,
                        "quantity": position,
                        "value": buy_value,
                    }
                )
                cash = 0.0

        equity = cash + position * row["close"]
        equity_curve.append({"date": date, "equity": equity})

    # Close position
    if position > 0:
        last_row = df.iloc[-1]
        final_price = last_row["close"] * (1 - SLIPPAGE)
        final_value = position * final_price
        final_fee = final_value * FEE
        cash += final_value - final_fee

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    final_equity = equity_df["equity"].iloc[-1]

    return {
        "final_equity": final_equity,
        "total_return": (final_equity / INITIAL_CAPITAL - 1) * 100,
        "num_trades": len([t for t in trades if t["action"] == "buy"]),
        "equity_curve": equity_df,
        "trades": trades,
    }


# =============================================================================
# Framework Implementation
# =============================================================================
def backtest_framework(symbol: str, start: str = None, end: str = None) -> dict:
    """Run framework backtest."""
    from bt.framework import BacktestFramework

    # Load data
    df = load_data(symbol)
    btc_df = load_data("BTC")

    if start:
        df = df[df.index >= pd.to_datetime(start)]
        btc_df = btc_df[btc_df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
        btc_df = btc_df[btc_df.index <= pd.to_datetime(end)]

    # CRITICAL: Align BTC data to coin's datetime index (matching standalone behavior)
    # The standalone does: btc_aligned = btc_df.reindex(df.index, method='ffill')
    btc_aligned = btc_df.reindex(df.index, method="ffill")

    # Reset index for framework
    df = df.reset_index()
    btc_aligned = btc_aligned.reset_index()

    # Prepare data dict (using aligned BTC data)
    data = {symbol: df, "BTC": btc_aligned}

    # Create framework with matching config (dict format)
    framework_config = {
        "initial_cash": INITIAL_CAPITAL,
        "fee": FEE,
        "slippage": SLIPPAGE,
        "btc_ma": BTC_MA,  # For start_idx calculation
    }

    framework = BacktestFramework(config=framework_config)

    # Run backtest
    result = framework.run_backtest(
        strategy="vbo_single_coin",
        symbols=[symbol],  # Only trade the target symbol
        data=data,
        config={
            "ma_short": MA_SHORT,
            "btc_ma": BTC_MA,
            "noise_ratio": NOISE_RATIO,
            "btc_symbol": "BTC",
        },
    )

    # Extract results
    equity_curve = result.get("equity_curve", {})
    dates = equity_curve.get("dates", [])
    values = equity_curve.get("values", [])

    if dates and values:
        equity_df = pd.DataFrame({"equity": values}, index=pd.to_datetime(dates))
        final_equity = values[-1]
    else:
        equity_df = pd.DataFrame()
        final_equity = INITIAL_CAPITAL

    return {
        "final_equity": final_equity,
        "total_return": (final_equity / INITIAL_CAPITAL - 1) * 100,
        "num_trades": len(result.get("trades", [])),
        "equity_curve": equity_df,
        "trades": result.get("trades", []),
        "raw_result": result,
    }


# =============================================================================
# Comparison
# =============================================================================
def debug_first_signal(symbol: str):
    """Debug why first signal differs."""
    df = load_data(symbol)
    btc_df = load_data("BTC")
    df = calculate_indicators(df, btc_df)

    print("\n" + "=" * 80)
    print("DEBUG: First few days with potential buy signals")
    print("=" * 80)

    # Find first few buy signals in standalone
    buy_count = 0
    for date, row in df.iterrows():
        if pd.isna(row["prev_ma5"]) or pd.isna(row["prev_btc_ma20"]):
            continue

        buy_signal = (
            row["high"] >= row["target_price"]
            and row["prev_close"] > row["prev_ma5"]
            and row["prev_btc_close"] > row["prev_btc_ma20"]
        )

        if buy_signal:
            buy_count += 1
            print(f"\n{date}")
            print(
                f"  high={row['high']:.2f}, target={row['target_price']:.2f}, breakout={row['high'] >= row['target_price']}"
            )
            print(
                f"  prev_close={row['prev_close']:.2f}, prev_ma5={row['prev_ma5']:.2f}, trend={row['prev_close'] > row['prev_ma5']}"
            )
            print(
                f"  prev_btc_close={row['prev_btc_close']:.2f}, prev_btc_ma20={row['prev_btc_ma20']:.2f}, btc_trend={row['prev_btc_close'] > row['prev_btc_ma20']}"
            )

            if buy_count >= 5:
                break


def compare_results(symbol: str, start: str = None, end: str = None):
    """Compare standalone vs framework results."""
    print("=" * 80)
    print(f"Comparing implementations for {symbol}")
    print("=" * 80)
    print(f"Config: MA{MA_SHORT}, BTC_MA{BTC_MA}, k={NOISE_RATIO}")
    print(f"Fee: {FEE * 100}%, Slippage: {SLIPPAGE * 100}%")
    print()

    # Debug first signals
    debug_first_signal(symbol)

    # Run both
    print("Running standalone backtest...", end=" ", flush=True)
    standalone = backtest_standalone(symbol, start, end)
    print("Done")

    print("Running framework backtest...", end=" ", flush=True)
    framework = backtest_framework(symbol, start, end)
    print("Done")

    # Compare
    print()
    print("-" * 80)
    print("RESULTS COMPARISON")
    print("-" * 80)
    print(f"{'Metric':<25} {'Standalone':<20} {'Framework':<20} {'Diff':<15}")
    print("-" * 80)

    # Final equity
    s_eq = standalone["final_equity"]
    f_eq = framework["final_equity"]
    diff_eq = (f_eq - s_eq) / s_eq * 100 if s_eq != 0 else 0
    print(f"{'Final Equity':<25} {s_eq:>18,.0f} {f_eq:>18,.0f} {diff_eq:>13.4f}%")

    # Total return
    s_ret = standalone["total_return"]
    f_ret = framework["total_return"]
    diff_ret = f_ret - s_ret
    print(f"{'Total Return (%)':<25} {s_ret:>18.2f} {f_ret:>18.2f} {diff_ret:>13.4f}")

    # Trades
    s_trades = standalone["num_trades"]
    f_trades = framework["num_trades"]
    print(f"{'Number of Trades':<25} {s_trades:>18} {f_trades:>18} {f_trades - s_trades:>13}")

    print("-" * 80)

    # Trade details
    print()
    print("TRADE DETAILS (first 10)")
    print("-" * 80)

    print("\nStandalone trades:")
    for i, t in enumerate(standalone["trades"][:10]):
        print(f"  {i + 1}. {t['date']} {t['action']:4} @ {t['price']:,.2f} x {t['quantity']:.6f}")

    print("\nFramework trades:")
    for i, t in enumerate(framework["trades"][:10]):
        if isinstance(t, dict):
            print(f"  {i + 1}. {t}")
        else:
            print(f"  {i + 1}. {t}")

    return standalone, framework


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare backtest implementations")
    parser.add_argument("--symbol", type=str, default="ETH", help="Symbol to test")
    parser.add_argument("--start", type=str, help="Start date")
    parser.add_argument("--end", type=str, help="End date")
    args = parser.parse_args()

    try:
        compare_results(args.symbol, args.start, args.end)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure data files exist in the 'data' directory")


if __name__ == "__main__":
    main()
