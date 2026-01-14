"""Technical Indicators Library.

Pure functions for calculating technical indicators from market data.
No trading logic, just math.
"""

import numpy as np
import pandas as pd


def calculate_noise_ratio(df: pd.DataFrame) -> pd.Series:
    """Calculate noise ratio: |Open - Close| / (High - Low)."""
    numerator = (df["open"] - df["close"]).abs()
    denominator = df["high"] - df["low"]

    # Division by zero handling: replace 0 with NaN, calculate, then fill 0
    noise = numerator / denominator.replace(0, np.nan)
    return noise.fillna(0)


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=window).mean()


def calculate_range(df: pd.DataFrame) -> pd.Series:
    """Calculate High - Low range."""
    return df["high"] - df["low"]
