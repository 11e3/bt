"""Regime model components for ML-based market classification."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from bt.strategies.components.conditions import BaseCondition

if TYPE_CHECKING:
    from bt.interfaces.protocols import IBacktestEngine


class RegimeModelLoader:
    """Singleton loader for regime classification model.

    Loads the ML model once and caches it for reuse.
    """

    _instance: RegimeModelLoader | None = None
    _model: dict[str, Any] | None = None

    def __new__(cls) -> RegimeModelLoader:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_path: str | Path) -> dict[str, Any]:
        """Load regime classifier model from joblib file.

        Args:
            model_path: Path to the model file (.joblib)

        Returns:
            Dict containing model, scaler, label_encoder, feature_names, classes

        Raises:
            ImportError: If joblib is not installed (install with `pip install bt[ml]`)
            FileNotFoundError: If model file does not exist
        """
        if self._model is not None:
            return self._model

        try:
            import joblib  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "ML dependencies not installed. Install with: pip install bt[ml]"
            ) from e

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = joblib.load(model_path)
        return self._model

    def get_model(self) -> dict[str, Any] | None:
        """Get cached model if loaded."""
        return self._model

    def clear_cache(self) -> None:
        """Clear cached model."""
        self._model = None


def get_regime_model_loader() -> RegimeModelLoader:
    """Get singleton instance of RegimeModelLoader."""
    return RegimeModelLoader()


def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 5 features for regime classification.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume
            Index should be datetime

    Returns:
        DataFrame with 5 features:
        - return_20d: 20-day return (momentum)
        - volatility: 20-day rolling volatility
        - rsi: 14-day RSI
        - ma_alignment: MA trend alignment score
        - volume_ratio_20: Volume vs 20-day average
    """
    result = pd.DataFrame(index=df.index)

    # 1. return_20d - 20-day return
    result["return_20d"] = df["close"].pct_change(20)

    # 2. volatility - 20-day rolling volatility
    daily_returns = df["close"].pct_change()
    result["volatility"] = daily_returns.rolling(window=20).std()

    # 3. rsi - 14-day RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    result["rsi"] = 100 - (100 / (1 + rs))

    # 4. ma_alignment - MA alignment score
    ma_5 = df["close"].rolling(window=5).mean()
    ma_20 = df["close"].rolling(window=20).mean()
    ma_60 = df["close"].rolling(window=60).mean()

    # Alignment score: 5 > 20 > 60 gives +2, reverse gives -2
    alignment = pd.Series(0, index=df.index, dtype=float)
    alignment = alignment + (ma_5 > ma_20).astype(int)
    alignment = alignment + (ma_20 > ma_60).astype(int)
    alignment = alignment - (ma_5 < ma_20).astype(int)
    alignment = alignment - (ma_20 < ma_60).astype(int)
    result["ma_alignment"] = alignment

    # 5. volume_ratio_20 - volume ratio
    volume_ma_20 = df["volume"].rolling(window=20).mean()
    result["volume_ratio_20"] = df["volume"] / volume_ma_20

    return result


def predict_regime(clf_data: dict[str, Any], ohlcv_df: pd.DataFrame) -> pd.Series:
    """Predict regime from OHLCV data.

    Args:
        clf_data: Loaded classifier dict from joblib
        ohlcv_df: OHLCV DataFrame

    Returns:
        Series with regime predictions ("BULL_TREND" or "NOT_BULL")
    """
    # Calculate features
    features = calculate_regime_features(ohlcv_df)

    # Drop NaN rows
    features = features.dropna()

    if len(features) == 0:
        raise ValueError("Not enough data to calculate features (need at least 60 rows)")

    # Scale and predict
    x_scaled = clf_data["scaler"].transform(features[clf_data["feature_names"]])
    pred_encoded = clf_data["model"].predict(x_scaled)
    predictions = clf_data["label_encoder"].inverse_transform(pred_encoded)

    return pd.Series(predictions, index=features.index, name="regime")


class VBORegimeBuyCondition(BaseCondition):
    """VBO buy condition with ML regime model filter.

    Buy signal when:
    - Current high >= target price (open + range * noise_ratio)
    - Previous close > Previous MA5
    - BTC regime == "BULL_TREND" (from ML model)
    """

    def __init__(
        self,
        ma_short: int = 5,
        noise_ratio: float = 0.5,
        btc_symbol: str = "BTC",
        model_path: str | None = None,
        **_kwargs: Any,
    ):
        self.ma_short = ma_short
        self.noise_ratio = noise_ratio
        self.btc_symbol = btc_symbol
        self.model_path = model_path
        self._model_loaded = False

    def _ensure_model_loaded(self) -> dict[str, Any] | None:
        """Ensure the regime model is loaded."""
        if not self._model_loaded and self.model_path:
            loader = get_regime_model_loader()
            try:
                loader.load_model(self.model_path)
                self._model_loaded = True
            except FileNotFoundError:
                return None
        return get_regime_model_loader().get_model()

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        # Get current bar
        current_bar = engine.get_bar(symbol)
        if current_bar is None:
            return False

        # Get historical bars for this symbol
        bars = engine.get_bars(symbol, self.ma_short + 1)
        if bars is None or len(bars) < self.ma_short + 1:
            return False

        # Calculate previous values (excluding current bar)
        prev_close = float(bars["close"].iloc[-2])
        prev_high = float(bars["high"].iloc[-2])
        prev_low = float(bars["low"].iloc[-2])

        # Calculate MA5 on previous closes (excluding current bar)
        close_series = bars["close"].iloc[:-1]
        if len(close_series) < self.ma_short:
            return False
        prev_ma5 = float(close_series.iloc[-self.ma_short :].mean())

        # Check coin trend condition: prev_close > prev_ma5
        if prev_close <= prev_ma5:
            return False

        # Check BTC regime using ML model
        model = self._ensure_model_loaded()
        if model is None:
            return False

        # Get BTC data for regime prediction (need at least 60 bars for features)
        btc_bars = engine.get_bars(self.btc_symbol, 65)
        if btc_bars is None or len(btc_bars) < 65:
            return False

        # Predict regime (excluding current bar)
        btc_df = btc_bars.iloc[:-1].copy()
        try:
            regime_series = predict_regime(model, btc_df)
            if len(regime_series) == 0:
                return False
            # Use the latest regime prediction
            latest_regime = regime_series.iloc[-1]
            if latest_regime != "BULL_TREND":
                return False
        except (ValueError, KeyError):
            return False

        # Calculate target price: open + (prev_high - prev_low) * noise_ratio
        current_open = float(current_bar["open"])
        target_price = current_open + (prev_high - prev_low) * self.noise_ratio

        # Check breakout condition: current high >= target price
        current_high = float(current_bar["high"])
        return current_high >= target_price


class VBORegimeSellCondition(BaseCondition):
    """VBO sell condition with ML regime model filter.

    Sell signal when:
    - Previous close < Previous MA5 OR
    - BTC regime == "NOT_BULL" (from ML model)
    """

    def __init__(
        self,
        ma_short: int = 5,
        btc_symbol: str = "BTC",
        model_path: str | None = None,
        **_kwargs: Any,
    ):
        self.ma_short = ma_short
        self.btc_symbol = btc_symbol
        self.model_path = model_path
        self._model_loaded = False

    def _ensure_model_loaded(self) -> dict[str, Any] | None:
        """Ensure the regime model is loaded."""
        if not self._model_loaded and self.model_path:
            loader = get_regime_model_loader()
            try:
                loader.load_model(self.model_path)
                self._model_loaded = True
            except FileNotFoundError:
                return None
        return get_regime_model_loader().get_model()

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        # Get historical bars for this symbol
        bars = engine.get_bars(symbol, self.ma_short + 1)
        if bars is None or len(bars) < self.ma_short + 1:
            return False

        # Calculate previous values (excluding current bar)
        close_series = bars["close"].iloc[:-1]
        if len(close_series) < self.ma_short:
            return False
        prev_close = float(close_series.iloc[-1])
        prev_ma5 = float(close_series.iloc[-self.ma_short :].mean())

        # Check coin trend exit condition
        coin_sell_signal = prev_close < prev_ma5

        # Check BTC regime using ML model
        model = self._ensure_model_loaded()
        if model is None:
            return coin_sell_signal  # If no model, rely on coin signal only

        # Get BTC data for regime prediction
        btc_bars = engine.get_bars(self.btc_symbol, 65)
        if btc_bars is None or len(btc_bars) < 65:
            return coin_sell_signal

        # Predict regime (excluding current bar)
        btc_df = btc_bars.iloc[:-1].copy()
        try:
            regime_series = predict_regime(model, btc_df)
            if len(regime_series) == 0:
                return coin_sell_signal
            latest_regime = regime_series.iloc[-1]
            regime_sell_signal = latest_regime != "BULL_TREND"
        except (ValueError, KeyError):
            return coin_sell_signal

        # Sell if either condition is met
        return coin_sell_signal or regime_sell_signal
