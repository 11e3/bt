"""Strategy building blocks - reusable components.

Consolidated allocation, conditions, pricing, and indicators
from scattered strategy files into organized components.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from bt.interfaces.strategy_types import IAllocation, ICondition, IPricing, IStrategyComponent
from bt.utils.decimal_cache import get_decimal
from bt.utils.indicator_cache import get_indicator_cache

if TYPE_CHECKING:
    from bt.interfaces.protocols import IBacktestEngine


# === ALLOCATION COMPONENTS ===


class BaseAllocation(IStrategyComponent):
    """Base class for allocation strategies."""

    def __init__(self, **config):
        self.config = config
        self.validate()

    def validate(self) -> bool:
        """Validate allocation configuration."""
        return True

    def get_description(self) -> str:
        """Get allocation description."""
        return f"{self.__class__.__name__}({self.config})"

    def calculate_quantity(self, engine: IBacktestEngine, symbol: str, price: float) -> float:
        """IAllocation protocol method - delegates to __call__."""
        return self(engine, symbol, price)


class AllInAllocation(BaseAllocation):
    """Buy with all available cash accounting for costs."""

    def __call__(self, engine: IBacktestEngine, _symbol: str, price: float) -> float:
        if engine.portfolio.cash <= 0 or price <= 0:
            return 0.0

        # Calculate cost multiplier (1 + fee + slippage)
        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        max_affordable = float(engine.portfolio.cash) / (price * cost_multiplier)

        return max_affordable * 0.99999  # Safety buffer


class EqualWeightAllocation(BaseAllocation):
    """Equal weight allocation across all symbols."""

    def __call__(self, engine: IBacktestEngine, _symbol: str, price: float) -> float:
        num_symbols = len(engine.data_provider.symbols)
        if num_symbols == 0:
            return 0.0

        target_allocation = float(engine.portfolio.cash) / num_symbols
        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        return target_allocation / (price * cost_multiplier)


class MomentumAllocation(BaseAllocation):
    """Momentum allocation - equal weight allocation with momentum filter."""

    def __call__(self, engine: IBacktestEngine, symbol: str, price: float) -> float:
        mom_lookback = self.config.get("mom_lookback", 20)

        # Check momentum for this symbol
        bars = engine.get_bars(symbol, mom_lookback + 2)
        if bars is None or len(bars) < mom_lookback + 2:
            return 0.0

        close_prices = bars["close"].values
        prev_close = close_prices[-2]
        old_close = close_prices[-(mom_lookback + 2)]

        momentum = prev_close / old_close - 1 if old_close > 0 else -999.0
        if np.isnan(momentum) or momentum <= 0:
            return 0.0

        # Equal allocation among symbols
        num_symbols = len(engine.data_provider.symbols)
        if num_symbols == 0:
            return 0.0

        target_allocation = float(engine.portfolio.cash) / num_symbols
        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        return target_allocation / (price * cost_multiplier)


class VolatilityBreakoutAllocation(BaseAllocation):
    """VBO momentum allocation - allocate to top N momentum assets equally."""

    def __call__(self, engine: IBacktestEngine, symbol: str, price: float) -> float:
        top_n = self.config.get("top_n", 3)
        mom_lookback = self.config.get("mom_lookback", 20)

        # Vectorized momentum calculation
        momentum_data = {}
        all_symbols = engine.data_provider.symbols

        for s in all_symbols:
            bars = engine.get_bars(s, mom_lookback + 2)
            if bars is not None and len(bars) >= mom_lookback + 2:
                close_prices = bars["close"].values
                prev_close = close_prices[-2]
                old_close = close_prices[-(mom_lookback + 2)]

                momentum = prev_close / old_close - 1 if old_close > 0 else -999.0
                momentum_data[s] = momentum if not np.isnan(momentum) else -999.0
            else:
                momentum_data[s] = -999.0

        # Get top symbols
        sorted_items = sorted(momentum_data.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [item[0] for item in sorted_items[:top_n]]

        if symbol not in top_symbols:
            return 0.0

        # Equal allocation among top symbols
        total_equity = float(engine.portfolio.value)
        target_amount = total_equity / top_n

        cash = float(engine.portfolio.cash)
        buy_amount = min(target_amount, cash * 0.999)  # Safety buffer

        if buy_amount <= 0:
            return 0.0

        cost_multiplier = 1 + float(engine.config.fee) + float(engine.config.slippage)
        return buy_amount / (price * cost_multiplier)


# === CONDITION COMPONENTS ===


class BaseCondition(IStrategyComponent):
    """Base class for trading conditions."""

    def validate(self) -> bool:
        """Validate condition configuration."""
        return True

    def get_description(self) -> str:
        """Get condition description."""
        return f"{self.__class__.__name__}"

    def evaluate(self, engine: IBacktestEngine, symbol: str) -> bool:
        """ICondition protocol method - delegates to __call__."""
        return self(engine, symbol)


class NoOpenPositionCondition(BaseCondition):
    """True when no open position exists."""

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        position = engine.portfolio.get_position(symbol)
        return not position.is_open


class PriceAboveSMACondition(BaseCondition):
    """True when price is above SMA."""

    def __init__(self, lookback: int, use_current_bar: bool = False):
        self.lookback = lookback
        self.use_current_bar = use_current_bar

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        bars = engine.get_bars(symbol, self.lookback + 1)
        if bars is None or len(bars) < self.lookback + 1:
            return False

        # Use cached SMA calculation
        cache = get_indicator_cache()
        close_series = bars["close"].iloc[:-1] if not self.use_current_bar else bars["close"]
        sma_value = cache.calculate_indicator(symbol, "sma", self.lookback, close_series)

        current_price = float(bars["close"].iloc[-1])
        return current_price > sma_value


class VolatilityBreakoutCondition(BaseCondition):
    """VBO breakout condition - price above volatility threshold."""

    def __init__(self, k_factor: float = 0.5, lookback: int = 5, **_kwargs):
        self.k_factor = k_factor
        self.lookback = lookback

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        current_bar = engine.get_bar(symbol)
        if current_bar is None:
            return False

        # Calculate buy price (using centralized pricing component)
        buy_price = self._calculate_vbo_buy_price(engine, symbol)
        current_high = float(current_bar["high"])

        return current_high >= buy_price

    def _calculate_vbo_buy_price(self, engine: IBacktestEngine, symbol: str) -> float:
        """Calculate VBO buy price using centralized logic."""
        lookback = self.lookback
        bars = engine.get_bars(symbol, lookback + 1)

        if bars is None or len(bars) < lookback + 1:
            return 0.0

        # Calculate noise ratio
        close_prices = bars["close"].iloc[:-1]  # Exclude current bar
        high_prices = bars["high"].iloc[:-1]
        low_prices = bars["low"].iloc[:-1]

        if len(close_prices) < lookback:
            return 0.0

        # Range calculation
        avg_range = np.mean(high_prices - low_prices)

        # Calculate noise ratio
        noise_ratio = self._calculate_noise_ratio(close_prices)

        # VBO buy price calculation
        last_close = float(close_prices.iloc[-1])
        noise_adjusted_range = avg_range * noise_ratio
        return last_close + noise_adjusted_range * self.k_factor

    def _calculate_noise_ratio(self, prices: pd.Series) -> float:
        """Calculate noise ratio for volatility adjustment.

        Noise ratio = sum of absolute daily changes / total price range
        Higher value indicates choppy/noisy market, lower indicates trending.
        """
        if len(prices) < 2:
            return 1.0

        # Sum of absolute daily changes
        changes = prices.diff().abs()
        total_abs_changes = changes.sum()

        # Total price range
        total_range = abs(prices.iloc[-1] - prices.iloc[0])

        if total_range == 0:
            return 1.0

        # Calculate and clamp to reasonable range
        noise_ratio = total_abs_changes / total_range
        return float(min(max(noise_ratio, 0.1), 2.0))


# === PRICING COMPONENTS ===


class BasePricing(IStrategyComponent):
    """Base class for pricing strategies."""

    def validate(self) -> bool:
        """Validate pricing configuration."""
        return True

    def get_description(self) -> str:
        """Get pricing description."""
        return f"{self.__class__.__name__}"

    def calculate_price(self, engine: IBacktestEngine, symbol: str) -> float:
        """IPricing protocol method - delegates to __call__."""
        return self(engine, symbol)


class CurrentClosePricing(BasePricing):
    """Uses current close price for execution."""

    def __call__(self, engine: IBacktestEngine, symbol: str) -> float:
        bar = engine.get_bar(symbol)
        if bar is None:
            return 0.0
        return float(bar["close"])


class CurrentOpenPricing(BasePricing):
    """Uses current open price for execution."""

    def __call__(self, engine: IBacktestEngine, symbol: str) -> float:
        bar = engine.get_bar(symbol)
        if bar is None:
            return 0.0
        return float(bar["open"])


class VolatilityBreakoutPricing(BasePricing):
    """VBO pricing - calculates breakout buy price."""

    def __init__(self, lookback: int = 5, k_factor: float = 0.5, **_kwargs):
        self.lookback = lookback
        self.k_factor = k_factor

    def __call__(self, engine: IBacktestEngine, symbol: str) -> float:
        lookback = self.lookback
        bars = engine.get_bars(symbol, lookback + 1)

        if bars is None or len(bars) < lookback + 1:
            return 0.0

        # Use centralized VBO price calculation
        return self._calculate_vbo_buy_price(symbol, bars)

    def _calculate_vbo_buy_price(self, _symbol: str, bars: pd.DataFrame) -> float:
        """Calculate VBO buy price."""
        lookback = self.lookback

        # Calculate range and noise
        close_prices = bars["close"].iloc[:-1]
        high_prices = bars["high"].iloc[:-1]
        low_prices = bars["low"].iloc[:-1]

        if len(close_prices) < lookback:
            return float(bars["close"].iloc[-1])

        avg_range = np.mean(high_prices - low_prices)
        noise_ratio = 1.0  # Simplified

        # Calculate buy price
        last_close = float(close_prices.iloc[-1])
        noise_adjusted_range = avg_range * noise_ratio
        return last_close + noise_adjusted_range * self.k_factor


# === INDICATOR COMPONENTS ===


class BaseIndicator(IStrategyComponent):
    """Base class for technical indicators."""

    def validate(self) -> bool:
        """Validate indicator configuration."""
        return True


class SMAIndicator(BaseIndicator):
    """Simple Moving Average indicator with caching."""

    def __init__(self, lookback: int):
        self.lookback = lookback

    def calculate(self, symbol: str, prices: pd.Series) -> float:
        """Calculate SMA using optimized cache."""
        cache = get_indicator_cache()
        return cache.calculate_indicator(symbol, "sma", self.lookback, prices)


class EMAIndicator(BaseIndicator):
    """Exponential Moving Average indicator with caching."""

    def __init__(self, lookback: int):
        self.lookback = lookback

    def calculate(self, symbol: str, prices: pd.Series) -> float:
        """Calculate EMA using optimized cache."""
        cache = get_indicator_cache()
        return cache.calculate_indicator(symbol, "ema", self.lookback, prices)


class RSIIndicator(BaseIndicator):
    """Relative Strength Index indicator with caching."""

    def __init__(self, lookback: int = 14):
        self.lookback = lookback

    def calculate(self, symbol: str, prices: pd.Series) -> float:
        """Calculate RSI using optimized cache."""
        cache = get_indicator_cache()
        return cache.calculate_indicator(symbol, "rsi", self.lookback, prices)


class MomentumIndicator(BaseIndicator):
    """Momentum indicator for trend analysis."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def calculate(self, prices: pd.Series) -> float:
        """Calculate momentum over lookback period."""
        if len(prices) < self.lookback + 1:
            return 0.0

        current_price = float(prices.iloc[-2])  # Previous day
        old_price = float(prices.iloc[-(self.lookback + 1)])  # Lookback days ago

        return (current_price / old_price - 1) if old_price > 0 else -999.0


# === VBO PORTFOLIO COMPONENTS ===


class VBOPortfolioBuyCondition(BaseCondition):
    """VBO Portfolio buy condition with BTC market filter.

    Buy signal when:
    - Current high >= target price (open + range * noise_ratio)
    - Previous close > Previous MA5
    - Previous BTC close > Previous BTC MA20
    """

    def __init__(
        self,
        ma_short: int = 5,
        btc_ma: int = 20,
        noise_ratio: float = 0.5,
        btc_symbol: str = "BTC",
        **_kwargs,
    ):
        self.ma_short = ma_short
        self.btc_ma = btc_ma
        self.noise_ratio = noise_ratio
        self.btc_symbol = btc_symbol

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        # Get current bar
        current_bar = engine.get_bar(symbol)
        if current_bar is None:
            return False

        # Get historical bars for this symbol
        # Need ma_short + 1 bars: ma_short for MA calculation + 1 current bar (which gets excluded)
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

        # Get BTC data for market filter
        # Need btc_ma + 1 bars: btc_ma for MA calculation + 1 current bar (which gets excluded)
        btc_bars = engine.get_bars(self.btc_symbol, self.btc_ma + 1)
        if btc_bars is None or len(btc_bars) < self.btc_ma + 1:
            return False

        # Calculate BTC previous values
        btc_close_series = btc_bars["close"].iloc[:-1]
        if len(btc_close_series) < self.btc_ma:
            return False
        prev_btc_close = float(btc_close_series.iloc[-1])
        prev_btc_ma20 = float(btc_close_series.iloc[-self.btc_ma :].mean())

        # Check BTC market condition: prev_btc_close > prev_btc_ma20
        if prev_btc_close <= prev_btc_ma20:
            return False

        # Calculate target price: open + (prev_high - prev_low) * noise_ratio
        current_open = float(current_bar["open"])
        target_price = current_open + (prev_high - prev_low) * self.noise_ratio

        # Check breakout condition: current high >= target price
        current_high = float(current_bar["high"])
        return current_high >= target_price


class VBOPortfolioSellCondition(BaseCondition):
    """VBO Portfolio sell condition.

    Sell signal when:
    - Previous close < Previous MA5 OR
    - Previous BTC close < Previous BTC MA20
    """

    def __init__(
        self,
        ma_short: int = 5,
        btc_ma: int = 20,
        btc_symbol: str = "BTC",
        **_kwargs,
    ):
        self.ma_short = ma_short
        self.btc_ma = btc_ma
        self.btc_symbol = btc_symbol

    def __call__(self, engine: IBacktestEngine, symbol: str) -> bool:
        # Get historical bars for this symbol
        # Need ma_short + 1 bars: ma_short for MA calculation + 1 current bar (which gets excluded)
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

        # Get BTC data for market filter
        # Need btc_ma + 1 bars: btc_ma for MA calculation + 1 current bar (which gets excluded)
        btc_bars = engine.get_bars(self.btc_symbol, self.btc_ma + 1)
        if btc_bars is None or len(btc_bars) < self.btc_ma + 1:
            return coin_sell_signal  # If no BTC data, rely on coin signal only

        # Calculate BTC previous values
        btc_close_series = btc_bars["close"].iloc[:-1]
        if len(btc_close_series) < self.btc_ma:
            return coin_sell_signal
        prev_btc_close = float(btc_close_series.iloc[-1])
        prev_btc_ma20 = float(btc_close_series.iloc[-self.btc_ma :].mean())

        # Check BTC market exit condition
        btc_sell_signal = prev_btc_close < prev_btc_ma20

        # Sell if either condition is met
        return coin_sell_signal or btc_sell_signal


# === REGIME MODEL COMPONENTS ===


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
        """
        if self._model is not None:
            return self._model

        import joblib  # type: ignore[import-untyped]

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


class VBOPortfolioPricing(BasePricing):
    """VBO Portfolio pricing - calculates target buy price.

    Target price = Open + (Prev High - Prev Low) * noise_ratio

    Note: This returns the raw target price without slippage.
    Slippage is applied by portfolio.buy() during execution.
    """

    def __init__(self, noise_ratio: float = 0.5, **_kwargs):
        self.noise_ratio = noise_ratio

    def __call__(self, engine: IBacktestEngine, symbol: str) -> float:
        current_bar = engine.get_bar(symbol)
        if current_bar is None:
            return 0.0

        # Get previous bar for range calculation
        bars = engine.get_bars(symbol, 2)
        if bars is None or len(bars) < 2:
            return float(current_bar["close"])

        prev_high = float(bars["high"].iloc[-2])
        prev_low = float(bars["low"].iloc[-2])
        current_open = float(current_bar["open"])

        # Target price = open + (prev_high - prev_low) * noise_ratio
        return current_open + (prev_high - prev_low) * self.noise_ratio


class VBOPortfolioAllocation(BaseAllocation):
    """VBO Portfolio allocation - equal weight among all portfolio coins (1/N).

    Allocates total_equity / n_strategies to each coin.
    Like Upbit, supports fractional (decimal) quantities.

    This returns the target buy_value, not quantity.
    The backtest engine will calculate the actual quantity based on execution price.

    To match standalone logic exactly:
    - buy_value = min(target_alloc, cash * 0.99)
    - Quantity calculation is done by backtest engine using:
      qty = (buy_value - buy_fee) / buy_price
    """

    def __call__(self, engine: IBacktestEngine, _symbol: str, price: float) -> float:
        if price <= 0:
            return 0.0

        # Get number of symbols (n_strategies)
        symbols = engine.data_provider.symbols
        n_strategies = len(symbols)
        if n_strategies == 0:
            return 0.0

        # Calculate total equity = cash + position values at open prices
        cash = float(engine.portfolio.cash)
        position_value = 0.0
        for sym in symbols:
            pos = engine.portfolio.get_position(sym)
            if pos.is_open:
                bar = engine.get_bar(sym)
                if bar is not None:
                    open_price = float(bar["open"])
                    position_value += float(pos.quantity) * open_price

        total_equity = cash + position_value
        target_alloc = total_equity / n_strategies

        # Match standalone logic: buy_value limited to 99% of cash
        buy_value = min(target_alloc, cash * 0.99)

        if buy_value <= 0:
            return 0.0

        # Return quantity that when processed by portfolio.buy() results in
        # spending exactly buy_value
        #
        # portfolio.buy() calculates:
        #   exec_price = price * (1 + slippage)
        #   cost = exec_price * qty * (1 + fee)
        #
        # We want: cost = buy_value
        # So: qty = buy_value / (price * (1 + slippage) * (1 + fee))
        fee = float(engine.config.fee)
        slippage = float(engine.config.slippage)
        return buy_value / (price * (1 + slippage) * (1 + fee))


# === FACTORY FUNCTIONS ===


def create_allocation(allocation_type: str, **config) -> IAllocation:
    """Factory function for allocation strategies."""

    allocations = {
        "all_in": AllInAllocation,
        "equal_weight": EqualWeightAllocation,
        "equal_weight_momentum": MomentumAllocation,
        "volatility_breakout": VolatilityBreakoutAllocation,
        "vbo_portfolio": VBOPortfolioAllocation,
    }

    if allocation_type not in allocations:
        raise ValueError(f"Unknown allocation type: {allocation_type}")

    return allocations[allocation_type](**config)


def create_condition(condition_type: str, **config: Any) -> ICondition:
    """Factory function for condition strategies."""

    conditions: dict[str, type[BaseCondition]] = {
        "no_open_position": NoOpenPositionCondition,
        "price_above_sma": PriceAboveSMACondition,
        "volatility_breakout": VolatilityBreakoutCondition,
        "vbo_portfolio_buy": VBOPortfolioBuyCondition,
        "vbo_portfolio_sell": VBOPortfolioSellCondition,
        "vbo_regime_buy": VBORegimeBuyCondition,
        "vbo_regime_sell": VBORegimeSellCondition,
    }

    if condition_type not in conditions:
        raise ValueError(f"Unknown condition type: {condition_type}")

    return conditions[condition_type](**config)


def create_pricing(pricing_type: str, **config) -> IPricing:
    """Factory function for pricing strategies."""

    pricing_strategies = {
        "current_close": CurrentClosePricing,
        "current_open": CurrentOpenPricing,
        "volatility_breakout": VolatilityBreakoutPricing,
        "vbo_portfolio": VBOPortfolioPricing,
    }

    if pricing_type not in pricing_strategies:
        raise ValueError(f"Unknown pricing type: {pricing_type}")

    return pricing_strategies[pricing_type](**config)


def create_indicator(indicator_type: str, **config) -> BaseIndicator:
    """Factory function for technical indicators."""

    indicators = {
        "sma": SMAIndicator,
        "ema": EMAIndicator,
        "rsi": RSIIndicator,
        "momentum": MomentumIndicator,
    }

    if indicator_type not in indicators:
        raise ValueError(f"Unknown indicator type: {indicator_type}")

    return indicators[indicator_type](**config)
