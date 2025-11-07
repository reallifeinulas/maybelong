"""Bar verilerinden özellik üreticileri."""

from __future__ import annotations

import numpy as np
import pandas as pd

_FEATURE_COLUMNS = [
    "return_1",
    "return_5",
    "return_10",
    "volatility_10",
    "sma_ratio",
    "volume_zscore",
]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Eksik sütunlar: {sorted(missing)}")
    return df.copy()


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window=window)
    mean = rolling.mean()
    std = rolling.std(ddof=1)
    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV çerçevesinden temel faktörleri üret."""

    frame = _prepare(df)
    returns = frame["close"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    frame["return_1"] = returns
    frame["return_5"] = frame["close"].pct_change(periods=5).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    frame["return_10"] = frame["close"].pct_change(periods=10).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    frame["volatility_10"] = (
        returns.rolling(window=10).std(ddof=1).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    fast = frame["close"].rolling(window=5).mean()
    slow = frame["close"].rolling(window=15).mean()
    ratio = fast / slow
    frame["sma_ratio"] = ratio.replace([np.inf, -np.inf], 0.0).fillna(1.0)
    frame["volume_zscore"] = _rolling_zscore(frame["volume"], window=10)

    return frame[_FEATURE_COLUMNS].fillna(0.0)
