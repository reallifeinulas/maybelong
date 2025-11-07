"""Tip tanımları ve ortak veri yapıları.

Örnek:
    from src.utils.types import BarData

    bar = BarData(timestamp=1234567890, open=100.0, high=110.0, low=95.0, close=105.0, volume=1.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Decision = Literal["LONG", "SHORT", "FLAT"]


@dataclass(slots=True)
class BarData:
    """Basit OHLCV bar verisi taşıyıcısı."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class TradeDecision:
    """Politika çıktısını ve meta verileri paketler."""

    decision: Decision
    confidence: float
    size: float
    reason: Optional[str] = None


@dataclass(slots=True)
class PerformanceSnapshot:
    """Anlık performans özetini modeller."""

    equity: float
    winrate: float
    profit_factor: float
    sharpe: float
    roi: float
    max_drawdown: float
