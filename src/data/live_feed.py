"""Binance testnet benzeri sentetik akış."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional

import numpy as np

from src.config.settings import get_settings
from src.utils.time import utc_timestamp
from src.utils.types import BarData


@dataclass(slots=True)
class LiveFeedConfig:
    symbol: str
    seed: Optional[int] = None
    delay_seconds: float = 0.0


class BinanceLiveFeed:
    """Gerçek zamanlı borsayı taklit eden basit akış."""

    def __init__(self, config: LiveFeedConfig | None = None) -> None:
        settings = get_settings()
        if config is None:
            config = LiveFeedConfig(symbol=settings.runtime.symbol)
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._last_price = 100.0

    async def stream_klines(self) -> AsyncIterator[BarData]:
        """Sonsuz bar akışı üret."""

        delay = max(0.0, self.config.delay_seconds)
        while True:
            yield self._next_bar()
            if delay:
                await asyncio.sleep(delay)

    def prime(self, prices: Iterable[float]) -> None:
        """Testler için başlangıç fiyat serisini yükle."""

        prices = list(prices)
        if prices:
            self._last_price = float(prices[-1])

    def _next_bar(self) -> BarData:
        drift = 0.0005
        shock = float(self._rng.normal(0.0, 0.002))
        close = max(1e-3, self._last_price * (1 + drift + shock))
        high = max(self._last_price, close) * (1 + abs(float(self._rng.normal(0.0, 0.0007))))
        low = min(self._last_price, close) * (1 - abs(float(self._rng.normal(0.0, 0.0007))))
        volume = float(abs(self._rng.normal(1.0, 0.2)))
        bar = BarData(
            timestamp=utc_timestamp(),
            open=self._last_price,
            high=max(high, close),
            low=min(low, close),
            close=close,
            volume=volume,
        )
        self._last_price = close
        return bar
