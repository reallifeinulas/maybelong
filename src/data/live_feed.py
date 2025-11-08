"""Gerçek ve sentetik veri akışlarını sağlayan yardımcılar."""

from __future__ import annotations

import asyncio
import csv
from datetime import datetime
from pathlib import Path
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


@dataclass(slots=True)
class HistoricalCSVFeedConfig:
    """CSV tabanlı gerçek veri akışı yapılandırması."""

    path: str
    timestamp_format: Optional[str] = None
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
            symbol=self.config.symbol,
        )
        self._last_price = close
        return bar


class HistoricalCSVFeed:
    """Yerel CSV dosyasından gerçek OHLCV barlarını yayınla."""

    def __init__(self, config: HistoricalCSVFeedConfig) -> None:
        self.config = config
        self._bars = self._load_bars(self._resolve_path(config.path))

    async def stream_klines(self) -> AsyncIterator[BarData]:
        delay = max(0.0, self.config.delay_seconds)
        for bar in self._bars:
            yield bar
            if delay:
                await asyncio.sleep(delay)

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _load_bars(self, path: Path) -> list[BarData]:
        if not path.exists():
            raise FileNotFoundError(f"CSV veri kaynağı bulunamadı: {path}")

        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV dosyasında başlık satırı bulunamadı.")

            required = {"timestamp", "open", "high", "low", "close", "volume"}
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV dosyasında eksik sütun(lar): {', '.join(sorted(missing))}")

            bars: list[BarData] = []
            has_symbol = "symbol" in reader.fieldnames
            for row in reader:
                timestamp = self._parse_timestamp(row["timestamp"])
                symbol = row["symbol"].strip() if has_symbol and row.get("symbol") else None
                bars.append(
                    BarData(
                        timestamp=timestamp,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=float(row["volume"]),
                        symbol=symbol,
                    )
                )

        if not bars:
            raise ValueError("CSV dosyası boş görünüyor; yayınlanacak bar yok.")

        bars.sort(key=lambda bar: (bar.timestamp, bar.symbol or ""))

        return bars

    def _parse_timestamp(self, value: str) -> int:
        value = value.strip()
        if not value:
            raise ValueError("Zaman damgası boş olamaz.")

        if self.config.timestamp_format:
            dt = datetime.strptime(value, self.config.timestamp_format)
            return int(dt.timestamp())

        if value.isdigit():
            return int(value)

        try:
            return int(float(value))
        except ValueError:
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"Zaman damgası çözümlenemedi: {value!r}") from exc
            return int(dt.timestamp())
