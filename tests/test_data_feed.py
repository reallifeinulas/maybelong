import pytest

from src.data.live_feed import HistoricalCSVFeed, HistoricalCSVFeedConfig


@pytest.mark.asyncio
async def test_historical_csv_feed_yields_real_bars():
    config = HistoricalCSVFeedConfig(path="data/binance_top10usdt_5m_recent.csv")
    feed = HistoricalCSVFeed(config)

    symbols = set()
    ada_timestamps: list[int] = []
    async for bar in feed.stream_klines():
        assert bar.symbol is not None
        symbols.add(bar.symbol)
        if bar.symbol == "ADAUSDT":
            ada_timestamps.append(bar.timestamp)
        if len(symbols) == 10 and len(ada_timestamps) >= 2:
            break

    assert len(symbols) == 10
    assert ada_timestamps[0] == 1762478700
    assert ada_timestamps[1] - ada_timestamps[0] == 300
