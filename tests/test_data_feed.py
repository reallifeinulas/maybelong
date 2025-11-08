import pytest

from src.data.live_feed import HistoricalCSVFeed, HistoricalCSVFeedConfig


@pytest.mark.asyncio
async def test_historical_csv_feed_yields_real_bars():
    config = HistoricalCSVFeedConfig(path="data/btcusdt_1m_2023-01-01.csv")
    feed = HistoricalCSVFeed(config)

    bars = []
    async for bar in feed.stream_klines():
        bars.append(bar)
        if len(bars) >= 3:
            break

    assert len(bars) == 3
    assert bars[0].timestamp == 1672531200
    assert bars[0].close == pytest.approx(16543.67)
    assert bars[1].open == pytest.approx(16543.04)
