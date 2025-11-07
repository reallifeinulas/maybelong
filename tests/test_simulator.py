from src.execution.simulator import PaperTrader
from src.utils.types import BarData


def make_bar(price: float) -> BarData:
    return BarData(timestamp=0, open=price, high=price, low=price, close=price, volume=1.0)


def test_paper_trader_min_hold():
    trader = PaperTrader()
    bar = make_bar(100.0)
    pnl = trader.step(bar, "LONG", size=1.0)
    assert pnl == 0.0
    for _ in range(5):
        pnl = trader.step(bar, "LONG", size=1.0)
    assert trader.position is not None
    pnl = trader.step(bar, "FLAT", size=0.0)
    assert trader.position is None
    assert isinstance(pnl, float)
