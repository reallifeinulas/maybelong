import pytest

from src.execution.risk import RiskManager
from src.execution.simulator import PaperTrader
from src.main import run_pipeline
from src.policy.bandit import ConstraintAwareBandit
from src.policy.constraints import ConstraintEvaluator
from src.signals.decision import DecisionBlender
from src.utils.types import BarData


class FiniteFeed:
    def __init__(self, bars):
        self._bars = list(bars)

    async def stream_klines(self):
        for bar in self._bars:
            yield bar


class DummyReporter:
    def __init__(self):
        self.rendered = []

    def render(self, summary):
        self.rendered.append(summary)


@pytest.mark.asyncio
async def test_run_pipeline_smoke():
    price = 100.0
    bars = []
    for i in range(60):
        close = price + 0.5 * i
        bars.append(
            BarData(
                timestamp=i,
                open=close - 0.2,
                high=close + 0.3,
                low=close - 0.3,
                close=close,
                volume=1.0 + i * 0.05,
            )
        )

    feed = FiniteFeed(bars)
    trader = PaperTrader()
    bandit = ConstraintAwareBandit()
    constraints = ConstraintEvaluator()
    blender = DecisionBlender()
    reporter = DummyReporter()
    risk = RiskManager()

    await run_pipeline(
        feed=feed,
        trader=trader,
        bandit=bandit,
        constraints=constraints,
        blender=blender,
        reporter=reporter,
        risk=risk,
        max_steps=50,
    )

    assert bandit._is_initialized is True
    assert len(constraints.pnl_window) > 0
    assert reporter.rendered, "Raporlayıcı en az bir özet üretmelidir"
