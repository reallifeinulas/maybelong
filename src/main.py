"""Ana asyncio pipeline.

Örnek:
    python -m src.main
"""

from __future__ import annotations

import asyncio
import signal
from collections import deque
from typing import Deque

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.data.feature_engineering import compute_features
from src.data.live_feed import BinanceLiveFeed
from src.evaluation.metrics import compute_summary
from src.evaluation.reporting import LiveReporter
from src.execution.risk import RiskManager, RiskState
from src.execution.simulator import PaperTrader
from src.policy.bandit import ConstraintAwareBandit
from src.policy.constraints import ConstraintEvaluator
from src.signals.decision import BlendInput, DecisionBlender
from src.utils.logging import setup_logger

LOGGER = setup_logger()


async def run_pipeline() -> None:
    """Canlı akışı başlat."""

    settings = get_settings()
    feed = BinanceLiveFeed()
    trader = PaperTrader()
    bandit = ConstraintAwareBandit()
    constraints = ConstraintEvaluator()
    blender = DecisionBlender()
    reporter = LiveReporter()
    risk = RiskManager()

    bars: Deque[dict[str, float]] = deque(maxlen=200)
    pnls: Deque[float] = deque(maxlen=settings.metrics.windows.winrate)
    equity: Deque[float] = deque(maxlen=settings.metrics.windows.mdd)

    async for bar in feed.stream_klines():
        bars.append({
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        })
        if len(bars) < 30:
            continue
        df = pd.DataFrame(list(bars))
        features = compute_features(df).iloc[-1].to_numpy()
        violation_level = 0.0
        action = bandit.select_action(features.astype(np.float64), violation_level=violation_level)
        size = risk.position_size(sharpe=0.5, max_drawdown=0.05)
        pnl = trader.step(bar, action, size)
        pnls.append(pnl)
        equity.append(trader.equity)
        result = constraints.update(pnl, trader.equity)
        bandit.update_feedback(features.astype(np.float64), action, result.reward)
        blend_input = BlendInput(
            model_scores={"LONG": 0.4, "SHORT": 0.3, "FLAT": 0.3},
            rule_bias={"LONG": 0.33, "SHORT": 0.33, "FLAT": 0.34},
            violation_level=result.violation_level,
        )
        decision = blender.blend(blend_input)
        kill_status = risk.kill_switch(
            RiskState(
                equity=trader.equity,
                max_drawdown=max(equity) - min(equity) if equity else 0.0,
                sharpe=0.5,
                roi=0.02,
            )
        )
        LOGGER.info(f"Karar: {decision}, Bandit eylemi: {action}, Kill-switch: {kill_status}\n")
        if len(pnls) > 10:
            summary = compute_summary(pnls, equity)
            reporter.render(summary)


def shutdown(loop: asyncio.AbstractEventLoop) -> None:
    for task in asyncio.all_tasks(loop=loop):
        task.cancel()


def main() -> None:
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: shutdown(loop))
    try:
        loop.run_until_complete(run_pipeline())
    except asyncio.CancelledError:
        LOGGER.info("Kapatma isteği alındı, çıkılıyor...\n")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
