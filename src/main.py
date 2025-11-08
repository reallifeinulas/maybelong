"""Ana asyncio pipeline.

Örnek:
    python -m src.main
"""

from __future__ import annotations

import asyncio
import signal
from collections import deque
from typing import Any, Deque, Optional

import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.data.feature_engineering import compute_features
from src.data.live_feed import BinanceLiveFeed, HistoricalCSVFeed, HistoricalCSVFeedConfig
from src.evaluation.metrics import compute_summary
from src.evaluation.reporting import LiveReporter
from src.execution.risk import RiskManager, RiskState
from src.execution.simulator import PaperTrader
from src.policy.bandit import ConstraintAwareBandit
from src.policy.constraints import ConstraintEvaluator
from src.signals.decision import BlendInput, DecisionBlender
from src.utils.logging import setup_logger

LOGGER = setup_logger()


async def run_pipeline(
    *,
    feed: Optional[Any] = None,
    trader: Optional[PaperTrader] = None,
    bandit: Optional[ConstraintAwareBandit] = None,
    constraints: Optional[ConstraintEvaluator] = None,
    blender: Optional[DecisionBlender] = None,
    reporter: Optional[LiveReporter] = None,
    risk: Optional[RiskManager] = None,
    max_steps: Optional[int] = None,
) -> None:
    """Canlı akışı başlat."""

    settings = get_settings()
    feed = feed or _build_feed(settings)
    trader = trader or PaperTrader()
    bandit = bandit or ConstraintAwareBandit()
    constraints = constraints or ConstraintEvaluator()
    blender = blender or DecisionBlender()
    reporter = reporter or LiveReporter()
    risk = risk or RiskManager()

    bars: Deque[dict[str, float]] = deque(maxlen=200)
    pnls: Deque[float] = deque(maxlen=settings.metrics.windows.winrate)
    equity: Deque[float] = deque(maxlen=settings.metrics.windows.mdd)

    step_count = 0
    position_size = risk.position_size(sharpe=0.0, max_drawdown=0.0)
    violation_level = 0.0
    async for bar in feed.stream_klines():
        bars.append({
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        })
        step_count += 1
        if len(bars) < 30:
            if max_steps is not None and step_count >= max_steps:
                break
            continue
        df = pd.DataFrame(list(bars))
        features = compute_features(df).iloc[-1].to_numpy()
        action = bandit.select_action(features.astype(np.float64), violation_level=violation_level)
        pnl = trader.step(bar, action, position_size)
        pnls.append(pnl)
        equity.append(trader.equity)

        sharpe_estimate = 0.0
        if len(pnls) > 1:
            pnl_array = np.array(pnls, dtype=float)
            std = pnl_array.std(ddof=1)
            if std > 0:
                sharpe_estimate = float(np.sqrt(252) * pnl_array.mean() / std)

        if equity:
            equity_list = list(equity)
            peak_equity = max(equity_list)
            current_equity = equity_list[-1]
            if peak_equity > 0:
                max_drawdown = (peak_equity - current_equity) / peak_equity
            else:
                max_drawdown = 0.0
            starting_equity = equity_list[0]
            roi_value = (current_equity / starting_equity - 1) if starting_equity > 0 else 0.0
        else:
            max_drawdown = 0.0
            roi_value = 0.0

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
                max_drawdown=max_drawdown,
                sharpe=sharpe_estimate,
                roi=roi_value,
            )
        )
        LOGGER.info(
            f"Karar: {decision}, Bandit eylemi: {action}, Kill-switch: {kill_status}, "
            f"Sharpe≈{sharpe_estimate:.2f}, MDD≈{max_drawdown:.2%}, ROI≈{roi_value:.2%}\n"
        )
        position_size = risk.position_size(sharpe=sharpe_estimate, max_drawdown=max_drawdown)
        violation_level = result.violation_level
        if len(pnls) > 10:
            summary = compute_summary(pnls, equity)
            reporter.render(summary)

        if max_steps is not None and step_count >= max_steps:
            break


def _build_feed(settings):
    data_cfg = settings.runtime.data_source
    if data_cfg is None:
        return BinanceLiveFeed()

    data_type = data_cfg.type.lower()
    if data_type == "csv":
        if not data_cfg.path:
            raise ValueError("CSV veri kaynağı için path belirtilmelidir.")
        csv_config = HistoricalCSVFeedConfig(
            path=data_cfg.path,
            delay_seconds=float(data_cfg.delay_seconds),
        )
        return HistoricalCSVFeed(csv_config)

    raise ValueError(f"Desteklenmeyen veri kaynağı türü: {data_cfg.type}")


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
