"""Ana asyncio pipeline.

Örnek:
    python -m src.main
"""

from __future__ import annotations

import asyncio
import signal
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

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


@dataclass
class _SymbolContext:
    bars: Deque[dict[str, float]]
    pnls: Deque[float]
    equity: Deque[float]
    trader: PaperTrader
    bandit: ConstraintAwareBandit
    constraints: ConstraintEvaluator
    blender: DecisionBlender
    reporter: LiveReporter
    risk: RiskManager
    position_size: float
    violation_level: float = 0.0
    steps: int = 0


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
    target_symbols = list(settings.runtime.symbols or [settings.runtime.symbol])

    def _resolve_dependency(symbol: str, provided, factory):
        if provided is None:
            return factory()
        if isinstance(provided, dict):
            if symbol not in provided:
                raise KeyError(f"{symbol} için bağımlılık bulunamadı")
            return provided[symbol]
        return provided

    contexts: Dict[str, _SymbolContext] = {}

    def _ensure_context(symbol: str) -> _SymbolContext:
        if symbol not in contexts:
            contexts[symbol] = _SymbolContext(
                bars=deque(maxlen=200),
                pnls=deque(maxlen=settings.metrics.windows.winrate),
                equity=deque(maxlen=settings.metrics.windows.mdd),
                trader=_resolve_dependency(symbol, trader, PaperTrader),
                bandit=_resolve_dependency(symbol, bandit, ConstraintAwareBandit),
                constraints=_resolve_dependency(symbol, constraints, ConstraintEvaluator),
                blender=_resolve_dependency(symbol, blender, DecisionBlender),
                reporter=_resolve_dependency(symbol, reporter, LiveReporter),
                risk=_resolve_dependency(symbol, risk, RiskManager),
                position_size=0.0,
            )
            contexts[symbol].position_size = contexts[symbol].risk.position_size(
                sharpe=0.0, max_drawdown=0.0
            )
        return contexts[symbol]

    for sym in target_symbols:
        _ensure_context(sym)

    processed_total = 0

    async for bar in feed.stream_klines():
        symbol = bar.symbol or target_symbols[0]
        if symbol not in contexts:
            target_symbols.append(symbol)
            _ensure_context(symbol)
        ctx = contexts[symbol]

        ctx.bars.append({
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        })
        ctx.steps += 1
        processed_total += 1

        if len(ctx.bars) < 30:
            if max_steps is not None and processed_total >= max_steps:
                break
            continue
        df = pd.DataFrame(list(ctx.bars))
        features = compute_features(df).iloc[-1].to_numpy()
        action = ctx.bandit.select_action(
            features.astype(np.float64), violation_level=ctx.violation_level
        )
        pnl = ctx.trader.step(bar, action, ctx.position_size)
        ctx.pnls.append(pnl)
        ctx.equity.append(ctx.trader.equity)

        sharpe_estimate = 0.0
        if len(ctx.pnls) > 1:
            pnl_array = np.array(ctx.pnls, dtype=float)
            std = pnl_array.std(ddof=1)
            if std > 0:
                sharpe_estimate = float(np.sqrt(252) * pnl_array.mean() / std)

        if ctx.equity:
            equity_list = list(ctx.equity)
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

        result = ctx.constraints.update(pnl, ctx.trader.equity)
        ctx.bandit.update_feedback(features.astype(np.float64), action, result.reward)
        blend_input = BlendInput(
            model_scores={"LONG": 0.4, "SHORT": 0.3, "FLAT": 0.3},
            rule_bias={"LONG": 0.33, "SHORT": 0.33, "FLAT": 0.34},
            violation_level=result.violation_level,
        )
        decision = ctx.blender.blend(blend_input)
        kill_status = ctx.risk.kill_switch(
            RiskState(
                equity=ctx.trader.equity,
                max_drawdown=max_drawdown,
                sharpe=sharpe_estimate,
                roi=roi_value,
            )
        )
        LOGGER.info(
            f"[{symbol}] Karar: {decision}, Bandit eylemi: {action}, Kill-switch: {kill_status}, "
            f"Sharpe≈{sharpe_estimate:.2f}, MDD≈{max_drawdown:.2%}, ROI≈{roi_value:.2%}\n"
        )
        ctx.position_size = ctx.risk.position_size(
            sharpe=sharpe_estimate, max_drawdown=max_drawdown
        )
        ctx.violation_level = result.violation_level
        if len(ctx.pnls) > 10:
            summary = compute_summary(ctx.pnls, ctx.equity)
            ctx.reporter.render(summary, symbol=symbol)

        if max_steps is not None and processed_total >= max_steps:
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
