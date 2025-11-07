"""Performans metriklerinin hesaplanması.

Örnek:
    from src.evaluation.metrics import compute_summary

    summary = compute_summary([0.01, -0.002], [1.0, 1.01])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from src.config.settings import get_settings


@dataclass
class MetricsSummary:
    winrate: float
    profit_factor: float
    sharpe: float
    roi: float
    mdd: float


def compute_summary(pnls: Iterable[float], equity_curve: Iterable[float]) -> MetricsSummary:
    """PnL ve equity verilerinden özet çıkar."""

    pnls_list = list(pnls)
    equity_list = list(equity_curve)
    wins = sum(1 for p in pnls_list if p > 0)
    total = len(pnls_list)
    winrate = wins / total if total else 0.0
    gains = sum(p for p in pnls_list if p > 0)
    losses = -sum(p for p in pnls_list if p < 0)
    profit_factor = gains / losses if losses > 0 else float("inf")
    sharpe = _sharpe_ratio(pnls_list)
    roi = _roi(pnls_list)
    mdd = _max_drawdown(equity_list)
    return MetricsSummary(winrate=winrate, profit_factor=profit_factor, sharpe=sharpe, roi=roi, mdd=mdd)


def evaluate_targets(summary: MetricsSummary) -> Dict[str, bool]:
    """Ayar dosyasındaki hedeflere göre durum raporu üret."""

    settings = get_settings()
    targets = settings.metrics.targets
    return {
        "winrate": summary.winrate >= targets.winrate,
        "profit_factor": summary.profit_factor >= targets.profit_factor,
        "sharpe": summary.sharpe >= targets.sharpe,
        "roi": summary.roi >= targets.roi,
        "mdd": summary.mdd <= targets.mdd,
    }


def _sharpe_ratio(pnls: List[float]) -> float:
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return np.sqrt(252) * arr.mean() / std


def _roi(pnls: List[float]) -> float:
    if not pnls:
        return 0.0
    cumulative = np.cumprod([1 + p for p in pnls])[-1]
    return cumulative - 1


def _max_drawdown(equity: List[float]) -> float:
    if not equity:
        return 0.0
    arr = np.array(equity)
    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / peaks
    return float(-drawdowns.min())
