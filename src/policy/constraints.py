"""Ödül ve kısıt yönetimi.

Örnek:
    from src.policy.constraints import ConstraintEvaluator

    evaluator = ConstraintEvaluator()
    result = evaluator.update(pnl=0.01, equity=1.02)
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Deque, Dict

import numpy as np

from src.config.settings import get_settings


@dataclass
class ConstraintResult:
    reward: float
    penalty: float
    lagrange_multipliers: Dict[str, float]
    violation_level: float


class ConstraintEvaluator:
    """Rolling metrikleri günceller ve cezaları hesaplar."""

    def __init__(self) -> None:
        self.settings = get_settings()
        win_window = self.settings.metrics.windows.winrate
        mdd_window = self.settings.metrics.windows.mdd
        self.pnl_window: Deque[float] = collections.deque(maxlen=win_window)
        self.trade_outcomes: Deque[float] = collections.deque(maxlen=win_window)
        self.returns: Deque[float] = collections.deque(maxlen=self.settings.metrics.windows.sharpe)
        self.equity_curve: Deque[float] = collections.deque(maxlen=mdd_window)
        self.alphas = {
            "winrate": self.settings.metrics.penalties.winrate,
            "profit_factor": self.settings.metrics.penalties.profit_factor,
            "sharpe": self.settings.metrics.penalties.sharpe,
            "roi": self.settings.metrics.penalties.roi,
            "mdd": self.settings.metrics.penalties.mdd,
        }

    def update(self, pnl: float, equity: float) -> ConstraintResult:
        """Yeni PnL gözlemini işler."""

        self.pnl_window.append(pnl)
        self.trade_outcomes.append(1.0 if pnl > 0 else 0.0)
        self.returns.append(pnl)
        self.equity_curve.append(equity)

        reward = self._compute_reward(pnl)
        metrics = self._compute_metrics()
        penalty, violation_level = self._compute_penalty(metrics)
        return ConstraintResult(
            reward=reward - penalty,
            penalty=penalty,
            lagrange_multipliers=self.alphas.copy(),
            violation_level=violation_level,
        )

    def _compute_reward(self, pnl: float) -> float:
        vola_window = self.settings.metrics.reward.vola_window
        lambda_ = self.settings.metrics.reward.vola_lambda
        pnl_scale = self.settings.metrics.reward.pnl_scale
        recent = list(self.pnl_window)[-vola_window:]
        if len(recent) < 2:
            vola = 0.0
        else:
            vola = float(np.std(recent, ddof=1))
        return pnl_scale * pnl - lambda_ * vola

    def _compute_metrics(self) -> Dict[str, float]:
        wins = sum(self.trade_outcomes)
        total = len(self.trade_outcomes)
        winrate = wins / total if total else 0.0
        gains = sum(x for x in self.pnl_window if x > 0)
        losses = -sum(x for x in self.pnl_window if x < 0)
        profit_factor = gains / losses if losses > 0 else float("inf")
        sharpe = self._rolling_sharpe(list(self.returns))
        roi = self._rolling_roi(list(self.equity_curve))
        mdd = self._max_drawdown(list(self.equity_curve))
        return {
            "winrate": winrate,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "roi": roi,
            "mdd": mdd,
        }

    def _compute_penalty(self, metrics: Dict[str, float]) -> tuple[float, float]:
        targets = self.settings.metrics.targets
        penalties = self.settings.metrics.penalties
        violation_level = 0.0
        total_penalty = 0.0
        for key, alpha in self.alphas.items():
            target = getattr(targets, key)
            value = metrics[key]
            if key == "mdd":
                violation = max(0.0, value - target)
            else:
                violation = max(0.0, target - value)
            if violation > 0:
                violation_level = max(violation_level, violation / (target + 1e-9))
                self.alphas[key] = min(penalties.alpha_cap, alpha * penalties.increase_factor)
                total_penalty += self.alphas[key] * violation
            else:
                self.alphas[key] = max(penalties.alpha_floor, alpha * penalties.decrease_factor)
        return total_penalty, violation_level

    @staticmethod
    def _rolling_sharpe(returns: list[float]) -> float:
        if len(returns) < 2:
            return 0.0
        arr = np.array(returns)
        mean = arr.mean()
        std = arr.std(ddof=1)
        if std == 0:
            return 0.0
        return math.sqrt(252) * mean / std

    @staticmethod
    def _rolling_roi(equity: list[float]) -> float:
        if len(equity) < 2:
            return 0.0
        start = equity[0]
        if start <= 0:
            return 0.0
        return equity[-1] / start - 1

    @staticmethod
    def _max_drawdown(equity: list[float]) -> float:
        if not equity:
            return 0.0
        peaks = np.maximum.accumulate(equity)
        drawdowns = (np.array(equity) - peaks) / peaks
        return float(abs(drawdowns.min()))
