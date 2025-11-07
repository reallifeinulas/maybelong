"""Risk kontrolleri ve pozisyon boyutlandırma.

Örnek:
    from src.execution.risk import RiskManager, RiskState

    manager = RiskManager()
    status = manager.kill_switch(RiskState())
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config.settings import get_settings


@dataclass
class RiskState:
    equity: float = 1.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    roi: float = 0.0


class RiskManager:
    """Kill-switch ve boyutlandırma mantığı."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def position_size(self, sharpe: float, max_drawdown: float) -> float:
        """Dinamik boyutlandırmayı uygula."""

        cfg = self.settings.sizing
        base = cfg.base
        beta0 = cfg.beta0
        beta_sharpe = cfg.beta_sharpe
        beta_mdd = cfg.beta_mdd
        raw = beta0 + beta_sharpe * (sharpe - 0.5) - beta_mdd * max(0.0, max_drawdown - 0.15)
        scaled = base * max(0.0, min(cfg.kappa_max, 0.5 + raw))
        return scaled

    def kill_switch(self, state: RiskState) -> str:
        """Risk metriklerine göre eylem öner."""

        safety = self.settings.safety
        if state.max_drawdown > safety.drawdown_hard:
            return "FLAT"
        if state.max_drawdown > safety.drawdown_soft:
            return "REDUCE"
        if state.sharpe < safety.sharpe_floor:
            return "DECREASE_MODEL"
        if state.roi < safety.roi_floor:
            return "REDUCE"
        return "NORMAL"
