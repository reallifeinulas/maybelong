"""Paper-trade simülatörü.

Örnek:
    from src.execution.simulator import PaperTrader
    from src.utils.types import BarData

    trader = PaperTrader()
    trader.step(BarData(0, 100, 101, 99, 100, 1), "LONG", 1.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config.settings import get_settings
from src.utils.types import BarData, Decision


@dataclass
class Position:
    side: Decision
    entry_price: float
    size: float
    bars_held: int = 0


class PaperTrader:
    """Basit PnL simülatörü."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.position: Optional[Position] = None
        self.equity = 1.0

    def step(self, bar: BarData, decision: Decision, size: float) -> float:
        """Yeni barda pozisyonu güncelle ve PnL döndür."""

        fee = self.settings.runtime.fee_bps / 10000
        slippage = self.settings.runtime.slippage_bps / 10000
        min_hold = self.settings.runtime.min_hold_bars

        if self.position:
            self.position.bars_held += 1
            pnl = self._mark_to_market(bar)
            if self.position.bars_held >= min_hold and decision != self.position.side:
                exit_pnl = pnl - fee - slippage
                self.equity += exit_pnl
                self.position = None
                return exit_pnl
            return pnl

        if decision != "FLAT" and size > 0:
            self.position = Position(side=decision, entry_price=bar.close * (1 + slippage), size=size)
            self.position.bars_held = 0
            self.equity -= fee
        return 0.0

    def _mark_to_market(self, bar: BarData) -> float:
        if not self.position:
            return 0.0
        price_diff = (bar.close - self.position.entry_price)
        if self.position.side == "SHORT":
            price_diff = -price_diff
        return price_diff * self.position.size
