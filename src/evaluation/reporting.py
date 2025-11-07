"""Canlı metrik raporlama.

Örnek:
    from src.evaluation.metrics import MetricsSummary
    from src.evaluation.reporting import LiveReporter

    reporter = LiveReporter()
    reporter.render(MetricsSummary(0.5, 1.2, 0.6, 0.03, 0.1))
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from src.evaluation.metrics import MetricsSummary


class LiveReporter:
    """Rich tabanlı terminal çıktısı üretir."""

    def __init__(self) -> None:
        self.console = Console()

    def render(self, summary: MetricsSummary) -> None:
        """Tabloyu yazdır."""

        table = Table(title="Performans Özeti")
        table.add_column("Metrik")
        table.add_column("Değer")
        table.add_row("WinRate", f"{summary.winrate:.2%}")
        table.add_row("Profit Factor", f"{summary.profit_factor:.2f}")
        table.add_row("Sharpe", f"{summary.sharpe:.2f}")
        table.add_row("ROI", f"{summary.roi:.2%}")
        table.add_row("MDD", f"{summary.mdd:.2%}")
        self.console.print(table)
