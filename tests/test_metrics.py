import numpy as np

from src.evaluation.metrics import compute_summary, evaluate_targets


def test_compute_summary_basic():
    pnls = [0.01, -0.005, 0.02, -0.001]
    equity = list(np.cumsum(pnls))
    summary = compute_summary(pnls, equity)
    assert summary.winrate == 0.5
    assert summary.profit_factor > 1
    assert summary.mdd >= 0


def test_evaluate_targets_structure():
    pnls = [0.01] * 10
    equity = list(np.cumsum(pnls))
    summary = compute_summary(pnls, equity)
    flags = evaluate_targets(summary)
    assert set(flags.keys()) == {"winrate", "profit_factor", "sharpe", "roi", "mdd"}
