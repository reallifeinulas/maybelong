import numpy as np
import pytest

from src.evaluation.metrics import compute_summary, evaluate_targets


def test_compute_summary_basic():
    pnls = [0.01, -0.005, 0.02, -0.001]
    equity = list(1.0 + np.cumsum(pnls))
    summary = compute_summary(pnls, equity)
    assert summary.winrate == 0.5
    assert summary.profit_factor > 1
    assert summary.mdd >= 0


def test_evaluate_targets_structure():
    pnls = [0.01] * 10
    equity = list(1.0 + np.cumsum(pnls))
    summary = compute_summary(pnls, equity)
    flags = evaluate_targets(summary)
    assert set(flags.keys()) == {"winrate", "profit_factor", "sharpe", "roi", "mdd"}


def test_compute_summary_roi_from_equity():
    pnls = [0.05, -0.02]
    equity = [1.0, 1.05, 1.03]
    summary = compute_summary(pnls, equity)
    assert summary.roi == pytest.approx(0.03, rel=1e-6)
