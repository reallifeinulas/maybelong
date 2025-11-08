import numpy as np
import pytest

from src.policy.bandit import ConstraintAwareBandit
from src.policy.constraints import ConstraintEvaluator


def test_exploration_adjustment():
    bandit = ConstraintAwareBandit()
    features = np.zeros(5)
    action = bandit.select_action(features, violation_level=0.0)
    assert action in {"LONG", "SHORT", "FLAT"}
    bandit.update_feedback(features, action, reward=0.1)
    low_exploration_action = bandit.select_action(features, violation_level=2.0)
    assert low_exploration_action in {"LONG", "SHORT", "FLAT"}


def test_constraint_roi_uses_equity_curve():
    evaluator = ConstraintEvaluator()
    evaluator.update(0.0, equity=1.0)
    evaluator.update(0.02, equity=1.02)
    metrics = evaluator._compute_metrics()  # noqa: SLF001 - test amaçlı erişim
    assert metrics["roi"] == pytest.approx(0.02, rel=1e-6)
