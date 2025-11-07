import numpy as np

from src.policy.bandit import ConstraintAwareBandit


def test_exploration_adjustment():
    bandit = ConstraintAwareBandit()
    features = np.zeros(5)
    action = bandit.select_action(features, violation_level=0.0)
    assert action in {"LONG", "SHORT", "FLAT"}
    bandit.update_feedback(features, action, reward=0.1)
    low_exploration_action = bandit.select_action(features, violation_level=2.0)
    assert low_exploration_action in {"LONG", "SHORT", "FLAT"}
