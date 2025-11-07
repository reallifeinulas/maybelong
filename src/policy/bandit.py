"""Kısıt farkındalıklı LinUCB/SGD politikası."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from src.config.settings import get_settings
from src.utils.types import Decision

ACTIONS = np.array(["LONG", "SHORT", "FLAT"], dtype=str)


class ConstraintAwareBandit:
    """Basit bir SGD tabanlı sınıflandırıcı ile eylem seçer."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = SGDClassifier(loss="log_loss")
        self._is_initialized = False
        self._exploration = self.settings.bandit.base_exploration

    def update_feedback(self, features: np.ndarray, action: Decision, reward: float) -> None:
        """Gözleme göre modeli güncelle."""

        y = np.where(ACTIONS == action)[0]
        if y.size == 0:
            raise ValueError(f"Bilinmeyen eylem: {action}")
        y_idx = y[0]
        if not self._is_initialized:
            self.model.partial_fit(features.reshape(1, -1), np.array([y_idx]), classes=np.arange(len(ACTIONS)))
            self._is_initialized = True
        else:
            self.model.partial_fit(features.reshape(1, -1), np.array([y_idx]))

    def select_action(self, features: np.ndarray, violation_level: float = 0.0) -> Decision:
        """Özelliklerden eylem seç."""

        exploration = self._adjust_exploration(violation_level)
        if not self._is_initialized or np.random.rand() < exploration:
            idx = np.random.randint(len(ACTIONS))
            return str(ACTIONS[idx])  # type: ignore[return-value]
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        return str(ACTIONS[idx])  # type: ignore[return-value]

    def _adjust_exploration(self, violation_level: float) -> float:
        """Kısıt ihlali şiddetine göre keşif oranını güncelle."""

        if violation_level >= 1.0:
            self._exploration = max(self.settings.bandit.min_exploration, self.settings.bandit.severe_penalty)
        elif violation_level > 0.0:
            self._exploration = max(self.settings.bandit.min_exploration, self.settings.bandit.mild_penalty)
        else:
            self._exploration = min(
                self.settings.bandit.max_exploration,
                self._exploration + self.settings.bandit.recovery_rate,
            )
        return self._exploration
