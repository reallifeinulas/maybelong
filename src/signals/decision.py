"""Model ve kural sinyallerini harmanlama.

Örnek:
    from src.signals.decision import DecisionBlender, BlendInput

    blender = DecisionBlender()
    decision = blender.blend(BlendInput({"LONG": 0.6}, {"FLAT": 0.8}, violation_level=0.0))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.types import Decision


@dataclass
class BlendInput:
    model_scores: dict[Decision, float]
    rule_bias: dict[Decision, float]
    violation_level: float


class DecisionBlender:
    """Model çıktısı ile kural tabanlı bias'ı harmanlar."""

    def __init__(self, base_weight: float = 0.5) -> None:
        self.model_weight = base_weight
        self.rule_weight = 1 - base_weight

    def blend(self, data: BlendInput) -> Decision:
        """Ağırlıklı toplamdan karar üret."""

        model_weight = max(0.0, min(1.0, self.model_weight - 0.2 * data.violation_level))
        rule_weight = 1.0 - model_weight
        combined = {}
        for action in {"LONG", "SHORT", "FLAT"}:
            combined[action] = model_weight * data.model_scores.get(action, 0.0) + rule_weight * data.rule_bias.get(action, 0.0)
        probs = np.array(list(combined.values()), dtype=float)
        if probs.sum() <= 0:
            return "FLAT"
        probs = probs / probs.sum()
        choice = np.random.choice(list(combined.keys()), p=probs)
        return str(choice)
