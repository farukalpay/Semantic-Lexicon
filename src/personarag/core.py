"""EXP3-based persona routing utilities."""

from __future__ import annotations

import collections.abc as cabc
import math
import random
from dataclasses import dataclass


@dataclass
class BrandStyle:
    """Persona definition that captures style and signature."""

    name: str
    system: str
    signature: str = ""


class PersonaPolicyEXP3:
    """Lightweight EXP3 bandit over persona styles."""

    def __init__(self, arms: cabc.Sequence[BrandStyle], gamma: float = 0.07) -> None:
        if not isinstance(arms, cabc.Sequence):
            raise TypeError("arms must be a sequence of BrandStyle")
        if not arms:
            raise ValueError("PersonaPolicyEXP3 requires at least one persona")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        self.arms: list[BrandStyle] = list(arms)
        self.gamma = gamma
        self.weights: list[float] = [1.0 for _ in arms]
        self.last_index: int | None = None

    def _probabilities(self) -> list[float]:
        total = sum(self.weights)
        if total == 0:
            count = len(self.weights)
            return [1.0 / count for _ in self.weights]
        probs = []
        count = len(self.weights)
        for weight in self.weights:
            base = weight / total if total else 0.0
            probs.append((1 - self.gamma) * base + self.gamma / count)
        return probs

    def choose(self, context: str | None = None) -> BrandStyle:  # noqa: ARG002 - context hook
        probs = self._probabilities()
        roll = random.random()
        accum = 0.0
        for idx, prob in enumerate(probs):
            accum += prob
            if roll <= accum:
                self.last_index = idx
                return self.arms[idx]
        self.last_index = len(self.arms) - 1
        return self.arms[-1]

    def update(self, reward: float) -> None:
        if self.last_index is None:
            return
        reward = max(0.0, min(1.0, reward))
        probs = self._probabilities()
        chosen_prob = probs[self.last_index]
        if chosen_prob <= 0.0:
            return
        importance_weighted = reward / chosen_prob
        growth = math.exp(self.gamma * importance_weighted / len(self.arms))
        self.weights[self.last_index] *= growth

    def telemetry(self) -> dict[str, cabc.Sequence[float]]:
        return {
            "weights": list(self.weights),
            "probs": self._probabilities(),
        }

    def inject_feedback(self, rewards: cabc.Iterable[float]) -> None:
        for reward in rewards:
            self.update(reward)
