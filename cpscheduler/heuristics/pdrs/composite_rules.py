"""Composite rules for priority dispatching heuristics."""

from collections.abc import Iterable

from typing_extensions import override

from cpscheduler.environment.observation import DefaultObservation
from cpscheduler.heuristics.pdrs.base import PriorityDispatchingRule


class CombinedRule(PriorityDispatchingRule):
    """
    Combined Rule heuristic.

    This heuristic combines multiple dispatching rules to select the next job to
    be scheduled by weighting them and summing their priority scores.
    """

    rules: list[PriorityDispatchingRule]
    weights: list[float]

    def __init__(
        self,
        rules: Iterable[PriorityDispatchingRule],
        weights: Iterable[float] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the Combined Rule heuristic.

        Parameters
        ----------
        rules : Iterable[PriorityDispatchingRule]
            Iterable of dispatching rules to combine.

        weights : Iterable[float] or None, optional
            Weights for each dispatching rule. If None, all rules are weighted
            equally. Default is None.

        seed : int or None, optional
            Random seed for reproducibility. Default is None.

        """
        super().__init__(seed)

        rules = list(rules)
        weights = list(weights) if weights is not None else [1.0] * len(rules)

        if len(rules) == 0:
            raise ValueError("Illegal rule: There are no PDRs to combine.")

        if len(rules) != len(weights):
            raise ValueError(
                f"Expected {len(rules)} weights, got {len(weights)}"
            )

        self.rules = rules
        self.weights = weights

    @override
    def priority_score(self, obs: DefaultObservation) -> list[float]:
        scores = self.rules[0].priority_score(obs)
        priorities = [self.weights[0] * p for p in scores]

        for rule, w in zip(self.rules[1:], self.weights[1:], strict=False):
            for t, p in enumerate(rule.priority_score(obs)):
                priorities[t] += w * p

        return priorities
