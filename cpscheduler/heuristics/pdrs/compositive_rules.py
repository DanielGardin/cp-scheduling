from collections.abc import Iterable

from cpscheduler.environment.state import ObsType

from cpscheduler.heuristics.pdrs.base import PriorityDispatchingRule

class CombinedRule(PriorityDispatchingRule):
    """
    Combined Rule heuristic.

    This heuristic combines multiple dispatching rules to select the next job to
    be scheduled by weighting them
    """

    def __init__(
        self,
        rules: Iterable[PriorityDispatchingRule],
        weights: Iterable[float] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed)

        rules = list(rules)
        weights = list(weights) if weights is not None else [1.0] * len(rules)

        if len(rules) == 0:
            raise ValueError(f"Illegal rule: There are no PDRs to combine.")

        if len(rules) != len(weights):
            raise ValueError(
                f"Expected {len(rules)} weights, got {len(weights)}"
            )

        self.rules = rules
        self.weights = weights

    def priority_score(self, obs: ObsType, time: int | None) -> list[float]:
        scores = self.rules[0].priority_score(obs, time)
        priorities = [self.weights[0] * p for p in scores]

        for rule, w in zip(self.rules[1:], self.weights[1:]):
            for t, p in enumerate(rule.priority_score(obs, time)):
                priorities[t] += w * p

        return priorities
