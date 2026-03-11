from itertools import combinations

from cpscheduler.environment import SchedulingEnv
from cpscheduler.environment.constants import TaskID, MachineID

from cpscheduler.solver.formulation import SymmetryBreaking

from cpscheduler.solver.milp.pulp_utils import implication_pulp

from cpscheduler.solver.milp.disjunctive.formulation import (
    DisjunctiveMILPFormulation,
)


class DominanceRuleSymmetryBreaking(
    SymmetryBreaking[DisjunctiveMILPFormulation], register=False
):
    """
    A symmetry breaking constraint that enforces a dominance rule on the schedule.

    This is a helper class that can be used to define specific dominance rules
    by implementing only the `dominance_rule` and `is_appliable` methods.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__(register=True)

    def dominance_rule(
        self,
        formulation: DisjunctiveMILPFormulation,
        env: SchedulingEnv,
        machine_id: MachineID,
        task_i: TaskID,
        task_j: TaskID,
    ) -> bool:
        """
        Check if task_i dominates task_j according to the dominance rule.

        For example, in parallel machine scheduling, if task_i has a shorter
        processing time than task_j, then task_i dominates task_j and should be
        scheduled before task_j.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the "
            "dominance_rule method."
        )

    def apply(
        self, formulation: DisjunctiveMILPFormulation, env: SchedulingEnv
    ) -> None:
        for task_i, task_j in combinations(range(env.state.n_tasks), 2):
            for machine_id in range(env.state.n_machines):
                if self.dominance_rule(
                    formulation, env, machine_id, task_i, task_j
                ):
                    prec_task, succ_task = task_i, task_j

                elif self.dominance_rule(
                    formulation, env, machine_id, task_j, task_i
                ):
                    prec_task, succ_task = task_j, task_i

                else:
                    continue

                antecedent = (
                    formulation.present[prec_task],
                    formulation.present[succ_task],
                    formulation.assignments[prec_task][machine_id],
                    formulation.assignments[succ_task][machine_id],
                )

                implication_pulp(
                    formulation.model,
                    antecedent,
                    consequent=(
                        formulation.end_times[prec_task],
                        "<=",
                        formulation.start_times[succ_task],
                    ),
                )

                implication_pulp(
                    formulation.model,
                    antecedent,
                    consequent=(
                        formulation.get_order(prec_task, succ_task),
                        "==",
                        1,
                    ),
                )


# class SmithRuleSymmetryBreaking(DominanceRuleSymmetryBreaking):
#     """
#     A symmetry breaking constraint that enforces the Smith's rule on the schedule.

#     Smith's rule states that for single machine scheduling with the objective of
#     minimizing the total weighted completion time, if task_i has a smaller
#     ratio of processing time to weight than task_j, then task_i should be
#     scheduled before task_j.

#     This class can be used as a template for other dominance rules by implementing
#     the `dominance_rule` method according to the specific rule.
#     """

#     def dominance_rule(
#         self,
#         formulation: DisjunctiveMILPFormulation,
#         env: SchedulingEnv,
#         machine_id: MachineID,
#         task_i: TaskID,
#         task_j: TaskID,
#     ) -> bool:
#         state = env.state

#         p_i = state.get_remaining_time(task_i, machine_id)
#         p_j = state.get_remaining_time(task_j, machine_id)
