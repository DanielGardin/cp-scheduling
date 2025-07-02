from collections.abc import Callable

from pulp import LpProblem, LpAffineExpression, lpSum

from .tasks import PulpVariables, PulpSchedulingVariables, PulpTimetable
from .pulp_utils import indicator_constraint

from cpscheduler.environment.env   import SchedulingEnv
from cpscheduler.environment.tasks import Tasks

import cpscheduler.environment.constraints    as constraints
import cpscheduler.environment.objectives     as objectives
import cpscheduler.environment.schedule_setup as setup

ParallelSetups = (
    setup.SingleMachineSetup,
    setup.IdenticalParallelMachineSetup,
    setup.UniformParallelMachineSetup,
    setup.UnrelatedParallelMachineSetup,
)

PackingObjectives = (
    objectives.Makespan,
    objectives.TotalCompletionTime,

)

def employ_symmetry_breaking_pulp(
    env: SchedulingEnv,
    variables: PulpVariables,
) -> Callable[[LpProblem], None]:

    def export_model(model: LpProblem) -> None:
        if (
            isinstance(variables, PulpSchedulingVariables) and
            isinstance(env.setup, ParallelSetups) and
            isinstance(env.objective, PackingObjectives) and
            env.setup.disjunctive and
            all(constraint_name.startswith('setup') for constraint_name in env.constraints)
        ):
            machine_ordering_symmetry_breaking(model, variables, env.tasks)
            job_ordering_symmetry_breaking(model, variables, env.tasks)

    return export_model

def job_ordering_symmetry_breaking(
    model: LpProblem,
    decision_vars: PulpSchedulingVariables,
    tasks: Tasks
) -> None:
    "When jobs inside machines are exchangeable, break symmetry by ordering by lexicographic order"
    n_tasks = len(decision_vars.tasks)

    for j in range(n_tasks):
        for i in range(j):
            for machine_id in range(tasks.n_machines):
                indicator_constraint(
                    model,
                    lhs        = decision_vars.end_times[i],
                    operator   = "<=",
                    rhs        = decision_vars.start_times[j],
                    indicators = (
                        decision_vars.assignments[i][machine_id],
                        decision_vars.assignments[j][machine_id],
                    ),
                    big_m      = tasks[i].get_end_ub() - tasks[j].get_start_lb(),
                    name       = f"SB_job_{i}_{j}_machine_{machine_id}"
                )

def machine_ordering_symmetry_breaking(
    model: LpProblem,
    decision_vars: PulpSchedulingVariables,
    tasks: Tasks
) -> None:
    "When machines are exchangeable, break symmetry by ordering by load"
    n_machines = tasks.n_machines

    processing_times: list[LpAffineExpression] = [
        lpSum([
            task.processing_times[machine_id] * decision_vars.assignments[task_id][machine_id]
            for task_id, task in enumerate(tasks) if machine_id in task.processing_times
        ]) for machine_id in range(n_machines)
    ]

    for machine_id in range(n_machines-1):
        model.addConstraint(
            processing_times[machine_id] >= processing_times[machine_id + 1],
            name=f"SB_machine_{machine_id}_order"
        )
