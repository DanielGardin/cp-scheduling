from cpscheduler.environment.constants import (
    TaskID,
    MachineID,
    GLOBAL_MACHINE_ID
)
from cpscheduler.environment.state.state import ScheduleState

def job_bounds(job: TaskID, state: ScheduleState, origin: str) -> None:
    if not (0 <= job < state.n_jobs):
        raise ValueError(f"{origin}: invalid job {job}")

def task_bounds(task: TaskID, state: ScheduleState, origin: str) -> None:
    if not (0 <= task < state.n_tasks):
        raise ValueError(f"{origin}: invalid task {task}")

def machine_bounds(
    machine: MachineID, state: ScheduleState, origin: str
) -> None:
    if not (0 <= machine < state.n_machines):
        raise ValueError(f"{origin}: invalid machine {machine}")

def validate_machine_id(
    task_id: TaskID,
    machine_id: MachineID,
    state: ScheduleState,
    *,
    allow_global: bool = False,
    origin: str,
) -> None:
    """
    Validate that a machine identifier is well-formed and admissible for a task.

    This check enforces three invariants:
    1. The machine index lies within the valid range [0, n_machines).

    2. The machine is allowed for the given task according to the instance
       (i.e., it appears in the processing time map).

    3. Optionally, a special GLOBAL_MACHINE_ID is accepted when `allow_global=True`.

    Parameters
    ----------
    task_id : TaskID
        Task whose machine assignment is being validated.
    machine_id : MachineID
        Machine identifier to validate.
    state : ScheduleState
        Provides problem dimensions and processing-time feasibility.
    allow_global : bool, optional
        If True, allows `machine_id == GLOBAL_MACHINE_ID` to bypass checks.
    origin : str
        Identifier of the caller, used to contextualize error messages.

    Raises
    ------
    RuntimeError
        If the machine identifier is out of bounds or not feasible for the task.
    """

    if allow_global and machine_id == GLOBAL_MACHINE_ID:
        return

    if machine_id < 0 or machine_id >= state.n_machines:
        raise RuntimeError(
            f"{origin}: invalid machine_id={machine_id} for task {task_id}"
        )

    if machine_id not in state.instance.processing_times[task_id]:
        raise RuntimeError(
            f"{origin}: machine {machine_id} not allowed for task {task_id}"
        )

def validate_domain_bounds(
    task_id: TaskID,
    state: ScheduleState,
    *,
    machine_id: MachineID = GLOBAL_MACHINE_ID,
    origin: str,
) -> None:
    """
    Validate temporal domain consistency for a task across one or more machines.

    This check enforces per-(task, machine) invariants on the current domain:
    1. Start bounds are ordered: start_lb <= start_ub.
    2. End bounds are ordered: end_lb <= end_ub.
    3. Processing-time consistency:
       - start_lb + p <= end_ub
       - end_lb - p <= start_ub

    The validation can be applied:
    - to a single machine (when `machine_id` is specified), or
    - to all currently feasible machines for the task (when `machine_id`
      equals GLOBAL_MACHINE_ID).

    Parameters
    ----------
    task_id : TaskID
        Task whose temporal domains are being checked.
    state : ScheduleState
        Provides access to domain arrays and feasibility information.
    machine_id : MachineID, optional
        Target machine. If GLOBAL_MACHINE_ID, all feasible machines are checked.
    origin : str
        Identifier of the caller, used to contextualize error messages.

    Raises
    ------
    RuntimeError
        If any bound ordering or processing-time consistency invariant is violated.
    """

    domains = state.domains
    row = task_id * state.n_machines

    if machine_id == GLOBAL_MACHINE_ID:
        machines = list(domains.feasible_machines[task_id])

    else:
        validate_machine_id(task_id, machine_id, state, origin=origin)
        machines = [machine_id]

    for m_id in machines:
        idx = row + m_id

        start_lb = domains.start.lbs[idx]
        start_ub = domains.start.ubs[idx]
        end_lb = domains.end.lbs[idx]
        end_ub = domains.end.ubs[idx]
        remaining = domains.remaining_times[idx]

        if start_lb > start_ub:
            raise RuntimeError(f"{origin}: invalid start bounds for task {task_id} on machine {m_id}")

        if end_lb > end_ub:
            raise RuntimeError(f"{origin}: invalid end bounds for task {task_id} on machine {m_id}")

        if start_lb + remaining > end_ub:
            raise RuntimeError(f"{origin}: start_lb + p > end_ub for task {task_id} on machine {m_id}")

        if end_lb - remaining > start_ub:
            raise RuntimeError(f"{origin}: end_lb - p > start_ub for task {task_id} on machine {m_id}")