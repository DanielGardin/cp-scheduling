from typing import Any, ClassVar, Iterable

from textwrap import dedent

from .common import ProcessTimeAllowedTypes
from .tasks import Tasks
from .constraints import Constraint, DisjunctiveConstraint, PrecedenceConstraint, MachineConstraint
from .utils import is_iterable_type, convert_to_list

from abc import ABC, abstractmethod

class ScheduleSetup(ABC):
    tasks: Tasks
    parallel: ClassVar[bool] = True

    def __init__(
        self,
        n_machines: int = -1,
    ):
        """
            Generic class for scheduling setups. Create a machine environment with a specific number
            of machines.
        
        Parameters:
        n_machines (int): Number of machines in the environment.
        """

        self.n_machines = n_machines

    @abstractmethod
    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        """
            Parse the process time of the tasks. The process time can be a list of dictionaries, a
            dictionary of lists, or a pandas DataFrame. The function will return a list of dictionaries
            with the machine as key and the process time as value.
        
        Parameters:
        data (dict): Dictionary containing the data of the tasks.
        process_time (ProcessTimeAllowedTypes): Process time of the tasks.
        
        Returns:
        list[dict[int, int]]: List of dictionaries with the machine as key and the process time as value.
        """
        ...

    def set_tasks(self, tasks: Tasks) -> None:
        self.tasks = tasks

    def setup_constraints(self) -> tuple[Constraint, ...]:
        return ()

    def get_machine(self, task_id: int) -> int:
        raise ValueError(
            f"The {self.__class__.__name__} setup does not have a default machine assignment."
        )

    @abstractmethod
    def export_model(self) -> str:
        ...

    def export_data(self) -> str:
        return ""

    def get_entry(self) -> str:
        return ""


class SingleMachineSetup(ScheduleSetup):
    parallel: ClassVar[bool] = False

    def __init__(self, disjunctive: bool = True) -> None:
        super().__init__(1)
        self.disjunctive = disjunctive
        

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        if is_iterable_type(process_time, int):
            return [{0: p_time} for p_time in process_time]
    
        if isinstance(process_time, str):
            return [{0: p_time} for p_time in data[process_time]]
    
        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def setup_constraints(self) -> tuple[Constraint, ...]:
        if not self.disjunctive:
            return ()

        disjunctive_tasks = {0: list(range(len(self.tasks)))}

        return (DisjunctiveConstraint(disjunctive_tasks, name="disjunctive"),)

    def export_model(self) -> str:
        return dedent("""\"
            % (Schedule Setup) Identical parallel machine processing time
            array[1..num_tasks] of int: processing_time;

            constraint forall(t in 1..num_tasks)(
                sum(p in 1..num_parts)(duration[t,p]) = processing_time[t]
            );
        """)

    def export_data(self) -> str:
        p_times = [task.processing_times[0] for task in self.tasks]
        return f"processing_time = [{', '.join(map(str, p_times))}];\n"

    def get_machine(self, task_id: int) -> int:
        return 0

    def get_entry(self) -> str:
        return "1"


class IdenticalParallelMachineSetup(ScheduleSetup):
    def __init__(
        self,
        n_machines: int,
        disjunctive: bool = True,
    ):
        super().__init__(n_machines)
        self.disjunctive = disjunctive

    def setup_constraints(self) -> tuple[Constraint, ...]:
        return (MachineConstraint(name="machine_disjunctive"), ) if self.disjunctive else ()

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        if is_iterable_type(process_time, int):
            return [
                {machine: p_time for machine in range(self.n_machines)}
                for p_time in process_time
            ]

        if isinstance(process_time, str):
            return [
                {machine: p_time for machine in range(self.n_machines)}
                for p_time in data[process_time]
            ]
    
        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def export_model(self) -> str:
        model = dedent("""\"
            % (Schedule Setup) Identical parallel machine processing time
            int: num_machines;

            array[1..num_tasks] of int: processing_time;
            array[1..num_tasks, 1..num_parts] of var 1..num_machines: assignment;

            constraint forall(t in 1..num_tasks)(
                sum(p in 1..num_parts)(duration[t,p]) = processing_time[t]
            );
        """)
    
        if self.disjunctive:
            model += dedent("""
            constraint cumulatives(
                [start[t, p] | t in 1..num_tasks, p in 1..num_parts],
                [duration[t, p] | t in 1..num_tasks, p in 1..num_parts],
                [1 | t in 1..num_tasks, p in 1..num_parts]
                [assignment[t, p] | t in 1..num_tasks, p in 1..num_parts],
                [1 | m in 1..num_machines]
            ))
            """)
        
        return model

    def export_data(self) -> str:
        return f"num_machines = {self.n_machines};\n" 

    def get_entry(self) -> str:
        return f"P{self.n_machines}" if self.n_machines > 1 else "Pm"

# Untested
class UniformParallelMachineSetup(ScheduleSetup):
    def __init__(
        self,
        n_machines: int,
        speed: Iterable[float],
        disjunctive: bool = True,
    ):
        super().__init__(n_machines)
        self.speed = convert_to_list(speed, int)
        self.disjunctive = disjunctive

    def setup_constraints(self) -> tuple[Constraint, ...]:
        return (MachineConstraint(name="machine_disjunctive"), ) if self.disjunctive else ()

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        self.processing_times: list[int]
        if is_iterable_type(process_time, int):
            self.processing_times = list(process_time)

            return [
                {machine: p_time // self.speed[machine] for machine in range(self.n_machines)}
                for p_time in process_time
            ]

        if isinstance(process_time, str):
            self.processing_times = data[process_time]

            return [
                {machine: p_time // self.speed[machine] for machine in range(self.n_machines)}
                for p_time in data[process_time]
            ]

        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def export_model(self) -> str:
        model = dedent("""\"
            % (Schedule Setup) Different speed parallel machine processing time
            int: num_machines;
            array[1..num_machines] of int: speed;

            array[1..num_tasks] of int: processing_time;
            array[1..num_tasks, 1..num_parts] of var 1..num_machines: assignment;

            constraint forall(t in 1..num_tasks)(
                sum(p in 1..num_parts)(duration[t,p] * speed[assignment[t, p]]) >= processing_time[t]
            );

        """)
    
        if self.disjunctive:
            model += dedent("""
            constraint cumulatives(
                [start[t, p] | t in 1..num_tasks, p in 1..num_parts],
                [duration[t, p] | t in 1..num_tasks, p in 1..num_parts],
                [1 | t in 1..num_tasks, p in 1..num_parts]
                [assignment[t, p] | t in 1..num_tasks, p in 1..num_parts],
                [1 | m in 1..num_machines]
            ))
        """)
        
        return model

    def export_data(self) -> str:
        return f"num_machines = {self.n_machines};\n" \
               f"speed = [{', '.join(map(str, self.speed))}];\n" \
               f"processing_time = [{', '.join(map(str, self.processing_times))}];\n"

    def get_entry(self) -> str:
        return f"U{self.n_machines}" if self.n_machines > 1 else "Um"

class JobShopSetup(ScheduleSetup):
    parallel: ClassVar[bool] = False

    def __init__(
        self,
        n_machines: int = -1,
        operation_order: str = "operation",
        machine_feature: str = "machine",
    ):
        super().__init__(n_machines)

        self.operation_order = operation_order
        self.machine_feature = machine_feature

    def get_machine(self, task_id: int) -> int:
        machine: int = self.tasks.data[self.machine_feature][task_id]
        return machine

    def setup_constraints(self) -> tuple[Constraint, ...]:
        disjunctive_constraint = DisjunctiveConstraint(
            self.machine_feature,
            name="setup_disjunctive"
        )

        edges: list[tuple[int, int]] = []

        operations: list[int] = self.tasks.data[self.operation_order]

        for job_tasks in self.tasks.jobs.values():
            ops = sorted(
                [(operations[task.task_id], task.task_id) for task in job_tasks]
            )

            for i in range(len(ops) - 1):
                edges.append(
                    (ops[i][1], ops[i + 1][1])
                )

        precedence_constraint = PrecedenceConstraint.from_edges(edges, name="setup_precedence")

        return (disjunctive_constraint, precedence_constraint)

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        if is_iterable_type(process_time, int):
            return [
                {machine: p_time}
                for machine, p_time in zip(data[self.machine_feature], process_time)
            ]
    
        if isinstance(process_time, str):
            return [
                {machine: p_time}
                for machine, p_time in zip(data[self.machine_feature], data[process_time])
            ]
    
        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )

    def export_model(self) -> str:
        return dedent("""\
            % (Schedule Setup) Job shop processing time
            array[1..num_tasks] of int: processing_time;
                      
            constraint forall(t in 1..num_tasks)(
                sum(p in 1..num_parts)(duration[t,p]) = processing_time[t]
            );
        """)

    def export_data(self) -> str:
        p_times = [next(iter(task.processing_times.values())) for task in self.tasks]
        return f"processing_time = [{', '.join(map(str, p_times))}];\n"


    def get_entry(self) -> str:
        return f"J{self.n_machines}" if self.n_machines > 1 else "Jm"


class OpenShopSetup(ScheduleSetup):
    parallel: ClassVar[bool] = False

    def __init__(
        self,
        n_machines: int = -1,
        machine_feature: str = "machine",
        disjunctive: bool = True,
    ):
        super().__init__(n_machines)

        self.machine_feature = machine_feature
        self.disjunctive = disjunctive
    
    def get_machine(self, task_id: int) -> int:
        machine: int = self.tasks.data[self.machine_feature][task_id]
        return machine
    
    def setup_constraints(self) -> tuple[Constraint, ...]:
        task_jobs = {
            job: [task.task_id for task in tasks] for job, tasks in self.tasks.jobs.items()
        }
        task_disjunction = DisjunctiveConstraint(task_jobs, name="setup_task_disjunctive")

        if not self.disjunctive:
            return (task_disjunction,)

        machine_disjunction = DisjunctiveConstraint(self.machine_feature, name='setup_machine_disjunctive')

        return (task_disjunction, machine_disjunction)

    def parse_process_time(
        self,
        data: dict[str, list[Any]],
        process_time: ProcessTimeAllowedTypes,
    ) -> list[dict[int, int]]:
        if is_iterable_type(process_time, int):
            return [
                {machine: p_time}
                for machine, p_time in zip(data[self.machine_feature], process_time)
            ]
    
        if isinstance(process_time, str):
            return [
                {machine: p_time}
                for machine, p_time in zip(data[self.machine_feature], data[process_time])
            ]
    
        raise ValueError(
            "Cannot parse the process time. Please provide an iterable of integers or a string."
        )
    
    def export_model(self) -> str:
        return dedent("""\
            % (Schedule Setup) Open shop processing time
            array[1..num_tasks] of int: processing_time;
                      
            constraint forall(t in 1..num_tasks)(
                sum(p in 1..num_parts)(duration[t,p]) = processing_time[t]
            );
        """)
    
    def export_data(self) -> str:
        p_times = [next(iter(task.processing_times.values())) for task in self.tasks]
        return f"processing_time = [{', '.join(map(str, p_times))}];\n"
    
    def get_entry(self) -> str:
        return f"O{self.n_machines}" if self.n_machines > 1 else "Om"
    
