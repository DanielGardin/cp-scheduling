# CP Scheduler

A modular, zero-dependency*, framework for modeling, simulating, and solving scheduling problems using constraint programming.
Problems are defined declaratively through composable **setups**, **constraints**, and **objectives** following the $\alpha \mid \beta \mid \gamma$ classification from scheduling theory ([Pinedo, 2022](https://link.springer.com/book/10.1007/978-3-031-05921-6)).

The environment automatically enforces feasibility during simulation via constraint propagation, making it suitable for priority dispatching rules, MILP solvers, and integration with reinforcement learning pipelines.

> Inspired by [JobShopCPEnv](https://github.com/ingambe/JobShopCPEnv), which only targets job-shop problems. This project generalises the idea to a wide range of scheduling problem types.

*The `environment` codebase is written in pure python (stdlib) and only depends on mypy for type checking and compilation through mypyC.

## Installation

Scheduling instances are stored in a separate [submodule repository](https://github.com/DanielGardin/scheduling-instances).
We use these instances in tests and reports across the library, but it is not necessary to use the environment by its own.

```bash
git clone --recurse-submodules https://github.com/DanielGardin/cp-scheduling.git
cd cp-scheduling
pip install -e .
```

> Drop `--recurse-submodules` if you don't need the benchmark instances (tests and benchmarks expect the `instances/` directory).

> The `environment` module is compiled to C extensions via [mypyc](https://github.com/mypyc/mypyc) automatically during `pip install`.
A C compiler and `python-dev` headers are required. 
> The library also works in interpreted python by setting the environment variable `DISABLE_MYPYC=1`.


The base install has minimal dependencies.
Other submodules require further third-party dependencies that are installed with the following options:

```bash
pip install -e .[gym]      # Gymnasium wrapper, Plotly rendering
pip install -e .[solver]   # PuLP MILP solver backend
pip install -e .[all]      # Everything (includes rl dependencies)
```

## Project Structure

```text
cpscheduler/
├── environment/       Core simulation engine (setups, constraints, objectives)
├── gym/               Gymnasium-compatible wrapper and observation/action adapters
├── heuristics/        Priority dispatching rules
├── instances/         Instance readers & generators (JSP, RCPSP, SMTWT)
└── solver/            MILP solver via PuLP
```

---

## Environment

`SchedulingEnv` is the central class. It combines a **schedule setup** ($\alpha$), a set of **constraints** ($\beta$), and an **objective** ($\gamma$) into a step-based simulation driven by constraint propagation.

### Quick Start

```python
from cpscheduler.environment import SchedulingEnv, JobShopSetup, Makespan
from cpscheduler.instances import read_jsp_instance
from cpscheduler.heuristics import ShortestProcessingTime

instance, _ = read_jsp_instance("instances/jobshop/ta01.txt")

env = SchedulingEnv(JobShopSetup(), objective=Makespan())
env.set_instance(instance)

heuristic = ShortestProcessingTime()
obs, info = env.reset()
action = heuristic(obs)
obs, reward, terminated, truncated, info = env.step(action)

print(f"Makespan: {info['current_time']}")
```

### Schedule Setups ($\alpha$)

A setup defines the machine environment and how processing times are assigned to tasks.

| Class | Notation | Description |
|---|---|---|
| `SingleMachineSetup` | $1$ | One machine, all tasks share a single queue. |
| `IdenticalParallelMachineSetup` | $P_m$ | $m$ identical machines, same processing time on every machine. |
| `UniformParallelMachineSetup` | $Q_m$ | $m$ machines with different speeds. |
| `UnrelatedParallelMachineSetup` | $R_m$ | $m$ machines with task-dependent processing times. |
| `JobShopSetup` | $J_m$ | Each task is pre-assigned to a machine with a fixed operation order per job. |
| `OpenShopSetup` | $O_m$ | Each task can run on any machine; no fixed operation order. |

### Constraints ($\beta$)

Constraints are split into **active** constraints (propagated during simulation) and **passive** constraints (compile-time properties set during initialisation).

| Constraint | Notation | Type | Description |
|---|---|---|---|
| `PrecedenceConstraint` | $prec$ | Active | Enforces ordering between tasks. |
| `NonOverlapConstraint` | — | Active | Prevents overlapping execution within groups. |
| `ResourceConstraint` | — | Active | Limits concurrent resource usage. |
| `NonRenewableResourceConstraint` | — | Active | Tracks cumulative consumption of a depletable resource. |
| `SetupConstraint` | $s_{jk}$ | Active | Adds sequence-dependent setup times between tasks. |
| `MachineBreakdownConstraint` | $brkdwn$ | Active | Models machine unavailability windows. |
| `ReleaseDateConstraint` | $r_j$ | Active | Tasks cannot start before a release time. |
| `DeadlineConstraint` | $\bar{d}_j$ | Active | Tasks must finish before a deadline. |
| `PreemptionConstraint` | $prmp$ | Passive | Allows tasks to be interrupted and resumed. |
| `OptionalityConstraint` | $opt$ | Passive | Marks tasks as optional (can be left unscheduled). |
| `MachineEligibilityConstraint` | $M_j$ | Passive | Restricts tasks to a subset of machines. |
| `ConstantProcessingTime` | $p_j = p$ | Passive | Forces all tasks to have the same processing time. |

### Objectives ($\gamma$)

| Objective | Notation | Description |
|---|---|---|
| `Makespan` | $C_{\max}$ | Time at which the last task finishes. |
| `TotalCompletionTime` | $\sum C_j$ | Sum of all job completion times. |
| `WeightedCompletionTime` | $\sum w_j C_j$ | Weighted sum of completion times. |
| `TotalFlowTime` | $\sum F_j$ | Sum of flow times (completion − release). |
| `MaximumLateness` | $L_{\max}$ | Worst-case lateness w.r.t. due dates. |
| `TotalTardiness` | $\sum T_j$ | Sum of positive lateness values. |
| `WeightedTardiness` | $\sum w_j T_j$ | Weighted sum of tardiness. |
| `TotalEarliness` | $\sum E_j$ | Sum of positive earliness values. |
| `WeightedEarliness` | $\sum w_j E_j$ | Weighted sum of earliness. |
| `TotalTardyJobs` | $\sum U_j$ | Number of late jobs. |
| `WeightedTardyJobs` | $\sum w_j U_j$ | Weighted count of late jobs. |
| `ComposedObjective` | — | Linear combination of any objectives above. |

### Composing a Custom Problem

```python
from cpscheduler.environment import (
    SchedulingEnv,
    IdenticalParallelMachineSetup,
    ReleaseDateConstraint,
    WeightedTardiness,
)

env = SchedulingEnv(
    machine_setup=IdenticalParallelMachineSetup(n_machines=3),
    constraints=[ReleaseDateConstraint()],
    objective=WeightedTardiness(),
)
```

### Observation & Info

`env.get_state()` returns a tuple of **(task_features, job_features)**, each as a dictionary of lists. The `info` dictionary returned by `step()` and `reset()` includes `current_time`, `objective_value`, `event_count`, and any custom metrics added via `env.add_metric()`.

---

## Gymnasium Wrapper

The `gym` module wraps `SchedulingEnv` into a [Gymnasium](https://gymnasium.farama.org/)-compatible interface, enabling interoperability with standard RL tooling.

### Registered Environments

```python
import gymnasium as gym

# Generic scheduling environment
env = gym.make("Scheduling-v0", machine_setup=JobShopSetup(), instance=instance)

# Pre-configured job-shop alias
env = gym.make("Jobshop-v0", instance=instance)
```

### Wrappers

| Wrapper | Description |
|---|---|
| `TabularObservationWrapper` | Merges task and job features into a single flat dict. |
| `CPStateWrapper` | Exposes the underlying constraint-propagation state. |
| `ArrayObservationWrapper` | Converts observations to NumPy arrays. |
| `PermutationActionWrapper` | Accepts a permutation of task/job IDs as the action. |
| `InstancePoolWrapper` | Samples a new instance from a pool on each reset. |

```python
from cpscheduler.gym import SchedulingEnvGym, PermutationActionWrapper, TabularObservationWrapper

gym_env = SchedulingEnvGym(JobShopSetup(), objective=Makespan(), instance=instance)
gym_env = TabularObservationWrapper(gym_env)
gym_env = PermutationActionWrapper(gym_env, strict=True)
```

---

## 🛠️ Heuristics

Built-in priority dispatching rules that work directly with `SchedulingEnv` observations.

| Heuristic | Description |
|---|---|
| `ShortestProcessingTime` | Prioritises tasks with the smallest processing time. |
| `WeightedShortestProcessingTime` | Ratio of weight to processing time. |
| `EarliestDueDate` | Prioritises the nearest due date. |
| `ModifiedDueDate` | Adjusts due dates based on remaining work. |
| `MinimumSlackTime` | Smallest slack (due date − processing time − current time). |
| `FirstInFirstOut` | Arrival order. |
| `CriticalRatio` | Ratio of remaining time to remaining processing time. |
| `CostOverTime` | Penalty-rate heuristic. |
| `ApparentTardinessCost` | ATC composite rule combining urgency and slack. |
| `TrafficPriority` | Based on downstream machine congestion. |
| `MostOperationsRemaining` | Most remaining operations in the job. |
| `MostWorkRemaining` | Most remaining processing time in the job. |
| `RandomPriority` | Random ordering. |

```python
from cpscheduler.heuristics import ShortestProcessingTime

heuristic = ShortestProcessingTime(strict=True)
action = heuristic(obs)
```

---

## ⚙️ Solver

The `solver` module automatically formulates the scheduling problem as a Mixed-Integer Linear Program using [PuLP](https://coin-or.github.io/pulp/).

```python
from cpscheduler.solver import PulpSolver

solver = PulpSolver(env, formulation="scheduling", symmetry_breaking=True)
solver.build()
actions, objective_value, status = solver.solve("GUROBI_CMD", timeLimit=60)
```

**Key features:**

- **Two formulations**: `"scheduling"` (precedence-based) and `"timetable"` (time-indexed).
- **Warm starting**: Feed an initial heuristic solution to speed up the solver.
- **Symmetry breaking**: Automatically adds cuts to reduce the search space.
- **Solver-agnostic**: Any PuLP-supported backend (Gurobi, CPLEX, CBC, GLPK, …) via `solver.available_solvers()`.

---

## Compilation

The `environment` module is automatically compiled to C extensions via [mypyc](https://github.com/mypyc/mypyc) during `pip install`.
Since the codebase already uses `mypy --strict`, mypyc can compile it with zero source changes, *i.e.*, the same `.py` files work **both interpreted and compiled**.

**Why compiling?** Environment interation is the main bottleneck for Reinforcement Learning, as simulating has a tight CPU-bound logic over Python objects, with heavy attribute access and integer arithmetic, exactly where intepreted code overhead hurts performance.
Every `step()` call triggers instruction parsing, constraint propagation (iterating over all constraints for every event until a fixed point is reached), and state updates.

This matters in practice because policy evaluation often runs thousands of episodes, and RL training can require millions of environment steps.
A highly optimized environment can, thus, can reduce training time significantly.

Our environment, specialized in the intersection of RL with Scheduling, achieves **~3× faster** simulation time when compared with the interpreted counterpart, and over **~20× faster** than existing scheduling environments in the literature.

The table below shows the compiled performance on standard job-shop instances (100 runs each):

| Instance |  Simulation time  | Tasks |  Time per task   |  Events |    Events/sec    |       Speedup      |
| :------: | :---------------: | :---: | :--------------: | :-----: | :--------------: | :-----------------: |
|  dmu10   |  10.71 ± 4.15 ms  |  400  | 26.76 ± 10.38 µs |  3.98K  | 384.07 ± 32.97 K | (3859.02 ± 331.29)% |
|  dmu20   |  24.73 ± 4.18 ms  |  600  | 41.22 ± 6.97 µs  |  9.87K  | 403.65 ± 27.56 K | (3271.03 ± 223.32)% |
|  dmu30   |  46.58 ± 4.94 ms  |  800  | 58.23 ± 6.18 µs  |  16.96K | 366.41 ± 22.80 K | (3024.62 ± 188.24)% |
|  dmu40   |  73.72 ± 4.46 ms  |   1K  | 73.72 ± 4.46 µs  |  26.55K | 361.04 ± 14.31 K | (2855.50 ± 113.19)% |
|  dmu50   |  12.38 ± 0.22 ms  |  400  | 30.95 ± 0.55 µs  |  6.80K  | 549.29 ± 9.36 K  |  (3231.60 ± 55.06)% |
|  dmu60   |  28.71 ± 4.25 ms  |  600  | 47.84 ± 7.09 µs  |  14.13K | 496.66 ± 30.47 K | (2811.94 ± 172.49)% |
|  dmu70   |  53.77 ± 4.32 ms  |  800  | 67.22 ± 5.39 µs  |  24.86K | 464.03 ± 21.47 K | (2986.28 ± 138.16)% |
|  dmu80   |  80.53 ± 4.99 ms  |   1K  | 80.53 ± 4.99 µs  |  36.06K | 448.98 ± 19.62 K | (2739.27 ± 119.72)% |
|   la10   | 848.65 ± 19.89 µs |   75  | 11.32 ± 0.27 µs  |   626   | 738.02 ± 16.53 K | (5894.75 ± 132.06)% |
|   la20   |   1.06 ± 0.02 ms  |  100  | 10.59 ± 0.15 µs  |   577   | 544.98 ± 7.94 K  |  (4722.50 ± 68.82)% |
|   la30   |   3.45 ± 0.05 ms  |  200  | 17.25 ± 0.25 µs  |  2.37K  | 685.89 ± 10.02 K |  (5218.06 ± 76.24)% |
|   la40   |   3.61 ± 0.05 ms  |  225  | 16.04 ± 0.24 µs  |  1.97K  | 546.14 ± 8.16 K  |  (4990.07 ± 74.57)% |
|  orb10   |   1.19 ± 0.05 ms  |  100  | 11.94 ± 0.50 µs  |   785   | 658.14 ± 22.32 K | (4192.00 ± 142.14)% |
|  swv10   |   7.61 ± 0.11 ms  |  300  | 25.38 ± 0.38 µs  |  5.15K  | 676.85 ± 9.88 K  |  (3940.51 ± 57.52)% |
|  swv20   |  22.18 ± 4.21 ms  |  500  | 44.37 ± 8.43 µs  |  12.98K | 592.87 ± 40.67 K | (4565.82 ± 313.20)% |
|   ta10   |   3.86 ± 0.05 ms  |  225  | 17.13 ± 0.24 µs  |  2.28K  | 592.56 ± 8.32 K  |  (4151.02 ± 58.29)% |
|   ta20   |   6.37 ± 0.14 ms  |  300  | 21.22 ± 0.46 µs  |  3.45K  | 541.96 ± 11.17 K |  (4714.03 ± 97.15)% |
|   ta30   |  11.32 ± 4.18 ms  |  400  | 28.30 ± 10.45 µs |  4.83K  | 439.71 ± 35.52 K | (3640.02 ± 294.02)% |
|   ta40   |  14.11 ± 0.46 ms  |  450  | 31.35 ± 1.01 µs  |  7.12K  | 504.91 ± 14.28 K | (4256.69 ± 120.36)% |
|   ta50   |  23.57 ± 4.27 ms  |  600  | 39.29 ± 7.11 µs  |  9.69K  | 416.01 ± 27.44 K | (3433.85 ± 226.51)% |
|   ta60   |  43.75 ± 4.33 ms  |  750  | 58.33 ± 5.77 µs  |  19.59K | 449.97 ± 23.15 K | (3675.84 ± 189.13)% |
|   ta70   |  67.95 ± 4.70 ms  |   1K  | 67.95 ± 4.70 µs  |  25.96K | 383.16 ± 16.13 K | (2952.02 ± 124.28)% |
|   ta80   |  288.98 ± 9.91 ms |   2K  | 144.49 ± 4.96 µs | 103.84K | 359.71 ± 11.02 K |  (2701.96 ± 82.76)% |

*Speedup is relative to existing frameworks.*
