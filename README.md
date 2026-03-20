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

| Instance |  Simulation time  | Tasks |  Time per task   |  Events |     Events/sec    |       Speedup        |
| :------: | :---------------: | :---: | :--------------: | :-----: | :---------------: | :------------------: |
|  dmu05   |   6.16 ± 9.96 ms  |  300  | 20.54 ± 33.19 µs |  3.51K  | 817.50 ± 156.84 K | (3724.36 ± 714.52)%  |
|  dmu10   |  7.73 ± 10.93 ms  |  400  | 19.33 ± 27.34 µs |  3.98K  | 726.95 ± 153.84 K | (3834.70 ± 811.54)%  |
|  dmu15   |  13.68 ± 16.10 ms |  450  | 30.39 ± 35.79 µs |  7.34K  | 787.50 ± 225.48 K | (4400.64 ± 1260.02)% |
|  dmu20   |  18.12 ± 17.54 ms |  600  | 30.21 ± 29.24 µs |  9.87K  | 765.70 ± 235.87 K | (3567.88 ± 1099.08)% |
|  dmu25   |  21.02 ± 17.93 ms |  600  | 35.03 ± 29.88 µs |  12.93K | 842.97 ± 270.64 K | (3715.84 ± 1192.98)% |
|  dmu30   |  28.90 ± 20.87 ms |  800  | 36.13 ± 26.09 µs |  16.96K | 781.54 ± 274.30 K | (4147.35 ± 1455.60)% |
|  dmu35   |  32.54 ± 22.26 ms |  750  | 43.39 ± 29.68 µs |  19.48K | 791.96 ± 291.15 K | (3578.37 ± 1315.55)% |
|  dmu40   |  44.29 ± 25.00 ms |   1K  | 44.29 ± 25.00 µs |  26.55K | 757.38 ± 284.42 K | (3422.91 ± 1285.43)% |
|  dmu45   |   6.89 ± 9.65 ms  |  300  | 22.95 ± 32.18 µs |  4.80K  | 938.65 ± 177.77 K | (3131.44 ± 593.05)%  |
|  dmu50   |  11.42 ± 13.56 ms |  400  | 28.54 ± 33.90 µs |  6.80K  | 873.76 ± 238.05 K | (2827.27 ± 770.27)%  |
|  dmu55   |  16.59 ± 16.80 ms |  450  | 36.87 ± 37.32 µs |  10.15K | 876.99 ± 267.33 K | (3022.62 ± 921.38)%  |
|  dmu60   |  22.56 ± 19.28 ms |  600  | 37.60 ± 32.13 µs |  14.13K | 871.33 ± 291.61 K | (2836.61 ± 949.34)%  |
|  dmu65   |  29.39 ± 22.15 ms |  600  | 48.98 ± 36.92 µs |  18.04K | 839.86 ± 309.74 K | (2746.32 ± 1012.85)% |
|  dmu70   |  41.59 ± 25.48 ms |  800  | 51.98 ± 31.85 µs |  24.86K | 778.81 ± 304.35 K | (2568.68 ± 1003.83)% |
|  dmu75   |  40.09 ± 24.00 ms |  750  | 53.45 ± 32.00 µs |  26.45K | 855.82 ± 332.41 K | (2846.92 ± 1105.76)% |
|  dmu80   |  54.02 ± 26.37 ms |   1K  | 54.02 ± 26.37 µs |  36.06K | 824.81 ± 326.99 K | (2744.88 ± 1088.17)% |
|   la05   | 458.59 ± 24.29 µs |   50  |  9.17 ± 0.49 µs  |   288   |  629.38 ± 26.28 K |  (2185.34 ± 91.24)%  |
|   la10   |   1.34 ± 4.61 ms  |   75  | 17.82 ± 61.50 µs |   626   |  708.99 ± 73.70 K | (2265.13 ± 235.47)%  |
|   la15   |   1.40 ± 0.03 ms  |  100  | 14.01 ± 0.31 µs  |  1.10K  |  788.34 ± 16.95 K |  (2856.29 ± 61.42)%  |
|   la20   | 948.87 ± 19.56 µs |  100  |  9.49 ± 0.20 µs  |   577   |  608.34 ± 12.24 K |  (3162.94 ± 63.64)%  |
|   la25   |   2.23 ± 4.95 ms  |  150  | 14.86 ± 33.02 µs |  1.34K  |  768.18 ± 77.97 K | (3429.38 ± 348.07)%  |
|   la30   |   3.95 ± 7.18 ms  |  200  | 19.73 ± 35.91 µs |  2.37K  | 813.85 ± 154.11 K | (3439.79 ± 651.35)%  |
|   la35   |  7.32 ± 10.37 ms  |  300  | 24.40 ± 34.57 µs |  4.74K  | 876.25 ± 165.84 K | (3885.39 ± 735.37)%  |
|   la40   |   3.71 ± 7.29 ms  |  225  | 16.49 ± 32.39 µs |  1.97K  | 723.80 ± 102.85 K | (3306.68 ± 469.88)%  |
|  orb10   |   1.07 ± 0.02 ms  |  100  | 10.70 ± 0.25 µs  |   785   |  734.06 ± 16.34 K |  (2805.32 ± 62.45)%  |
|  swv05   |   4.28 ± 6.76 ms  |  200  | 21.39 ± 33.79 µs |  3.01K  | 898.11 ± 142.26 K | (2980.77 ± 472.14)%  |
|  swv10   |  9.09 ± 13.63 ms  |  300  | 30.30 ± 45.43 µs |  5.15K  | 900.80 ± 228.25 K | (2796.96 ± 708.70)%  |
|  swv20   |  20.73 ± 18.49 ms |  500  | 41.45 ± 36.98 µs |  12.98K | 879.49 ± 287.51 K | (3725.23 ± 1217.79)% |
|   ta10   |   3.72 ± 6.94 ms  |  225  | 16.55 ± 30.83 µs |  2.28K  | 819.16 ± 113.72 K | (3586.51 ± 497.91)%  |
|   ta20   |   6.17 ± 9.95 ms  |  300  | 20.58 ± 33.16 µs |  3.45K  | 800.51 ± 153.66 K | (3481.47 ± 668.28)%  |
|   ta30   |  8.04 ± 10.44 ms  |  400  | 20.10 ± 26.09 µs |  4.83K  | 789.47 ± 152.36 K | (3431.04 ± 662.14)%  |
|   ta40   |  13.59 ± 16.00 ms |  450  | 30.20 ± 35.56 µs |  7.12K  | 767.85 ± 217.84 K | (3560.37 ± 1010.08)% |
|   ta50   |  17.15 ± 16.80 ms |  600  | 28.58 ± 28.00 µs |  9.69K  | 797.09 ± 241.15 K | (3700.91 ± 1119.64)% |
|   ta60   |  31.92 ± 22.42 ms |  750  | 42.56 ± 29.90 µs |  19.59K | 821.35 ± 301.51 K | (3690.32 ± 1354.68)% |
|   ta70   |  42.01 ± 24.40 ms |   1K  | 42.01 ± 24.40 µs |  25.96K | 791.25 ± 303.39 K | (3657.68 ± 1402.46)% |
|   ta80   | 158.38 ± 18.77 ms |   2K  | 79.19 ± 9.39 µs  | 103.84K |  663.40 ± 66.52 K | (2938.77 ± 294.67)%  |

**Speedup is relative to JobShopCPEnv.*
