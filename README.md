# CP Scheduler

A modular, zero-dependency*, framework for modeling, simulating, and solving arbitrary scheduling problems, designed for Reinforcement Learning agents.
The simulator implements a parameterized environment, where problems are defined declaratively through composable **setups**, **constraints**, and **objectives** following the Graham classification ($\alpha \mid \beta \mid \gamma$ ) from scheduling theory ([Pinedo, 2022](https://link.springer.com/book/10.1007/978-3-031-05921-6)).

The environment automatically enforces feasibility during simulation via constraint propagation, making it suitable for priority dispatching rules, MILP solvers, and integration with reinforcement learning pipelines, with automatic action masking for
arbitrary problems.

> Inspired by [JobShopCPEnv](https://github.com/ingambe/JobShopCPEnv), which only targets job-shop problems. This project generalises the idea to a wide range of scheduling problem types **without compromising performance**.

---
*The `environment` module is written in pure python (stdlib) and only depends on mypy for type checking and compilation through mypyC.

## Project Structure

```text
cpscheduler/
├── environment/       Core simulation engine (setups, constraints, objectives)
├── gym/               Gymnasium-compatible wrapper and observation/action adapters
├── heuristics/        Priority dispatching rules
├── instances/         Instance readers & generators (JSP, RCPSP, SMTWT)
├── renderers/         Render capabilities used in the environment
└── solver/            Mathematical programming backends (MILP/CP formulations)
```

---

## Environment
The main component of this library is the `SchedulingEnv` environment.
It combines three kinds of components: a **schedule setup** ($\alpha$), a set of **constraints** ($\beta$), and an **objective** ($\gamma$) into a step-based simulation driven by constraint propagation.

One problem is the result of composing a schedule setup with (none or at least one) constraints and an objective.
This problem is then identified as a triple $\alpha \mid \beta \mid \gamma$.

### Schedule Setups $(\alpha)$

A setup defines the machine environment and how processing times are assigned to tasks.

| Class | Notation | Description |
|---|---|---|
| `SingleMachineSetup` | $1$ | One machine, all tasks share a single queue. |
| `IdenticalParallelMachineSetup` | $P_m$ | $m$ identical machines, same processing time on every machine. |
| `UniformParallelMachineSetup` | $Q_m$ | $m$ machines with different speeds. |
| `UnrelatedParallelMachineSetup` | $R_m$ | $m$ machines with task-dependent processing times. |
| `JobShopSetup` | $J_m$ | Each task is pre-assigned to a machine with a fixed operation order per job. |
| `FlexibleJobShopSetup` | $FJ_m$ | Job-shop precedence with machine alternatives per task. |
| `FlowShopSetup` | $F_m$ | Job-shop style precedence with shared machine order across jobs. |
| `OpenShopSetup` | $O_m$ | Each task can run on any machine, no fixed operation order. |

### Constraints $(\beta)$

Constraints are split into **active** constraints (propagated during simulation) and **passive** constraints (compile-time properties set during initialisation).

| Constraint | Notation | Type | Description |
|---|---|---|---|
| `PrecedenceConstraint` | $prec$ | Active | Enforces ordering between tasks. |
| `NoWaitConstraint` | $nwt$ | Active | Enforces zero waiting time across precedence-linked tasks. |
| `ORPrecedenceConstraint` | $or\text{-}prec$ | Active | Enforces disjunctive (OR) predecessor requirements. |
| `NonOverlapConstraint` | — | Active | Prevents overlapping execution within groups. |
| `ResourceConstraint` | — | Active | Limits concurrent resource usage. |
| `NonRenewableResourceConstraint` | — | Active | Tracks cumulative consumption of a depletable resource. |
| `SetupConstraint` | $s_{jk}$ | Active | Adds sequence-dependent setup times between tasks. |
| `MachineBreakdownConstraint` | $brkdwn$ | Active | Models machine unavailability windows. |
| `BatchConstraint` | $batch$ | Active | Couples tasks that must be processed together on machines. |
| `ReleaseDateConstraint` | $r_j$ | Active | Tasks cannot start before a release time. |
| `DeadlineConstraint` | $\bar{d}_j$ | Active | Tasks must finish before a deadline. |
| `HorizonConstraint` | — | Active | Restricts task completion to a finite horizon. |
| `PreemptionConstraint`* | $prmp$ | Passive | Allows tasks to be interrupted and resumed. |
| `OptionalityConstraint` | $opt$ | Passive | Marks tasks as optional (can be left unscheduled). |
| `MachineEligibilityConstraint` | $M_j$ | Passive | Restricts tasks to a subset of machines. |
| `ConstantProcessingTime` | $p_j = p$ | Passive | Forces all tasks to have the same processing time. |

**Preemption is partially covered at the moment, we are working for its full functionality.*

### Objectives $(\gamma)$

| Objective | Notation | Description |
|---|---|---|
| `Makespan` | $C_{\max}$ | Time at which the last task finishes. |
| `TotalCompletionTime` | $\sum C_j$ | Sum of all job completion times. |
| `WeightedCompletionTime` | $\sum w_j C_j$ | Weighted sum of completion times. |
| `DiscountedTotalCompletionTime` | $\sum \rho^{C_j} C_j$ | Discounted completion-time objective. |
| `TotalFlowTime` | $\sum F_j$ | Sum of flow times (completion − release). |
| `MaximumLateness` | $L_{\max}$ | Worst-case lateness w.r.t. due dates. |
| `TotalTardiness` | $\sum T_j$ | Sum of positive lateness values. |
| `WeightedTardiness` | $\sum w_j T_j$ | Weighted sum of tardiness. |
| `TotalEarliness` | $\sum E_j$ | Sum of positive earliness values. |
| `WeightedEarliness` | $\sum w_j E_j$ | Weighted sum of earliness. |
| `TotalTardyJobs` | $\sum U_j$ | Number of late jobs. |
| `WeightedTardyJobs` | $\sum w_j U_j$ | Weighted count of late jobs. |
| `ComposedObjective` | — | Linear combination of any objectives above. |


## Getting Started
The composition happens in the `SchedulingEnv` initialization, which accepts
components of `ScheduleSetup`, `Constraint`, and `Objective`, as listed above.

```
SchedulingEnv(
    machine_setup: ScheduleSetup,
    constraints: Iterable[Constraint],
    objective: Objective | None,
    ...
)
```

**Example:**
```python
from cpscheduler.environment import SchedulingEnv, JobShopSetup, Makespan
from cpscheduler.instances.formats import read_standard_jobshop_instance
from cpscheduler.heuristics.pdrs import ShortestProcessingTime

instance, _ = read_standard_jobshop_instance("instances/jobshop/ta01.txt")

env = SchedulingEnv(JobShopSetup(), objective=Makespan())
env.load_instance(instance)

heuristic = ShortestProcessingTime()
obs, info = env.reset()

terminated = False
while not terminated:
    action = heuristic(obs)
    obs, reward, terminated, truncated, info = env.step(action)

print(f"Makespan: {info['current_time']}")
```
**Output:**
```text
Makespan: 1462
```


### Composing a Custom Problem

Every combination of setups, constraints and objectives can yield a valid
scheduling environment, beyond the more popular configurations.
Each component require a different data to be imputed as an instance, which
can be a dictionary, or dataframe containing the data.

You can check which feature names are required by the environment via
`env.required_features()`

```python
from cpscheduler.environment import (
    SchedulingEnv,
    IdenticalParallelMachineSetup,
    ReleaseDateConstraint,
    WeightedTardiness,
)

# Create an environment for the P2|r_j|Σw_jT_j problem
env = SchedulingEnv(
    machine_setup=IdenticalParallelMachineSetup(n_machines=2),
    constraints=[ReleaseDateConstraint()],
    objective=WeightedTardiness(),
)

print(env.required_features().keys())
```

**Output:**
```text
['job', 'processing_time', 'release_time', 'due_date', 'weight']
```

The feature `'job'` is a reserved name that allows grouping tasks into jobs.
If not provided, each task is treated as a single-job task.

To load an instance, you can use a dictionary with the required features as
keys and lists of values for each task.

```
# An example instance with 4 tasks for this problem
instance = {
    "processing_time": [1, 2, 3, 4],
    "release_time": [0, 1, 2, 1],
    "due_date": [5, 4, 5, 5],
    "weight": [1, 2, 1, 3]
}

env.load_instance(instance)
```


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
> The library also works in interpreted python by setting the environment variable `MYPYC_DISABLE=1`.


The base install has minimal dependencies.
Other submodules require further third-party dependencies that are installed with the following options:

```bash
pip install -e .[gym]      # Gymnasium wrapper, Plotly rendering
pip install -e .[solver]   # Pyomo MILP + MiniZinc CP backends
pip install -e .[all]      # Gym + solver extras
```

## Testing

Install development dependencies and run tests:

```bash
pip install -e .[dev]
pytest
```

Run test coverage:

```bash
MYPYC_DISABLE=1 pytest --cov --cov-report=term-missing --cov-report=xml
```

`MYPYC_DISABLE=1` keeps the environment in interpreted Python mode, which makes
line-level coverage reporting reliable for the `cpscheduler` source files.


## Compilation

The `environment` module is automatically compiled to C extensions via [mypyc](https://github.com/mypyc/mypyc) during `pip install`.
Since the codebase already uses `mypy --strict`, mypyc can compile it with zero source changes, *i.e.*, the same `.py` files work **both interpreted and compiled**.

**Why compiling?** Environment interation is the main bottleneck for Reinforcement Learning, as simulating has a tight CPU-bound logic over Python objects, with heavy attribute access and integer arithmetic, exactly where intepreted code overhead hurts performance.
Every `step()` call triggers instruction parsing, constraint propagation (iterating over all constraints for every event until a fixed point is reached), and state updates.

This matters in practice because policy evaluation often runs thousands of episodes, and RL training can require millions of environment steps.
A highly optimized environment can, thus, can reduce training time significantly.

Our environment, specialized in the intersection of RL with Scheduling, achieves **~3× faster** simulation time when compared with the interpreted counterpart, and over **~20× faster** than existing scheduling environments in the literature.

The table below shows the compiled performance on standard job-shop instances (100 runs each):

| Instance |  Simulation time  | Tasks |  Time per task  |  Events |    Events/sec    |       Speedup       |
| :------: | :---------------: | :---: | :-------------: | :-----: | :--------------: | :-----------------: |
|  dmu05   |   4.16 ± 0.11 ms  |  300  | 13.86 ± 0.35 µs |  3.51K  | 845.01 ± 20.43 K |  (3849.71 ± 93.07)% |
|  dmu10   |   5.03 ± 0.09 ms  |  400  | 12.59 ± 0.22 µs |  3.98K  | 790.97 ± 13.16 K |  (4172.43 ± 69.40)% |
|  dmu15   |   7.73 ± 0.14 ms  |  450  | 17.19 ± 0.31 µs |  7.34K  | 949.00 ± 17.13 K |  (5303.11 ± 95.71)% |
|  dmu20   |  10.24 ± 0.17 ms  |  600  | 17.06 ± 0.28 µs |  9.87K  | 964.43 ± 15.06 K |  (4493.90 ± 70.15)% |
|  dmu25   |  12.73 ± 0.72 ms  |  600  | 21.22 ± 1.21 µs |  12.93K |  1.02 ± 0.05 M   | (4489.42 ± 208.52)% |
|  dmu30   |  16.62 ± 0.23 ms  |  800  | 20.78 ± 0.29 µs |  16.96K |  1.02 ± 0.01 M   |  (5415.62 ± 73.06)% |
|  dmu35   |  18.31 ± 0.44 ms  |  750  | 24.42 ± 0.58 µs |  19.48K |  1.06 ± 0.02 M   | (4807.48 ± 103.49)% |
|  dmu40   |  24.66 ± 1.76 ms  |   1K  | 24.66 ± 1.76 µs |  26.55K |  1.08 ± 0.05 M   | (4881.17 ± 226.39)% |
|  dmu45   |   4.68 ± 0.10 ms  |  300  | 15.59 ± 0.34 µs |  4.80K  |  1.03 ± 0.02 M   |  (3422.79 ± 70.81)% |
|  dmu50   |   6.61 ± 0.12 ms  |  400  | 16.52 ± 0.30 µs |  6.80K  |  1.03 ± 0.02 M   |  (3330.37 ± 57.10)% |
|  dmu55   |   9.46 ± 0.12 ms  |  450  | 21.02 ± 0.27 µs |  10.15K |  1.07 ± 0.01 M   |  (3701.17 ± 46.23)% |
|  dmu60   |  12.73 ± 0.14 ms  |  600  | 21.21 ± 0.24 µs |  14.13K |  1.11 ± 0.01 M   |  (3614.49 ± 39.21)% |
|  dmu65   |  15.70 ± 0.18 ms  |  600  | 26.17 ± 0.29 µs |  18.04K |  1.15 ± 0.01 M   |  (3757.23 ± 41.41)% |
|  dmu70   |  21.54 ± 0.64 ms  |  800  | 26.93 ± 0.80 µs |  24.86K |  1.16 ± 0.03 M   | (3809.56 ± 100.15)% |
|  dmu75   |  22.65 ± 0.46 ms  |  750  | 30.20 ± 0.61 µs |  26.45K |  1.17 ± 0.02 M   |  (3886.95 ± 73.20)% |
|  dmu80   |  30.14 ± 0.41 ms  |   1K  | 30.14 ± 0.41 µs |  36.06K |  1.20 ± 0.02 M   |  (3982.02 ± 51.81)% |
|   la05   | 447.88 ± 30.29 µs |   50  |  8.96 ± 0.61 µs |   288   | 644.94 ± 28.98 K | (2239.37 ± 100.61)% |
|   la10   | 833.66 ± 28.38 µs |   75  | 11.12 ± 0.38 µs |   626   | 751.64 ± 21.98 K |  (2401.40 ± 70.22)% |
|   la15   |   1.34 ± 0.04 ms  |  100  | 13.44 ± 0.37 µs |  1.10K  | 821.99 ± 19.39 K |  (2978.24 ± 70.26)% |
|   la20   | 911.82 ± 37.62 µs |  100  |  9.12 ± 0.38 µs |   577   | 633.63 ± 20.44 K | (3294.44 ± 106.28)% |
|   la25   |   1.65 ± 0.05 ms  |  150  | 10.98 ± 0.30 µs |  1.34K  | 816.22 ± 20.15 K |  (3643.85 ± 89.95)% |
|   la30   |   2.56 ± 0.05 ms  |  200  | 12.81 ± 0.27 µs |  2.37K  | 923.90 ± 18.13 K |  (3904.90 ± 76.62)% |
|   la35   |   4.89 ± 0.08 ms  |  300  | 16.30 ± 0.25 µs |  4.74K  | 968.73 ± 14.67 K |  (4295.48 ± 65.05)% |
|   la40   |   2.47 ± 0.06 ms  |  225  | 10.96 ± 0.25 µs |  1.97K  | 799.55 ± 17.24 K |  (3652.77 ± 78.76)% |
|  orb10   |   1.03 ± 0.04 ms  |  100  | 10.28 ± 0.37 µs |   785   | 764.38 ± 23.18 K |  (2921.21 ± 88.58)% |
|  swv05   |   2.99 ± 0.06 ms  |  200  | 14.93 ± 0.31 µs |  3.01K  |  1.01 ± 0.02 M   |  (3350.15 ± 64.49)% |
|  swv10   |   4.87 ± 0.08 ms  |  300  | 16.23 ± 0.25 µs |  5.15K  |  1.06 ± 0.02 M   |  (3287.19 ± 48.96)% |
|  swv20   |  12.10 ± 0.15 ms  |  500  | 24.19 ± 0.31 µs |  12.98K |  1.07 ± 0.01 M   |  (4547.19 ± 56.78)% |
|   ta10   |   2.58 ± 0.06 ms  |  225  | 11.47 ± 0.25 µs |  2.28K  | 885.03 ± 17.68 K |  (3874.89 ± 77.41)% |
|   ta20   |   3.87 ± 0.07 ms  |  300  | 12.91 ± 0.22 µs |  3.45K  | 890.59 ± 14.75 K |  (3873.27 ± 64.17)% |
|   ta30   |   5.36 ± 0.11 ms  |  400  | 13.40 ± 0.26 µs |  4.83K  | 901.79 ± 16.59 K |  (3919.22 ± 72.10)% |
|   ta40   |   7.49 ± 0.68 ms  |  450  | 16.64 ± 1.52 µs |  7.12K  | 954.90 ± 52.80 K | (4427.68 ± 244.82)% |
|   ta50   |  10.02 ± 0.42 ms  |  600  | 16.70 ± 0.69 µs |  9.69K  | 968.73 ± 35.31 K | (4497.82 ± 163.96)% |
|   ta60   |  17.57 ± 0.21 ms  |  750  | 23.43 ± 0.27 µs |  19.59K |  1.11 ± 0.01 M   |  (5008.58 ± 57.53)% |
|   ta70   |  23.45 ± 0.55 ms  |   1K  | 23.45 ± 0.55 µs |  25.96K |  1.11 ± 0.02 M   | (5119.61 ± 107.05)% |
|   ta80   |  82.54 ± 4.11 ms  |   2K  | 41.27 ± 2.05 µs | 103.84K |  1.26 ± 0.05 M   | (5583.40 ± 205.63)% |

**Speedup is relative to JobShopCPEnv.*
