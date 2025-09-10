# 🧩 Constraint Programming Scheduler

This repository provides a comprehensive gym environment for applying constraint programming techniques to scheduling problems.
It focuses on the learning and optimization of priority dispatch rules, enabling the development and testing of intelligent heuristics for dynamic scheduling scenarios.
Designed for researchers and practitioners, the framework supports customization, benchmarking, and integration with machine learning approaches to enhance scheduling performance.

This work was directly inspired by [JobShopCPEnv](https://github.com/ingambe/JobShopCPEnv), which only considers jobshop scheduling problems.
We expand the previous idea of building an environment that automatically ensure the problem's constraints along the decisions.
The major contribution here is providing a flexible and expansible framework for enumerous scheduling problems following the scheduling literature (see [Pinedo, 2022](https://link.springer.com/book/10.1007/978-3-031-05921-6) for a complete review).

## 🚀 Installation

 We use a submodule for storing the scheduling instances in this [standalone repository](https://github.com/DanielGardin/scheduling-instances).
To install the project, clone the repository and install the dependencies, along with the scheduling instances, run

```bash'
git clone --recurse-submodules https://github.com/DanielGardin/cp-scheduling.git
cd cp-scheduling
pip install -e .
```

You can also install the project without the scheduling instances (note that the tests and benchmarks directly look for the instances/ directory) by dropping the `--recurse-submodules` argument.

We separetely distribute the module according to the use case, the installation above only install minimal dependencies for running the environment.
To install the reinforcement learning dependencies, run:

```bash
pip install -e .[rl]
```

To install the solver dependencies with CP and MILP formulations for the scheduling environments, run:

```bash
pip install -e .[solver]
```

## 🗂️ High-Level Structure

The project is organized into the following main directories inside `cpscheduler`:

```text
🧩 cpscheduler
├── 🧠 rl
│   ├── offline
│   │   └── bc
│   │
│   ├── online
│   │   └── reinforce.py
│   │
│   └── policies
│
├── 🏭 environment
│   ├── constraints
│   ├── data
│   ├── env
│   ├── instructions
│   ├── objectives
│   ├── schedule_setup
│   ├── tasks
│   └── utils
│
├── 🏋️ gym
│   └── wrappers
│
├── 🛠️ heuristics
│   └── pdr_heuristics
│
├── 📦 instances
│   ├── customer
│   ├── jobshop
│   ├── rcpsp
│   └── smtwt
│
└── ⚙️ solver
    └── pulp
```

- **`environment`**: This is the core of the project. It provides the `SchedulingEnv` class, which is a flexible and extensible environment for modeling and simulating scheduling problems. It also includes various scheduling setups (e.g., `JobShopSetup`, `SingleMachineSetup`), constraints (e.g., `PrecedenceConstraint`, `ResourceConstraint`), and objectives (e.g., `Makespan`, `TotalCompletionTime`).

- **`instances`**: Provides functions for reading and generating scheduling problem instances, including well-known benchmark sets like Taillard's for Job Shop and RCPSP instances.
- **`heuristics`**: Implements a variety of priority dispatching rules (PDRs) such as `ShortestProcessingTime`, `MostOperationsRemaining`, and `EarliestDueDate`. These can be used as baseline policies or as building blocks for more complex scheduling agents.
- **`solver`**: Integrates with external solvers. Currently, it includes a `pulp` backend for formulating and solving scheduling problems as Mixed-Integer Linear Programs (MILPs). This module depends on the `environment` module.
- **`rl`**: Module for Reinforcement-Learning-based agents, it is split into the modules for
  - **`policies`**: Contains neural network-based policies that can be trained to make scheduling decisions. This includes common architectures like `MLP` and `TransformerEncoder`, as well as the `PlackettLucePolicy` for learning permutation-based actions.
  - **`algorithms`**: Contains implementations of reinforcement learning algorithms, both online (e.g., REINFORCE) and offline (e.g., Behavioral Cloning). This module depends on the `environment`, `policies`, and `heuristics` modules.

## 🛠️ Usage

The following example shows how to use the scheduling environment to solve a simple scheduling problem:

```python
from cpscheduler.environment import SchedulingEnv, JobShopSetup
from cpscheduler.instances import read_jsp_instance
from cpscheduler.heuristics import ShortestProcessingTime

# Load a job shop scheduling instance
instance, metadata = read_jsp_instance("instances/jobshop/ta01.txt")

# Create a scheduling environment
env = SchedulingEnv(JobShopSetup())
env.set_instance(instance)

# Use a priority dispatching rule to solve the scheduling problem
heuristic = ShortestProcessingTime()
obs, info = env.reset()
action = heuristic(obs)
obs, reward, terminated, truncated, info = env.step(action)

# Print the makespan
print(info["current_time"])
```

## 🏗️ Compilation

This project uses `mypyc` to compile the Python code to C extensions for performance automatically via `pip install`.
If the installation fail due to compilation issues, guarantee a C compiler is available and python-dev is installed in your machine.
For any issues with mypyc, we refer to the [original project](https://github.com/mypyc/mypyc).
Due to compilation, we can achieve great performance, for example in jobshop instances,

| Instance | Benchmark |  Simulation time  |   Time per task   |      Speedup      |
| :------: | :-------: | :---------------: | :---------------: | :---------------: |
|  dmu10   |   0.40 s  |  19.39 ±  6.21 ms |  48.49 ± 15.53 µs | 2015.13 ± 166.75% |
|  dmu20   |   0.80 s  |  42.78 ±  6.25 ms |  71.30 ± 10.41 µs | 1787.09 ± 117.52% |
|  dmu30   |   1.40 s  |  76.14 ±  8.59 ms |  95.17 ± 10.74 µs | 1752.23 ± 119.29% |
|  dmu40   |   2.10 s  | 123.11 ± 12.15 ms | 123.11 ± 12.15 µs | 1617.29 ± 116.98% |
|  dmu50   |   0.40 s  |  19.13 ±  0.34 ms |  47.84 ±  0.84 µs |  1991.10 ± 35.77% |
|  dmu60   |   0.80 s  |  41.98 ±  0.55 ms |  69.97 ±  0.92 µs |  1805.86 ± 24.99% |
|  dmu70   |   1.60 s  |  79.04 ± 10.45 ms |  98.80 ± 13.06 µs | 1945.14 ± 157.56% |
|  dmu80   |   2.20 s  | 124.68 ± 10.89 ms | 124.68 ± 10.89 µs | 1673.82 ± 106.65% |
|   la10   |   0.05 s  | 946.39 ± 43.46 µs |  12.62 ±  0.58 µs | 5191.85 ± 189.83% |
|   la20   |   0.05 s  |   1.53 ±  0.04 ms |  15.32 ±  0.36 µs |  3164.42 ± 74.45% |
|   la30   |   0.18 s  |   4.57 ±  0.08 ms |  22.83 ±  0.40 µs |  3844.09 ± 69.67% |
|   la40   |   0.18 s  |   6.57 ±  6.24 ms |  29.20 ± 27.72 µs | 2902.35 ± 282.67% |
|  orb10   |   0.05 s  |   1.49 ±  0.06 ms |  14.94 ±  0.57 µs | 3250.84 ± 111.38% |
|  swv10   |   0.30 s  |  11.23 ±  6.36 ms |  37.42 ± 21.19 µs | 2711.50 ± 255.97% |
|  swv20   |   1.00 s  |  27.87 ±  0.52 ms |  55.74 ±  1.04 µs |  3489.10 ± 63.76% |
|   ta10   |   0.16 s  |   5.88 ±  0.23 ms |  26.15 ±  1.01 µs |  2622.66 ± 97.15% |
|   ta20   |   0.30 s  |  10.25 ±  0.16 ms |  34.18 ±  0.52 µs |  2826.26 ± 44.91% |
|   ta30   |   0.40 s  |  19.67 ±  6.37 ms |  49.17 ± 15.92 µs | 1986.39 ± 165.19% |
|   ta40   |   0.60 s  |  23.53 ±  0.57 ms |  52.28 ±  1.26 µs |  2451.75 ± 58.84% |
|   ta50   |   0.80 s  |  42.63 ±  5.83 ms |  71.05 ±  9.71 µs | 1792.01 ± 113.45% |
|   ta60   |   1.60 s  |  66.30 ±  8.78 ms |  88.41 ± 11.71 µs | 2335.70 ± 170.94% |
|   ta70   |   2.00 s  | 123.33 ± 12.17 ms | 123.33 ± 12.17 µs | 1532.66 ± 111.71% |
|   ta80   |   7.80 s  | 515.57 ± 13.22 ms | 257.79 ±  6.61 µs |  1413.81 ± 36.05% |

average and standard variation between 100 runs per instance.\
*benchmarked with available JSP environments, estimated time.
