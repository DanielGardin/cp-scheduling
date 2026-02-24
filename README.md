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

| Instance | Benchmark* |  Simulation time   |   Time per task   |  Events | Events/sec |      Speedup      |
| :------: | :--------: | :----------------: | :---------------: | :-----: | :--------: | :---------------: |
|  dmu10   |   0.40 s   |  12.69 ±  5.94 ms  |  31.72 ± 14.84 µs |  3.98K  |  327.40K   | 3189.58 ± 322.45% |
|  dmu20   |   0.80 s   |  30.07 ±  5.83 ms  |  50.12 ±  9.72 µs |  9.87K  |  333.75K   | 2604.65 ± 246.62% |
|  dmu30   |   1.40 s   |  57.42 ±  6.60 ms  |  71.77 ±  8.25 µs |  16.96K |  297.66K   | 2357.13 ± 170.47% |
|  dmu40   |   2.10 s   |  93.37 ±  7.05 ms  |  93.37 ±  7.05 µs |  26.55K |  285.54K   | 2158.35 ± 124.84% |
|  dmu50   |   0.40 s   |  14.68 ±  0.73 ms  |  36.70 ±  1.81 µs |  6.80K  |  464.18K   | 2630.85 ± 126.11% |
|  dmu60   |   0.80 s   |  34.54 ±  5.94 ms  |  57.57 ±  9.90 µs |  14.13K |  414.09K   | 2244.47 ± 172.89% |
|  dmu70   |   1.60 s   |  66.54 ±  6.38 ms  |  83.18 ±  7.98 µs |  24.86K |  375.73K   | 2318.03 ± 147.19% |
|  dmu80   |   2.20 s   | 101.11 ±  8.30 ms  | 101.11 ±  8.30 µs |  36.06K |  358.45K   | 2086.95 ± 138.32% |
|   la10   |   0.05 s   | 964.29 ± 134.14 µs |  12.86 ±  1.79 µs |   626   |  659.57K   | 5168.13 ± 602.59% |
|   la20   |   0.05 s   |   1.21 ±  0.17 ms  |  12.14 ±  1.66 µs |   577   |  482.47K   | 4080.88 ± 451.74% |
|   la30   |   0.18 s   |   4.03 ±  0.46 ms  |  20.13 ±  2.28 µs |  2.37K  |  593.54K   | 4415.53 ± 401.62% |
|   la40   |   0.18 s   |   4.22 ±  0.32 ms  |  18.76 ±  1.40 µs |  1.97K  |  469.09K   | 4186.11 ± 288.09% |
|  orb10   |   0.05 s   |   1.34 ±  0.19 ms  |  13.39 ±  1.92 µs |   785   |  595.29K   | 3691.65 ± 408.48% |
|  swv10   |   0.30 s   |   9.42 ±  5.80 ms  |  31.40 ± 19.34 µs |  5.15K  |  580.80K   | 3281.33 ± 369.71% |
|  swv20   |   1.00 s   |  24.78 ±  0.99 ms  |  49.57 ±  1.98 µs |  12.98K |  524.72K   | 3940.94 ± 155.72% |
|   ta10   |   0.16 s   |   4.69 ±  0.41 ms  |  20.86 ±  1.83 µs |  2.28K  |  490.10K   | 3333.26 ± 284.52% |
|   ta20   |   0.30 s   |   7.08 ±  0.32 ms  |  23.59 ±  1.07 µs |  3.45K  |  488.35K   | 4147.73 ± 180.37% |
|   ta30   |   0.40 s   |  13.19 ±  5.97 ms  |  32.98 ± 14.93 µs |  4.83K  |  381.39K   | 3057.23 ± 301.75% |
|   ta40   |   0.60 s   |  16.30 ±  0.66 ms  |  36.22 ±  1.47 µs |  7.12K  |  437.36K   | 3587.19 ± 139.91% |
|   ta50   |   0.80 s   |  28.79 ±  5.80 ms  |  47.99 ±  9.66 µs |  9.69K  |  341.78K   | 2721.09 ± 226.84% |
|   ta60   |   1.60 s   |  52.47 ±  5.92 ms  |  69.96 ±  7.89 µs |  19.59K |  375.78K   | 2969.82 ± 185.36% |
|   ta70   |   2.00 s   |  86.72 ±  6.44 ms  |  86.72 ±  6.44 µs |  25.96K |  300.57K   | 2215.70 ± 129.34% |
|   ta80   |   7.80 s   | 408.74 ± 23.35 ms  | 204.37 ± 11.67 µs | 103.84K |  254.87K   | 1814.45 ± 107.91% |

average and standard variation between 100 runs per instance.\
*benchmarked with available JSP environments, estimated time.
