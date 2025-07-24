# 🧩 Constraint Programming Scheduler

This repository provides a comprehensive gym environment for applying constraint programming techniques to scheduling problems.
It focuses on the learning and optimization of priority dispatch rules, enabling the development and testing of intelligent heuristics for dynamic scheduling scenarios.
Designed for researchers and practitioners, the framework supports customization, benchmarking, and integration with machine learning approaches to enhance scheduling performance.

This work was directly inspired by [JobShopCPEnv  ](https://github.com/ingambe/JobShopCPEnv), which only considers jobshop scheduling problems.
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

```
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

-   **`environment`**: This is the core of the project. It provides the `SchedulingEnv` class, which is a flexible and extensible environment for modeling and simulating scheduling problems. It also includes various scheduling setups (e.g., `JobShopSetup`, `SingleMachineSetup`), constraints (e.g., `PrecedenceConstraint`, `ResourceConstraint`), and objectives (e.g., `Makespan`, `TotalCompletionTime`).

-   **`instances`**: Provides functions for reading and generating scheduling problem instances, including well-known benchmark sets like Taillard's for Job Shop and RCPSP instances.
-   **`heuristics`**: Implements a variety of priority dispatching rules (PDRs) such as `ShortestProcessingTime`, `MostOperationsRemaining`, and `EarliestDueDate`. These can be used as baseline policies or as building blocks for more complex scheduling agents.
-   **`solver`**: Integrates with external solvers. Currently, it includes a `pulp` backend for formulating and solving scheduling problems as Mixed-Integer Linear Programs (MILPs). This module depends on the `environment` module.
- **`rl`**: Module for Reinforcement-Learning-based agents, it is split into the modules for
  -   **`policies`**: Contains neural network-based policies that can be trained to make scheduling decisions. This includes common architectures like `MLP` and `TransformerEncoder`, as well as the `PlackettLucePolicy` for learning permutation-based actions.
    -   **`algorithms`**: Contains implementations of reinforcement learning algorithms, both online (e.g., REINFORCE) and offline (e.g., Behavioral Cloning). This module depends on the `environment`, `policies`, and `heuristics` modules.

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

| Instance | Benchmark | Simulation time | Time per task  |      Speedup      |
| :------: | :-------: | :-------------: | :------------: | :---------------: |
|  dmu10   |   0.40s   | 0.026 ± 0.003 s | 0.07 ± 0.01 ms |  1424.79 ± 82.74% |
|  dmu20   |   0.80s   | 0.059 ± 0.001 s | 0.10 ± 0.00 ms |  1256.08 ± 24.25% |
|  dmu30   |   1.40s   | 0.107 ± 0.005 s | 0.13 ± 0.01 ms |  1205.81 ± 50.94% |
|  dmu40   |   2.10s   | 0.173 ± 0.006 s | 0.17 ± 0.01 ms |  1115.56 ± 38.35% |
|  dmu50   |   0.40s   | 0.025 ± 0.000 s | 0.06 ± 0.00 ms |  1480.56 ± 10.55% |
|  dmu60   |   0.80s   | 0.059 ± 0.003 s | 0.10 ± 0.00 ms |  1262.02 ± 43.10% |
|  dmu70   |   1.60s   | 0.109 ± 0.005 s | 0.14 ± 0.01 ms |  1372.71 ± 60.15% |
|  dmu80   |   2.20s   | 0.170 ± 0.004 s | 0.17 ± 0.00 ms |  1191.38 ± 29.01% |
|   la10   |   0.05s   | 0.001 ± 0.002 s | 0.02 ± 0.03 ms | 4679.67 ± 476.91% |
|   la20   |   0.05s   | 0.002 ± 0.000 s | 0.02 ± 0.00 ms |  2720.45 ± 32.21% |
|   la30   |   0.18s   | 0.006 ± 0.000 s | 0.03 ± 0.00 ms |  3013.59 ± 35.06% |
|   la40   |   0.18s   | 0.008 ± 0.002 s | 0.04 ± 0.01 ms | 2238.22 ± 182.18% |
|  orb10   |   0.05s   | 0.002 ± 0.000 s | 0.02 ± 0.00 ms |  2682.24 ± 38.29% |
|  swv10   |   0.30s   | 0.014 ± 0.000 s | 0.05 ± 0.00 ms |  2099.66 ± 19.89% |
|  swv20   |   1.00s   | 0.038 ± 0.002 s | 0.08 ± 0.00 ms | 2553.09 ± 107.20% |
|   ta10   |   0.16s   | 0.007 ± 0.000 s | 0.03 ± 0.00 ms |  2036.48 ± 20.03% |
|   ta20   |   0.30s   | 0.014 ± 0.000 s | 0.05 ± 0.00 ms |  2113.01 ± 31.99% |
|   ta30   |   0.40s   | 0.025 ± 0.003 s | 0.06 ± 0.01 ms |  1493.83 ± 88.12% |
|   ta40   |   0.60s   | 0.031 ± 0.003 s | 0.07 ± 0.01 ms | 1822.18 ± 118.70% |
|   ta50   |   0.80s   | 0.058 ± 0.003 s | 0.10 ± 0.01 ms |  1273.19 ± 55.66% |
|   ta60   |   1.60s   | 0.092 ± 0.003 s | 0.12 ± 0.00 ms |  1633.49 ± 53.32% |
|   ta70   |   2.00s   | 0.168 ± 0.005 s | 0.17 ± 0.01 ms |  1088.43 ± 34.31% |
|   ta80   |   7.80s   | 0.726 ± 0.013 s | 0.36 ± 0.01 ms |  974.54 ± 19.65%  |

average and standard variation between 100 runs per instance.\
*benchmarked with available JSP environments, estimated time.