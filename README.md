# Constraint Programming Scheduler

This repository provides a comprehensive gym environment for applying constraint programming techniques to scheduling problems.
It focuses on the learning and optimization of priority dispatch rules, enabling the development and testing of intelligent heuristics for dynamic scheduling scenarios.
Designed for researchers and practitioners, the framework supports customization, benchmarking, and integration with machine learning approaches to enhance scheduling performance.

This work was directly inspired by [another project](https://github.com/ingambe/JobShopCPEnv), which only considers jobshop scheduling problems.
We expand the previous idea of building an environment that automatically ensure the problem's constraints along the decisions.
The major contribution here is providing a flexible and expansible framework for enumerous scheduling problems following the scheduling literature (see [Pinedo, 2022](https://link.springer.com/book/10.1007/978-3-031-05921-6) for a complete review).


## Installation

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

## High-Level Structure

The project is organized into the following main directories inside `cpscheduler`:

```
🧩 cpscheduler
├── 🧠 algorithms
│   ├── offline
│   │   └── bc
│   │
│   └── online
│       └── reinforce.py
│
├── 🏭 environment
│   ├── constraints
│   ├── env
│   ├── instructions
│   ├── objectives
│   ├── schedule_setup
│   ├── tasks
│   └── utils
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
├── 🎮 policies
│
└── ⚙️ solver
    └── pulp
```

-   **`environment`**: This is the core of the project. It provides the `SchedulingEnv` class, which is a flexible and extensible environment for modeling and simulating scheduling problems. It also includes various scheduling setups (e.g., `JobShopSetup`, `SingleMachineSetup`), constraints (e.g., `PrecedenceConstraint`, `ResourceConstraint`), and objectives (e.g., `Makespan`, `TotalCompletionTime`).

-   **`instances`**: Provides functions for reading and generating scheduling problem instances, including well-known benchmark sets like Taillard's for Job Shop and RCPSP instances.
-   **`heuristics`**: Implements a variety of priority dispatching rules (PDRs) such as `ShortestProcessingTime`, `MostOperationsRemaining`, and `EarliestDueDate`. These can be used as baseline policies or as building blocks for more complex scheduling agents.
-   **`solver`**: Integrates with external solvers. Currently, it includes a `pulp` backend for formulating and solving scheduling problems as Mixed-Integer Linear Programs (MILPs). This module depends on the `environment` module.
-   **`policies`**: Contains neural network-based policies that can be trained to make scheduling decisions. This includes common architectures like `MLP` and `TransformerEncoder`, as well as the `PlackettLucePolicy` for learning permutation-based actions.
-   **`algorithms`**: Contains implementations of reinforcement learning algorithms, both online (e.g., REINFORCE) and offline (e.g., Behavioral Cloning). This module depends on the `environment`, `policies`, and `heuristics` modules.

## Usage

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


## Compilation

This project uses `mypyc` to compile the Python code to C extensions for performance automatically via `pip install`.
If the installation fail due to compilation issues, guarantee a C compiler is available and python-dev is installed in your machine.
For any issues with mypyc, we refer to the [original project](https://github.com/mypyc/mypyc). 
Due to compilation, we can achieve great performance, for example in jobshop instances,

| Instance | Benchmark* |    Step time    |      Speedup      |
| :------: | :-------: | :-------------: | :---------------: |
|  dmu10   |   0.40s   | 0.037 ± 0.001 s |  978.85 ± 21.73%  |
|  dmu20   |   0.80s   | 0.087 ± 0.002 s |  822.52 ± 19.40%  |
|  dmu30   |   1.40s   | 0.157 ± 0.004 s |  790.73 ± 18.44%  |
|  dmu40   |   2.10s   | 0.248 ± 0.003 s |  747.46 ± 10.12%  |
|  dmu50   |   0.40s   | 0.036 ± 0.000 s |  1003.91 ± 6.33%  |
|  dmu60   |   0.80s   | 0.085 ± 0.000 s |   840.46 ± 4.47%  |
|  dmu70   |   1.60s   | 0.156 ± 0.001 s |   924.24 ± 8.85%  |
|  dmu80   |   2.20s   | 0.247 ± 0.001 s |   789.78 ± 3.61%  |
|   la10   |   0.05s   | 0.001 ± 0.001 s | 3865.74 ± 357.50% |
|   la20   |   0.05s   | 0.002 ± 0.000 s |  2146.39 ± 15.07% |
|   la30   |   0.18s   | 0.008 ± 0.000 s |  2092.00 ± 17.52% |
|   la40   |   0.18s   | 0.011 ± 0.000 s |  1583.38 ± 12.15% |
|  orb10   |   0.05s   | 0.002 ± 0.000 s |  2127.47 ± 15.52% |
|  swv10   |   0.30s   | 0.020 ± 0.000 s |  1424.40 ± 10.15% |
|  swv20   |   1.00s   | 0.057 ± 0.000 s |  1662.94 ± 12.48% |
|   ta10   |   0.16s   | 0.011 ± 0.000 s |  1423.44 ± 12.02% |
|   ta20   |   0.30s   | 0.019 ± 0.000 s |  1447.13 ± 9.62%  |
|   ta30   |   0.40s   | 0.036 ± 0.000 s |  1009.38 ± 5.76%  |
|   ta40   |   0.60s   | 0.046 ± 0.000 s |  1203.32 ± 6.31%  |
|   ta50   |   0.80s   | 0.085 ± 0.000 s |   838.05 ± 4.56%  |
|   ta60   |   1.60s   | 0.134 ± 0.001 s |  1092.41 ± 5.09%  |
|   ta70   |   2.00s   | 0.247 ± 0.001 s |   709.77 ± 3.94%  |
|   ta80   |   7.80s   | 1.047 ± 0.021 s |  645.45 ± 14.35%  |

average and standard variation between 100 runs per instance.\
*benchmarked with available JSP environments, estimated time.