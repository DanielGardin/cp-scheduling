# Constraint Programming Scheduler

This repository provides a comprehensive environment for applying constraint programming techniques to scheduling problems. It focuses on the learning and optimization of priority dispatch rules, enabling the development and testing of intelligent heuristics for dynamic scheduling scenarios. Designed for researchers and practitioners, the framework supports customization, benchmarking, and integration with machine learning approaches to enhance scheduling performance.

## Installation

To install the project, clone the repository and install the dependencies:

```bash
git clone https://github.com/DanielGardin/cp-scheduling.git
cd cp-scheduling
pip install -e .
```

To install the reinforcement learning dependencies, run:

```bash
pip install -e .[rl]
```

To install the solver dependencies with CP and MILP formulations for the scheduling environments, run:

```bash
pip install -e .[solver]
```

## Compilation

This project uses `mypyc` to compile the Python code to C extensions for performance. The compiled files are specified in the `setup.py` file. During instalation, mypy and its extension will be installed.
Due to compilation, we can achieve great performance, for example in jobshop instances,

| Instance | Benchmark |     Time took     |      Speedup      |
| :------: | :-------: | :---------------: | :---------------: |
|  dmu10   |   0.40 s  | 0.038 s ± 0.000 s |  955.96% ± 5.60%  |
|  dmu20   |   0.80 s  | 0.091 s ± 0.000 s |  775.48% ± 2.78%  |
|  dmu30   |   1.40 s  | 0.169 s ± 0.001 s |  730.00% ± 3.62%  |
|  dmu40   |   2.10 s  | 0.271 s ± 0.002 s |  675.35% ± 5.74%  |
|  dmu50   |   0.40 s  | 0.039 s ± 0.000 s |  934.69% ± 3.53%  |
|  dmu60   |   0.80 s  | 0.092 s ± 0.001 s |  765.36% ± 4.99%  |
|  dmu70   |   1.60 s  | 0.174 s ± 0.001 s |  818.09% ± 3.50%  |
|  dmu80   |   2.20 s  | 0.280 s ± 0.001 s |  686.81% ± 3.49%  |
|   la10   |   0.05 s  | 0.001 s ± 0.000 s | 3489.12% ± 20.70% |
|   la20   |   0.05 s  | 0.002 s ± 0.000 s | 1990.04% ± 10.30% |
|   la30   |   0.18 s  | 0.009 s ± 0.000 s |  1920.84% ± 7.75% |
|   la40   |   0.18 s  | 0.012 s ± 0.000 s | 1463.21% ± 11.46% |
|  orb10   |   0.05 s  | 0.002 s ± 0.000 s |  1916.23% ± 5.79% |
|  swv10   |   0.30 s  | 0.022 s ± 0.000 s |  1291.80% ± 6.46% |
|  swv20   |   1.00 s  | 0.062 s ± 0.000 s |  1520.93% ± 6.27% |
|   ta10   |   0.16 s  | 0.011 s ± 0.000 s |  1294.56% ± 6.80% |
|   ta20   |   0.30 s  | 0.021 s ± 0.000 s |  1308.55% ± 9.41% |
|   ta30   |   0.40 s  | 0.039 s ± 0.000 s |  935.56% ± 3.80%  |
|   ta40   |   0.60 s  | 0.050 s ± 0.000 s |  1101.11% ± 5.13% |
|   ta50   |   0.80 s  | 0.094 s ± 0.001 s |  754.69% ± 4.57%  |
|   ta60   |   1.60 s  | 0.151 s ± 0.000 s |  962.86% ± 3.23%  |
|   ta70   |   2.00 s  | 0.276 s ± 0.001 s |  624.76% ± 1.67%  |
|   ta80   |   7.80 s  | 1.185 s ± 0.007 s |  558.02% ± 4.09%  |

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
