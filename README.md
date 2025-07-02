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
