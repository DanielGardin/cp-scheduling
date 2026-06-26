"""CPScheduler is a Python framework for modeling, simulating and solving scheduling problems.

The core of the framework is a pure-Python environment that can model
a wide variety of scheduling problems within a unified interface.
The environment generalizes the constraint-based approach to scheduling,
allowing custom constraints and objectives to be easily defined and integrated,
and enabling the support diverse problems (e.g., job-shop, flow-shop, RCPSP, etc.)
rather than being limited to one problem class.

The framework also includes optional support for Gymnasium via a separate
wrapper that provides a friendly interface for reinforcement learning applications,
and can be used to train and evaluate RL agents on scheduling problems.

"""

from . import environment
from .common import is_compiled

__compiled__ = is_compiled()
__version__ = "0.8.1"

__all__ = ["environment"]
