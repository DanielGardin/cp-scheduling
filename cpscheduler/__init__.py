from . import environment
from .common import is_compiled

__compiled__ = is_compiled()
__version__ = "0.8.0"

__all__ = ["environment"]
