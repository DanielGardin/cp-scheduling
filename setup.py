import os
from setuptools import setup, Extension
from pathlib import Path

# This solution is a workaround, rethink it
USE_MYPYC = os.environ.get("DISABLE_MYPYC", "0") != "1"

ext_modules: list[Extension] = []
if USE_MYPYC:
    from mypyc.build import mypycify
    compiling_dirs = [
        "cpscheduler/environment",
        # "cpscheduler/instances",
        # "cpscheduler/heuristics",
        # "cpscheduler/utils",
    ]

    compiling_files = [
        str(file)
        for d in compiling_dirs
        for file in Path(d).rglob("*.py")
        if not file.name.startswith("_")
    ]

    ext_modules = mypycify(compiling_files)

setup(
    ext_modules=ext_modules,
)
