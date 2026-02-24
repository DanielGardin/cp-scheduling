import os
from setuptools import setup, Extension
from pathlib import Path

# This solution is a workaround, rethink it
USE_MYPYC = os.environ.get("DISABLE_MYPYC", "0") != "1"

ext_modules: list[Extension] = []
if USE_MYPYC:
    from mypyc.build import mypycify
    compiling_dirs: list[str] = [
        "cpscheduler/environment",
        # "cpscheduler/utils",
        # "cpscheduler/instances",
        # "cpscheduler/heuristics",
    ]

    blacklist: set[str] = set([
        # "cpscheduler/environment/instructions.py",
    ])

    compiling_files = [
        str(file)
        for d in compiling_dirs
        for file in Path(d).rglob("*.py")
        if not file.name.startswith("_") and str(file) not in blacklist
    ]

    ext_modules = mypycify(compiling_files)

setup(
    ext_modules=ext_modules,
)
