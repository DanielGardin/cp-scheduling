from pathlib import Path
import sys

from setuptools import setup
from mypyc.build import mypycify

compiling_dirs = [
    "cpscheduler/environment",
    "cpscheduler/instances",
    "cpscheduler/heuristics",
    "cpscheduler/utils",
]

compiling_files: list[str] = []

for dir_name in compiling_dirs:
    compiling_files.extend(
        [str(file) for file in Path(dir_name).rglob("*.py") if not file.name.startswith("_")]
    )

setup(
    ext_modules=mypycify(compiling_files),
)