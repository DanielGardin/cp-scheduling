from pathlib import Path
from typing import Any
from collections.abc import Callable

import sys

if sys.version_info >= (3, 11):
    from tomllib import load

else:
    from tomli import load

from setuptools import setup
from mypyc.build import mypycify

# Load metadata from pyproject.toml
with open("pyproject.toml", "rb") as f:
    project_info = load(f)["project"]

compiling_dirs = [
    "cpscheduler/environment",
    "cpscheduler/instances",
]


compiling_files = [
    "cpscheduler/heuristics/pdr_heuristics.py",
]
for dir in compiling_dirs:
    compiling_files.extend(
        [str(file) for file in Path(dir).rglob("*.py") if not file.name.startswith("_")]
    )


setup(
    name=project_info["name"],
    version=project_info["version"],
    description=project_info["description"],
    author=project_info["authors"][0]["name"],
    author_email=project_info["authors"][0]["email"],
    packages=["cpscheduler"],
    install_requires=project_info.get("dependencies", []),
    tests_require=["pytest"],
    include_package_data=True,
    ext_modules=mypycify(compiling_files),
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
