"""Setup script for the cpscheduler package."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from setuptools import setup

USE_MYPYC = (
    os.environ.get("MYPYC_DISABLE", "0") != "1" and "--no-mypyc" not in sys.argv
)

if "--no-mypyc" in sys.argv:
    sys.argv.remove("--no-mypyc")

ext_modules: list[Any] = []
if USE_MYPYC:
    from mypyc.build import mypycify

    MYPYC_DIRS = (
        "cpscheduler/environment",
        "cpscheduler/heuristics/pdrs",
    )

    MYPYC_BLACKLIST = frozenset(
        (
            # Protocols with @runtime_checkable, mypyc strips Protocol identity
            "cpscheduler/environment/utils/protocols.py",
        )
    )

    for blacklisted in MYPYC_BLACKLIST:
        if not Path(blacklisted).exists():
            sys.stderr.write(
                f"ERROR: Mypyc blacklist file '{blacklisted}' does not exist.\n"
            )
            sys.exit(1)

    mypyc_targets: list[str] = []

    for directory in MYPYC_DIRS:
        if not Path(directory).exists():
            sys.stderr.write(
                f"ERROR: Mypyc target directory '{directory}' does not exist.\n"
            )
            sys.exit(1)

        for file in Path(directory).rglob("*.py"):
            if str(file) in MYPYC_BLACKLIST:
                continue

            if file.name == "__init__.py":
                continue

            if "/test" in str(file):
                continue

            mypyc_targets.append(str(file))

    opt_level = os.environ.get("MYPYC_OPT_LEVEL", "3")
    debug_level = os.environ.get("MYPYC_DEBUG_LEVEL", "1")

    ext_modules = mypycify(
        mypyc_targets,
        opt_level=opt_level,
        debug_level=debug_level,
        strict_dunder_typing=True,
        strip_asserts=True,
        verbose=True,
    )

setup(ext_modules=ext_modules)
