from __future__ import annotations

import os
import sys
from pathlib import Path
from setuptools import Extension, setup

if sys.version_info < (3, 10, 0):
    sys.stderr.write("ERROR: You need Python 3.10 or later to use cpscheduler.\n")
    sys.exit(1)

USE_MYPYC = (
    os.environ.get("MYPYC_DISABLE", "0") != "1"
    and "--no-mypyc" not in sys.argv
)

if "--no-mypyc" in sys.argv:
    sys.argv.remove("--no-mypyc")

ext_modules: list[Extension] = []
if USE_MYPYC:
    MYPYC_DIRS = (
        "cpscheduler/environment",
        "cpscheduler/heuristics/pdrs",
    )

    MYPYC_BLACKLIST = frozenset((
        # Protocols with @runtime_checkable, mypyc strips Protocol identity
        "cpscheduler/environment/protocols.py",
    ))

    mypyc_targets = sorted(
        str(file)
        for d in MYPYC_DIRS
        for file in Path(d).rglob("*.py")
        if str(file) not in MYPYC_BLACKLIST
        and file.name != "__init__.py"
        and "/test" not in str(file)
    )

    from mypyc.build import mypycify

    opt_level = os.environ.get("MYPYC_OPT_LEVEL", "3")
    debug_level = os.environ.get("MYPYC_DEBUG_LEVEL", "1")

    ext_modules = mypycify(
        mypyc_targets,
        opt_level=opt_level,
        debug_level=debug_level,
        strict_dunder_typing=True,
        strip_asserts=True,
        verbose=True
    )

setup(ext_modules=ext_modules)
