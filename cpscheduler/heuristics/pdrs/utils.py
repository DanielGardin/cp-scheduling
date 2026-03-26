from typing import SupportsFloat

from math import exp, log, sqrt

def explst(lst: list[SupportsFloat]) -> list[float]:
    return [exp(i) for i in lst]


def loglst(lst: list[SupportsFloat]) -> list[float]:
    return [log(i) for i in lst]


def sqrtlst(lst: list[SupportsFloat]) -> list[float]:
    return [sqrt(i) for i in lst]
