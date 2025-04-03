from functools import partial as prt
from operator import gt, lt
import numpy as np
from toolz import compose as cmp

tpfilter = cmp(tuple, filter)
tpmap = cmp(tuple, map)
gt0 = prt(lt, 0)


def pop_dict(mapping, *x):
    for key in x:
        mapping.pop(key)
    return mapping


def identity(x):
    return x


def identity_mult(*x):
    return x


def first(*x, **y):
    return x[0]


def constant_1(*x, **y) -> int:
    return 1


def constant_none(*x, **y) -> None:
    return None


def p4_str(x) -> str:
    if isinstance(x, (float, np.floating)):
        return "{:.4f}".format(x)
    # if x is None:
    #     return ""
    return str(x)


def tpmap_p4_str(x) -> tuple[str]:
    return tpmap(p4_str, x)
