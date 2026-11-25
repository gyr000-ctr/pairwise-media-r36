import math

import pytest

import particles

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _p():
    return particles.Particles(100, 1)

CASES = [
    ("initial", lambda: _p().live() == 0),
    ("emit", lambda: (_p().emit(10.0, 1.0), _p().live())[1] > 0),
    ("cap", lambda: (_p().emit(1000.0, 1.0), _p().live())[1] <= 100),
    ("recycle", lambda: (_p().emit(10.0, 1.0), _p().update(10.0), _p().live())[2] == 0),
    ("bad_cap", lambda: _raises(lambda: particles.Particles(0, 1))),
    ("bad_rate", lambda: _raises(lambda: _p().emit(-1.0, 1.0))),
    ("bad_dt", lambda: _raises(lambda: _p().update(-1.0))),
    ("type", lambda: isinstance(_p().live(), int)),
    ("det", lambda: (_p().emit(5.0, 1.0), _p().live())[1] == (_p().emit(5.0, 1.0), _p().live())[1]),
    ("seed_diff", lambda: (_p().emit(5.0, 1.0), _p().positions())[1] != (particles.Particles(100, 2).emit(5.0, 1.0), particles.Particles(100, 2).positions())[1]),
    ("nan", lambda: _raises(lambda: _p().update(float('nan')))),
    ("no_mut", lambda: _p().update(0.1) is not None),
    ("positions", lambda: isinstance(_p().positions(), list)),
    ("reuse_count", lambda: isinstance(_p().reuse_count(), int)),
    ("float_dt", lambda: _p().update(0.0) is not None),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
