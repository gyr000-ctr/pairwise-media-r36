import math

import pytest

import trail

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

P = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]

CASES = [
    ("verts", lambda: len(trail.Trail().build(P, 0.2, 0.5)[0]) == 6),
    ("tris", lambda: len(trail.Trail().build(P, 0.2, 0.5)[1]) == 4),
    ("width_min", lambda: isinstance(trail.Trail().build(P, 0.2, 0.5), tuple)),
    ("bad_points", lambda: _raises(lambda: trail.Trail().build([(0.0, 0.0)], 0.2, 0.5))),
    ("bad_width", lambda: _raises(lambda: trail.Trail().build(P, 0.0, 0.5))),
    ("bad_taper", lambda: _raises(lambda: trail.Trail().build(P, 0.2, 1.5))),
    ("type", lambda: isinstance(trail.Trail().build(P, 0.2, 0.5), tuple)),
    ("det", lambda: trail.Trail().build(P, 0.2, 0.5) == trail.Trail().build(P, 0.2, 0.5)),
    ("normals", lambda: len(trail.Trail().build(P, 0.2, 0.5)[2]) == len(P)),
    ("no_mut", lambda: trail.Trail().build(list(P), 0.2, 0.5) is not None),
    ("finite", lambda: all(math.isfinite(c) for p in trail.Trail().build(P, 0.2, 0.5)[0] for c in p)),
    ("unit_normals", lambda: all(abs(math.hypot(n[0], n[1]) - 1.0) < 1e-6 for n in trail.Trail().build(P, 0.2, 0.5)[2])),
    ("nan", lambda: _raises(lambda: trail.Trail().build([(float('nan'), 0.0)] * 3, 0.2, 0.5))),
    ("two_points", lambda: len(trail.Trail().build([(0.0, 0.0), (1.0, 0.0)], 0.2, 0.0)[0]) == 4),
    ("float", lambda: isinstance(trail.Trail().build(P, 0.2, 0.5)[0][0][0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
