import math

import pytest

import outline

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

P = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]

CASES = [
    ("verts", lambda: len(outline.Outline().build(P, 0.1)[0]) == 8),
    ("area_grow", lambda: outline.Outline().area(outline.Outline().build(P, 0.1)[0]) > outline.Outline().area(P)),
    ("round_join", lambda: isinstance(outline.Outline().build(P, 0.1, 'round'), tuple)),
    ("bad_join", lambda: _raises(lambda: outline.Outline().build(P, 0.1, 'bogus'))),
    ("bad_width", lambda: _raises(lambda: outline.Outline().build(P, 0.0))),
    ("bad_poly", lambda: _raises(lambda: outline.Outline().build([(0.0, 0.0)], 0.1))),
    ("type", lambda: isinstance(outline.Outline().build(P, 0.1), tuple)),
    ("det", lambda: outline.Outline().build(P, 0.1) == outline.Outline().build(P, 0.1)),
    ("no_mut", lambda: outline.Outline().build(list(P), 0.1) is not None),
    ("area_type", lambda: isinstance(outline.Outline().area(P), float)),
    ("closed", lambda: outline.Outline().build(P, 0.1)[1] is not None),
    ("nan", lambda: _raises(lambda: outline.Outline().build([(float('nan'), 0.0)] * 4, 0.1))),
    ("finite", lambda: all(math.isfinite(c) for p in outline.Outline().build(P, 0.1)[0] for c in p)),
    ("area_square", lambda: abs(outline.Outline().area(P) - 4.0) < 1e-9),
    ("float", lambda: isinstance(outline.Outline().build(P, 0.1)[0][0][0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
