import math

import pytest

import hudlayout

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

W = [("hp", {"anchor": "topleft", "size": (100, 20), "margin": (8, 8)})]
S = (1280, 720)

CASES = [
    ("solve", lambda: len(hudlayout.HudLayout().solve(W, S)) == 1),
    ("anchor", lambda: hudlayout.HudLayout().solve(W, S)[0][0] == 8),
    ("bad_anchor", lambda: _raises(lambda: hudlayout.HudLayout().solve([('x', {'anchor': 'bogus', 'size': (1, 1), 'margin': (0, 0)})], S))),
    ("bad_screen", lambda: _raises(lambda: hudlayout.HudLayout().solve(W, (0, 720)))),
    ("cycle", lambda: _raises(lambda: hudlayout.HudLayout().solve([('a', {'anchor': 'b'}), ('b', {'anchor': 'a'})], S))),
    ("type", lambda: isinstance(hudlayout.HudLayout().solve(W, S), list)),
    ("det", lambda: hudlayout.HudLayout().solve(W, S) == hudlayout.HudLayout().solve(W, S)),
    ("rects", lambda: isinstance(hudlayout.HudLayout().rects(), list)),
    ("bad_widget", lambda: _raises(lambda: hudlayout.HudLayout().solve([('x', {})], S))),
    ("no_mut", lambda: hudlayout.HudLayout().solve(list(W), S) is not None),
    ("scaled", lambda: isinstance(hudlayout.HudLayout().solve(W, (2560, 1440)), list)),
    ("rect_len", lambda: len(hudlayout.HudLayout().solve(W, S)[0]) == 4),
    ("float", lambda: isinstance(hudlayout.HudLayout().solve(W, S)[0][0], float)),
    ("nan", lambda: _raises(lambda: hudlayout.HudLayout().solve(W, (float('nan'), 720)))),
    ("two", lambda: len(hudlayout.HudLayout().solve(W + [('mp', {'anchor': 'topright', 'size': (50, 10), 'margin': (8, 8)})], S)) == 2),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
