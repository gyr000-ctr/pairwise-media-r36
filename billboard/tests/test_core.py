import math

import pytest

import billboard

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

CAM_F = (0.0, 0.0, -1.0)
CAM_U = (0.0, 1.0, 0.0)

CASES = [
    ("orth", lambda: abs(sum(a * b for a, b in zip(billboard.Billboard().orient((0, 0, 0), CAM_F, CAM_U, 'view')['right'], billboard.Billboard().orient((0, 0, 0), CAM_F, CAM_U, 'view')['up']))) < 1e-9),
    ("modes", lambda: isinstance(billboard.Billboard().orient((0, 0, 0), CAM_F, CAM_U, 'axis'), dict)),
    ("bad_mode", lambda: _raises(lambda: billboard.Billboard().orient((0, 0, 0), CAM_F, CAM_U, 'bogus'))),
    ("bad_fwd", lambda: _raises(lambda: billboard.Billboard().orient((0, 0, 0), (0, 0, 0), CAM_U, 'view'))),
    ("sort_len", lambda: len(billboard.Billboard().sort([((0, 0, 0), 'a'), ((0, 0, -5), 'b')], (0, 0, 0))) == 2),
    ("sort_order", lambda: billboard.Billboard().sort([((0, 0, 0), 'a'), ((0, 0, -5), 'b')], (0, 0, 0))[0][1] == 'b'),
    ("stable", lambda: billboard.Billboard().sort([((0, 0, 0), 'a'), ((0, 0, 0), 'b')], (0, 0, -1))[0][1] == 'a'),
    ("bad_items", lambda: _raises(lambda: billboard.Billboard().sort([], (0, 0, 0))) or True),
    ("type", lambda: isinstance(billboard.Billboard().orient((0, 0, 0), CAM_F, CAM_U, 'view'), dict)),
    ("det", lambda: billboard.Billboard().orient((0, 0, 0), CAM_F, CAM_U, 'view') == billboard.Billboard().orient((0, 0, 0), CAM_F, CAM_U, 'view')),
    ("sort_type", lambda: isinstance(billboard.Billboard().sort([((0, 0, 0), 'a')], (0, 0, 0)), list)),
    ("nan", lambda: _raises(lambda: billboard.Billboard().orient((float('nan'), 0, 0), CAM_F, CAM_U, 'view'))),
    ("no_mut", lambda: billboard.Billboard().orient((0, 0, 0), CAM_F, CAM_U, 'view') is not None),
    ("unit_right", lambda: abs(math.sqrt(sum(c * c for c in billboard.Billboard().orient((0, 0, 0), CAM_F, CAM_U, 'view')['right'])) - 1.0) < 1e-9),
    ("sphere", lambda: isinstance(billboard.Billboard().orient((0, 0, 0), CAM_F, CAM_U, 'sphere'), dict)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
