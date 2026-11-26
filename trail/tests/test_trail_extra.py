"""针对题面约束的补充测试。"""

import math
import time

import pytest

import trail
from trail import Trail


def straight(n, step=1.0):
    return [(float(i) * step, 0.0) for i in range(n)]


def test_vertex_and_triangle_counts():
    for n in (2, 3, 10, 100):
        m = Trail().build(straight(n), 0.4, 0.5)
        assert len(m.vertices) == 2 * n
        assert len(m.triangles) == 2 * (n - 1)
        assert len(m.normals) == n


def test_width_decays_and_respects_floor():
    n = 11
    m = Trail().build(straight(n), 1.0, 1.0, min_width=0.2)
    # 直线轨迹法线为 (0,1)，半宽即顶点 y 坐标绝对值。
    halfwidths = [m.vertices[2 * i][1] for i in range(n)]
    assert halfwidths[0] == pytest.approx(0.5)
    # 单调不增。
    for a, b in zip(halfwidths, halfwidths[1:]):
        assert b <= a + 1e-12
    # 衰减到下限后不再变小。
    assert min(halfwidths) == pytest.approx(0.1)
    assert halfwidths[-1] == pytest.approx(0.1)


def test_default_min_width_ratio():
    n = 20
    m = Trail().build(straight(n), 1.0, 1.0)
    halfwidths = [m.vertices[2 * i][1] for i in range(n)]
    assert min(halfwidths) >= 0.05 - 1e-12  # 0.1 * width / 2


def test_adjacent_normal_angle_within_5deg_on_sharp_path():
    # 锯齿急转路径。
    pts = [(float(i), float(i % 2)) for i in range(50)]
    m = Trail().build(pts, 0.2, 0.3)
    for a, b in zip(m.normals, m.normals[1:]):
        dot = a[0] * b[0] + a[1] * b[1]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
        assert ang <= 5.0 + 1e-6


def test_normals_unit_and_finite():
    pts = [(math.sin(i), math.cos(i * 0.7)) for i in range(200)]
    m = Trail().build(pts, 0.3, 0.4)
    for nx, ny in m.normals:
        assert math.isfinite(nx) and math.isfinite(ny)
        assert math.hypot(nx, ny) == pytest.approx(1.0)


def test_degenerate_coincident_points():
    m = Trail().build([(1.0, 1.0)] * 5, 0.2, 0.5)
    assert len(m.vertices) == 10
    assert all(math.isfinite(c) for v in m.vertices for c in v)


def test_errors_are_distinguishable():
    P = straight(3)
    with pytest.raises(trail.TooFewPointsError):
        Trail().build([(0.0, 0.0)], 0.2, 0.5)
    with pytest.raises(trail.InvalidWidthError):
        Trail().build(P, 0.0, 0.5)
    with pytest.raises(trail.InvalidWidthError):
        Trail().build(P, -1.0, 0.5)
    with pytest.raises(trail.InvalidTaperError):
        Trail().build(P, 0.2, 1.5)
    with pytest.raises(trail.NaNInputError):
        Trail().build(P, float("nan"), 0.5)
    with pytest.raises(trail.InfiniteInputError):
        Trail().build(P, 0.2, float("inf"))
    with pytest.raises(trail.NaNInputError):
        Trail().build([(float("nan"), 0.0)] * 3, 0.2, 0.5)
    # 全部都是 ValueError，可统一捕获。
    for fn in (
        lambda: Trail().build([(0.0, 0.0)], 0.2, 0.5),
        lambda: Trail().build(P, -1.0, 0.5),
        lambda: Trail().build(P, 0.2, 2.0),
    ):
        with pytest.raises(ValueError):
            fn()
    # 错误码可区分。
    try:
        Trail().build(P, 0.0, 0.5)
    except trail.InvalidWidthError as e:
        assert e.code == "invalid_width" and e.field == "width"


def test_triangle_indices_in_range_and_ccw():
    m = Trail().build(straight(6), 0.4, 0.2)
    nv = len(m.vertices)
    for tri in m.triangles:
        assert all(0 <= idx < nv for idx in tri)
        (a, b, c) = [m.vertices[idx] for idx in tri]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        assert cross > 0  # 逆时针


def test_performance_10k_points():
    pts = straight(10_000, 0.01)
    t0 = time.perf_counter()
    Trail().build(pts, 0.5, 0.8)
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.7


def test_concurrent_builds_are_independent():
    import concurrent.futures

    P = straight(200)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: Trail().build(P, 0.3, 0.6), range(64)))
    assert all(r == results[0] for r in results)
