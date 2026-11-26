"""hudlayout 需求维度测试（官方 test_core.py 之外的补充）。

覆盖：锚点几何、线性约束精度、重叠消解与优先级、安全区/缩放重算、
循环与非法锚点分类、非有限输入、性能上界、并发安全。
"""

import os
import random
import threading
import time

import pytest

import hudlayout
from hudlayout import (
    AnchorError,
    CycleError,
    HudLayout,
    InvalidScreenError,
    InvalidWidgetError,
)

S = (1280, 720)


# --------------------------------------------------------------------------
# 锚点几何
# --------------------------------------------------------------------------


def rect(anchor, size, margin=(0, 0), **extra):
    spec = {"anchor": anchor, "size": size, "margin": margin}
    spec.update(extra)
    return HudLayout().solve([("w", spec)], S)[0]


@pytest.mark.parametrize(
    "anchor,size,margin,expected",
    [
        ("topleft", (100, 20), (8, 8), (8.0, 8.0)),
        ("topright", (50, 10), (8, 8), (1280 - 8 - 50.0, 8.0)),
        ("bottomleft", (50, 10), (8, 8), (8.0, 720 - 8 - 10.0)),
        ("bottomright", (50, 10), (8, 8), (1280 - 8 - 50.0, 720 - 8 - 10.0)),
        ("center", (100, 20), (0, 0), ((1280 - 100) / 2, (720 - 20) / 2)),
        ("topcenter", (100, 20), (0, 5), ((1280 - 100) / 2, 5.0)),
        ("centerleft", (40, 40), (7, 0), (7.0, (720 - 40) / 2)),
        ("centerright", (40, 40), (7, 0), (1280 - 7 - 40.0, (720 - 40) / 2)),
        ("bottomcenter", (40, 40), (0, 7), ((1280 - 40) / 2, 720 - 7 - 40.0)),
    ],
)
def test_screen_anchor_geometry(anchor, size, margin, expected):
    x, y, _, _ = rect(anchor, size, margin)
    assert (x, y) == pytest.approx(expected)


def test_anchor_alias_equivalence():
    a = rect("top", (40, 10))
    b = rect("topcenter", (40, 10))
    assert a == b


def test_relative_anchor_basic():
    res = HudLayout().solve(
        [
            ("a", {"anchor": "topleft", "size": (100, 100), "margin": (10, 10)}),
            ("b", {"anchor": "a", "size": (50, 20), "margin": (5, 7)}),
        ],
        S,
        resolve_overlaps=False,
    )
    assert res[0][:2] == (10.0, 10.0)
    assert res[1][:2] == (15.0, 17.0)


def test_relative_anchor_attach_points():
    res = HudLayout().solve(
        [
            ("a", {"anchor": "topleft", "size": (100, 50)}),
            ("b", {
                "anchor": "a",
                "self_anchor": "centerleft",
                "target_anchor": "centerright",
                "size": (20, 50),
            }),
        ],
        S,
        resolve_overlaps=False,
    )
    # b 的左边贴 a 的右边，垂直居中对齐
    assert res[1][0] == pytest.approx(100.0)
    assert res[1][1] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 线性约束精度
# --------------------------------------------------------------------------


def test_chain_matches_linear_recurrence():
    chain = [("n0", {"anchor": "topleft", "size": (10, 10), "margin": (3, 4)})]
    for i in range(1, 500):
        chain.append(
            (f"n{i}", {"anchor": f"n{i-1}", "size": (10, 10), "margin": (1.5, -2.25)})
        )
    res = HudLayout().solve(chain, S, resolve_overlaps=False)
    ex, ey = 3.0, 4.0
    for i in range(1, 500):
        ex += 1.5
        ey += -2.25
        assert res[i][0] == pytest.approx(ex, abs=1e-9)
        assert res[i][1] == pytest.approx(ey, abs=1e-9)


def test_dag_matches_independent_reference():
    rng = random.Random(42)
    n = 200
    specs = [("w0", {"anchor": "center", "size": (12, 12)})]
    for i in range(1, n):
        if rng.random() < 0.7:
            specs.append((
                f"w{i}",
                {"anchor": f"w{rng.randrange(i)}",
                 "size": (rng.uniform(5, 50), rng.uniform(5, 50)),
                 "margin": (rng.uniform(-20, 20), rng.uniform(-20, 20))},
            ))
        else:
            specs.append((
                f"w{i}",
                {"anchor": rng.choice(["topleft", "topright", "bottomleft",
                                       "bottomright", "center"]),
                 "size": (rng.uniform(5, 50), rng.uniform(5, 50)),
                 "margin": (rng.uniform(0, 30), rng.uniform(0, 30))},
            ))
    order = list(specs)
    rng.shuffle(order)
    got = dict(
        zip([w for w, _ in order],
            HudLayout().solve(order, (1920, 1080), resolve_overlaps=False))
    )

    # 独立参考解：前向代入（与被测实现分别编写）
    pt = {"topleft": (0., 0.), "topcenter": (.5, 0.), "topright": (1., 0.),
          "centerleft": (0., .5), "center": (.5, .5), "centerright": (1., .5),
          "bottomleft": (0., 1.), "bottomcenter": (.5, 1.), "bottomright": (1., 1.)}
    side = {"topleft": (-1, -1), "topcenter": (0, -1), "topright": (1, -1),
            "centerleft": (-1, 0), "center": (0, 0), "centerright": (1, 0),
            "bottomleft": (-1, 1), "bottomcenter": (0, 1), "bottomright": (1, 1)}
    ref = {}
    for wid, sp in specs:
        w, h = sp["size"]
        mx, my = sp.get("margin", (0.0, 0.0))
        if sp["anchor"] in pt:
            kx, ky = pt[sp["anchor"]]
            sdx, sdy = side[sp["anchor"]]
            px = (sdx + 1) * .5 * 1920 + (mx if sdx <= 0 else -mx)
            py = (sdy + 1) * .5 * 1080 + (my if sdy <= 0 else -my)
            ref[wid] = (px - kx * w, py - ky * h)
        else:
            skx, sky = pt[sp.get("self_anchor", "topleft")]
            tkx, tky = pt[sp.get("target_anchor", "topleft")]
            pw, ph = specs[int(sp["anchor"][1:])][1]["size"]
            px0, py0 = ref[sp["anchor"]]
            ref[wid] = (px0 + tkx * pw + mx - skx * w,
                        py0 + tky * ph + my - sky * h)
    for wid, _ in specs:
        assert got[wid][0] == pytest.approx(ref[wid][0], abs=1e-9)
        assert got[wid][1] == pytest.approx(ref[wid][1], abs=1e-9)


# --------------------------------------------------------------------------
# 重叠消解与优先级
# --------------------------------------------------------------------------


def _count_overlaps(rs):
    total = 0
    for a in range(len(rs)):
        for b in range(a + 1, len(rs)):
            ra, rb = rs[a], rs[b]
            if (ra[0] < rb[0] + rb[2] and rb[0] < ra[0] + ra[2]
                    and ra[1] < rb[1] + rb[3] and rb[1] < ra[1] + ra[3]):
                total += 1
    return total


def test_overlap_resolved_feasible_grid():
    widgets = [
        (f"s{i}", {"anchor": "topleft", "size": (40, 40),
                   "pos": ((i % 5) * 30.0, (i // 5) * 30.0)})
        for i in range(25)
    ]
    assert _count_overlaps(HudLayout().solve(widgets, (2000, 2000))) == 0


def test_priority_high_stays():
    res = HudLayout().solve(
        [
            ("lo", {"anchor": "topleft", "size": (100, 100), "priority": 1}),
            ("hi", {"anchor": "topleft", "size": (100, 100), "priority": 10}),
        ],
        S,
    )
    assert res[1][:2] == (0.0, 0.0)       # 高优先级不动
    assert res[0][:2] != (0.0, 0.0)       # 低优先级被推开


def test_ancestor_descendant_overlap_ignored():
    # b 锚定在 a 内部，属于有意嵌套，不应被推开
    res = HudLayout().solve(
        [
            ("a", {"anchor": "topleft", "size": (100, 100)}),
            ("b", {"anchor": "a", "size": (20, 20), "margin": (5, 5)}),
        ],
        S,
    )
    assert res[1][:2] == (5.0, 5.0)


def test_child_moves_rigidly_with_parent():
    res = HudLayout().solve(
        [
            ("a", {"anchor": "topleft", "size": (100, 100), "priority": 5}),
            ("b", {"anchor": "topleft", "size": (100, 100), "priority": 0}),
            ("c", {"anchor": "b", "size": (20, 20), "margin": (5, 5)}),
        ],
        S,
    )
    assert (res[2][0] - res[1][0]) == pytest.approx(5.0)
    assert (res[2][1] - res[1][1]) == pytest.approx(5.0)


def test_no_overlap_no_spurious_movement():
    ws = [
        ("a", {"anchor": "topleft", "size": (10, 10)}),
        ("b", {"anchor": "topright", "size": (10, 10)}),
    ]
    assert HudLayout().solve(ws, S) == HudLayout().solve(
        ws, S, resolve_overlaps=False
    )


# --------------------------------------------------------------------------
# 安全区 / 缩放重算
# --------------------------------------------------------------------------


def test_safe_area_anchors():
    screen = {"width": 1280, "height": 720, "safe_area": (100, 50, 1080, 620)}
    tl = HudLayout().solve(
        [("w", {"anchor": "topleft", "size": (10, 10)})], screen
    )[0]
    br = HudLayout().solve(
        [("w", {"anchor": "bottomright", "size": (10, 10)})], screen
    )[0]
    assert tl[:2] == (100.0, 50.0)
    assert br[:2] == (100 + 1080 - 10, 50 + 620 - 10)


def test_safe_area_insets():
    screen = {"width": 1280, "height": 720,
              "safe_area_insets": (20, 30, 40, 50)}
    r = HudLayout().solve(
        [("w", {"anchor": "topleft", "size": (10, 10)})], screen
    )[0]
    assert r[:2] == (20.0, 30.0)


def test_resize_recomputes_for_new_screen():
    h = HudLayout()
    h.solve([("w", {"anchor": "topright", "size": (50, 10), "margin": (8, 8)})],
            (1280, 720))
    r = h.resize((2560, 1440))
    assert r[0][0] == pytest.approx(2560 - 8 - 50)


# --------------------------------------------------------------------------
# 错误分类
# --------------------------------------------------------------------------


def test_invalid_screen_anchor():
    with pytest.raises(AnchorError):
        HudLayout().solve(
            [("x", {"anchor": "bogus", "size": (1, 1)})], S
        )


def test_unknown_widget_anchor():
    with pytest.raises(AnchorError):
        HudLayout().solve(
            [("x", {"anchor": "ghost", "size": (1, 1)})], S
        )


def test_cycle_distinct_from_bad_anchor():
    with pytest.raises(CycleError):
        HudLayout().solve(
            [("a", {"anchor": "b", "size": (1, 1)}),
             ("b", {"anchor": "a", "size": (1, 1)})], S
        )
    assert AnchorError is not CycleError


def test_self_cycle():
    with pytest.raises(CycleError):
        HudLayout().solve(
            [("a", {"anchor": "a", "size": (1, 1)})], S
        )


def test_cycle_reported_without_size():
    # 环检测优先于缺尺寸报错
    with pytest.raises(CycleError):
        HudLayout().solve([("a", {"anchor": "b"}), ("b", {"anchor": "a"})], S)


def test_missing_anchor_and_pos():
    with pytest.raises(InvalidWidgetError):
        HudLayout().solve([("x", {})], S)


@pytest.mark.parametrize(
    "screen",
    [(0, 720), (-1, 720), (float("nan"), 720), (100, float("inf")),
     (float("inf"), 100)],
)
def test_bad_screen(screen):
    with pytest.raises(InvalidScreenError):
        HudLayout().solve(
            [("x", {"anchor": "center", "size": (1, 1)})], screen
        )


@pytest.mark.parametrize(
    "field,value",
    [("size", (1, float("nan"))), ("size", (0, 5)), ("size", (5, -1)),
     ("margin", (float("nan"), 0)), ("pos", (1, float("inf")))],
)
def test_non_finite_or_invalid_widget_fields(field, value):
    spec = {"anchor": "center", "size": (4, 4)}
    if field == "size":
        spec = {"anchor": "center", "size": value}
    else:
        spec[field] = value
    with pytest.raises(InvalidWidgetError):
        HudLayout().solve([("x", spec)], S)


def test_duplicate_id():
    with pytest.raises(InvalidWidgetError):
        HudLayout().solve(
            [("x", {"anchor": "center", "size": (1, 1)}),
             ("x", {"anchor": "center", "size": (1, 1)})], S
        )


# --------------------------------------------------------------------------
# 不变量 / 快照
# --------------------------------------------------------------------------


def test_input_not_mutated():
    import copy
    src = [("a", {"anchor": "topleft", "size": (10, 10), "margin": (1, 2)})]
    snap = copy.deepcopy(src)
    HudLayout().solve(src, S)
    assert src == snap


def test_rects_returns_defensive_copy():
    h = HudLayout()
    h.solve([("a", {"anchor": "topleft", "size": (10, 10)})], S)
    snap = h.rects()
    snap[0] = (999, 999, 1, 1)
    assert h.rects()[0] != (999, 999, 1, 1)


def test_result_order_matches_input_and_is_float():
    ws = [
        ("hp", {"anchor": "topleft", "size": (100, 20), "margin": (8, 8)}),
        ("mp", {"anchor": "topright", "size": (50, 10), "margin": (8, 8)}),
    ]
    res = HudLayout().solve(ws, S)
    assert [r is not None for r in res]
    assert len(res) == 2 and all(len(r) == 4 for r in res)
    assert all(isinstance(v, float) for r in res for v in r)


# --------------------------------------------------------------------------
# 性能上界
# --------------------------------------------------------------------------


PERF_BUDGET = float(os.environ.get("HUDLAYOUT_PERF_BUDGET", "0.7"))


def _mixed_10k():
    rng = random.Random(7)
    ws = []
    for i in range(10000):
        sp = {"size": (rng.uniform(20, 80), rng.uniform(20, 80))}
        if i == 0 or rng.random() < 0.3:
            sp.update(
                anchor=rng.choice(["topleft", "topright", "bottomleft",
                                   "bottomright", "center"]),
                margin=(rng.uniform(0, 100), rng.uniform(0, 100)),
            )
        else:
            sp.update(anchor=f"w{rng.randrange(i)}",
                      margin=(rng.uniform(-30, 30), rng.uniform(-30, 30)))
        ws.append((f"w{i}", sp))
    return ws


def test_perf_10k_under_budget():
    t0 = time.perf_counter()
    res = HudLayout().solve(_mixed_10k(), (3840, 2160))
    assert time.perf_counter() - t0 < PERF_BUDGET
    assert len(res) == 10000


def test_perf_10k_chain():
    chain = [("c0", {"anchor": "topleft", "size": (5, 5)})]
    for i in range(1, 10000):
        chain.append((f"c{i}", {"anchor": f"c{i-1}", "size": (5, 5),
                                "margin": (0.5, 0.5)}))
    t0 = time.perf_counter()
    res = HudLayout().solve(chain, (10 ** 9, 10 ** 9))
    assert time.perf_counter() - t0 < PERF_BUDGET
    assert res[-1][0] == pytest.approx(0.5 * 9999)


def test_perf_adversarial_same_cell():
    # 一万个互不相交的极小控件落入同一网格单元，候选对爆炸由预算截断
    ws = [(f"t{i}", {"anchor": "topleft", "size": (1e-6, 1e-6),
                     "pos": (i * 1e-3, 0.0)}) for i in range(10000)]
    t0 = time.perf_counter()
    HudLayout().solve(ws, (1e6, 1e6))
    assert time.perf_counter() - t0 < PERF_BUDGET


# --------------------------------------------------------------------------
# 并发安全
# --------------------------------------------------------------------------


def test_concurrent_solve_thread_safe():
    shared = HudLayout()
    errors = []

    def worker():
        try:
            for _ in range(100):
                r = shared.solve(
                    [("a", {"anchor": "topleft", "size": (10, 10)})], (640, 480)
                )
                assert r[0][:2] == (0.0, 0.0)
        except Exception as exc:  # pragma: no cover - 失败路径
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
