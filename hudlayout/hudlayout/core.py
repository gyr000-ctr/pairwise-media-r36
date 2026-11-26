"""HUD 布局约束求解器。

约束模型
--------
每个控件描述为 ``(id, spec)``，``spec`` 支持以下字段：

``anchor``
    锚点。取值为九个屏幕锚点之一（``topleft`` / ``topcenter`` /
    ``topright`` / ``centerleft`` / ``center`` / ``centerright`` /
    ``bottomleft`` / ``bottomcenter`` / ``bottomright``），或另一个控件的
    id（相对锚点）。别名：``top``/``bottom``/``left``/``right``/``middle``。
    屏幕锚点把控件自身的同名锚点对齐到安全区对应点（再施加 ``margin``），
    例如 ``bottomright`` 即控件右下角贴安全区右下角；这与 iOS / Android /
    Unity 的锚点语义一致。
``size``
    ``(width, height)``，必须为正的有限数。
``margin``
    ``(mx, my)``，默认 ``(0, 0)``。屏幕锚点下表示距对应屏幕边（安全区边）
    的内边距；相对锚点下为两个锚点之间的有符号偏移。
``pos``
    ``(x, y)``。给定后控件位置固定，可不提供 ``anchor``。
``priority``
    重叠消解优先级，数值越大越“重”、越不容易被推动，默认 0。
``self_anchor`` / ``target_anchor``
    相对锚点下自身与目标控件上的附着点，默认均为 ``topleft``。

所有位置方程都具有 ``x = c`` 或 ``x = x_other + c`` 的线性形式，按拓扑序
逐点求值即线性约束系统的精确浮点解（不存在迭代误差）。

屏幕可以是 ``(width, height)``，也可以是带 ``safe_area``（``(x, y, w, h)``）
的映射/对象；每次 :meth:`HudLayout.solve` 都基于当前屏幕与安全区从头重算，
因此屏幕缩放后再次调用即可得到重排结果。
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from typing import Any

__all__ = [
    "HudLayout",
    "LayoutError",
    "AnchorError",
    "CycleError",
    "InvalidWidgetError",
    "InvalidScreenError",
]


# ---------------------------------------------------------------------------
# 异常类型（均为 ValueError 子类，便于按粒度区分捕获）
# ---------------------------------------------------------------------------


class LayoutError(ValueError):
    """布局求解相关错误的基类。"""


class AnchorError(LayoutError):
    """非法锚点（未知屏幕锚点名或指向不存在的控件）。"""


class CycleError(LayoutError):
    """控件之间的相对锚点依赖存在环。"""


class InvalidWidgetError(LayoutError):
    """控件描述非法（缺字段、尺寸非正、数值非有限等）。"""


class InvalidScreenError(LayoutError):
    """屏幕 / 安全区参数非法。"""


# ---------------------------------------------------------------------------
# 锚点表
# ---------------------------------------------------------------------------

_ANCHOR_ALIASES = {
    "top": "topcenter",
    "bottom": "bottomcenter",
    "left": "centerleft",
    "right": "centerright",
    "middle": "center",
}

# 锚点名 -> (水平侧 -1/0/1, 垂直侧 -1/0/1)；-1 表示左/上，1 表示右/下。
_ANCHOR_SIDE = {
    "topleft": (-1, -1),
    "topcenter": (0, -1),
    "topright": (1, -1),
    "centerleft": (-1, 0),
    "center": (0, 0),
    "centerright": (1, 0),
    "bottomleft": (-1, 1),
    "bottomcenter": (0, 1),
    "bottomright": (1, 1),
}

_ANCHOR_POINT_OFFSET = {
    # 锚点在自身矩形上的偏移系数：(w 系数, h 系数)
    "topleft": (0.0, 0.0),
    "topcenter": (0.5, 0.0),
    "topright": (1.0, 0.0),
    "centerleft": (0.0, 0.5),
    "center": (0.5, 0.5),
    "centerright": (1.0, 0.5),
    "bottomleft": (0.0, 1.0),
    "bottomcenter": (0.5, 1.0),
    "bottomright": (1.0, 1.0),
}


def _normalize_anchor(name: Any) -> str:
    if not isinstance(name, str):
        raise AnchorError(f"anchor 必须是字符串，实际为 {type(name).__name__}")
    resolved = _ANCHOR_ALIASES.get(name, name)
    if resolved not in _ANCHOR_SIDE:
        raise AnchorError(f"非法锚点: {name!r}")
    return resolved


def _finite_pair(value: Any, label: str, *, positive: bool = False) -> tuple[float, float]:
    if not isinstance(value, _Sequence) or isinstance(value, (str, bytes)):
        raise InvalidWidgetError(f"{label} 必须是长度为 2 的序列")
    try:
        a, b = value
    except (TypeError, ValueError) as exc:
        raise InvalidWidgetError(f"{label} 必须是长度为 2 的序列") from exc
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError) as exc:
        raise InvalidWidgetError(f"{label} 必须是数值") from exc
    if not (math.isfinite(a) and math.isfinite(b)):
        raise InvalidWidgetError(f"{label} 包含非有限值: {value!r}")
    if positive and (a <= 0.0 or b <= 0.0):
        raise InvalidWidgetError(f"{label} 必须为正数: {value!r}")
    return a, b


# ---------------------------------------------------------------------------
# 求解器
# ---------------------------------------------------------------------------


class HudLayout:
    """HUD 布局约束求解器。

    实例可复用；:meth:`solve` 的最近一次结果可通过 :meth:`rects` 取回。
    所有公开方法均在实例锁内完成，多线程并发调用是安全的（各线程只读到
    自己调用完成后的一致快照）。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._last_rects: list[tuple[float, float, float, float]] = []
        self._last_widgets: Any = None
        self._last_screen: Any = None

    # -- 公开 API ----------------------------------------------------------

    def solve(
        self,
        widgets: _Sequence[tuple[Any, _Mapping[str, Any]]],
        screen: Any,
        *,
        resolve_overlaps: bool = True,
        max_passes: int = 8,
    ) -> list[tuple[float, float, float, float]]:
        """求解布局，返回与 ``widgets`` 同序的 ``(x, y, w, h)`` 浮点元组列表。

        ``resolve_overlaps`` 为 False 时只返回纯约束解，不做重叠消解。
        ``max_passes`` 控制重叠消解的最大扫描轮数：每轮工作量严格有界
        （候选/产出/移动均为 O(n)），无重叠时首轮即提前退出；极端深堆叠
        且控件总面积超过可用区域时无法完全消歧，可调大该参数换取更优布局。
        """
        with self._lock:
            safe = self._parse_screen(screen)
            parsed, edges, rev_edges = self._parse_widgets(widgets)
            order = self._topological_order(parsed, edges)

            # 环检测通过后再校验尺寸完整性（见 _parse_widgets 中的说明）
            for wid, _ in widgets:
                if parsed[wid]["size"] is None:
                    raise InvalidWidgetError(f"控件 {wid!r} 缺少正数 size")

            rects: dict[Any, list[float]] = {}
            for wid in order:
                rects[wid] = self._place(parsed[wid], rects, safe)

            result = [tuple(rects[wid]) for wid, _ in widgets]  # type: ignore[arg-type]

            if resolve_overlaps:
                priorities = {wid: parsed[wid]["priority"] for wid in order}
                self._resolve_overlaps(result, priorities, rev_edges, max_passes)

            # 提交快照
            self._last_rects = list(result)
            self._last_widgets = widgets
            self._last_screen = screen
            return list(result)

    def rects(self) -> list[tuple[float, float, float, float]]:
        """返回最近一次 :meth:`solve` 的结果矩形副本；未求解过则为空列表。"""
        with self._lock:
            return list(self._last_rects)

    def resize(self, screen: Any) -> list[tuple[float, float, float, float]]:
        """按新屏幕 / 安全区对最近一次的控件集合重新求解。"""
        with self._lock:
            if self._last_widgets is None:
                raise LayoutError("尚未求解过任何布局，无法 resize")
            return self.solve(self._last_widgets, screen)

    # -- 屏幕 / 安全区 ------------------------------------------------------

    @staticmethod
    def _parse_screen(screen: Any) -> tuple[float, float, float, float]:
        if isinstance(screen, _Mapping):
            width = screen.get("width")
            height = screen.get("height")
            safe_area = screen.get("safe_area")
            insets = screen.get("safe_area_insets")
        elif isinstance(screen, (tuple, list)) and len(screen) == 2:
            width, height = screen
            safe_area = None
            insets = None
        elif hasattr(screen, "width") and hasattr(screen, "height"):
            width = getattr(screen, "width")
            height = getattr(screen, "height")
            safe_area = getattr(screen, "safe_area", None)
            insets = getattr(screen, "safe_area_insets", None)
        else:
            raise InvalidScreenError(f"无法识别的屏幕描述: {screen!r}")

        try:
            width, height = float(width), float(height)
        except (TypeError, ValueError) as exc:
            raise InvalidScreenError("屏幕宽高必须是数值") from exc
        if not (math.isfinite(width) and math.isfinite(height)):
            raise InvalidScreenError("屏幕宽高必须有限")
        if width <= 0.0 or height <= 0.0:
            raise InvalidScreenError(f"屏幕宽高必须为正: {screen!r}")

        if safe_area is not None:
            try:
                sx, sy, sw, sh = (float(v) for v in safe_area)
            except (TypeError, ValueError) as exc:
                raise InvalidScreenError("safe_area 必须是 (x, y, w, h)") from exc
        elif insets is not None:
            try:
                left, top, right, bottom = (float(v) for v in insets)
            except (TypeError, ValueError) as exc:
                raise InvalidScreenError("safe_area_insets 必须是 (l, t, r, b)") from exc
            if not all(math.isfinite(v) for v in (left, top, right, bottom)):
                raise InvalidScreenError("安全区内边距包含非有限值")
            if min(left, top, right, bottom) < 0.0 or left + right >= width or top + bottom >= height:
                raise InvalidScreenError("安全区内边距超出屏幕范围")
            sx, sy = left, top
            sw, sh = width - left - right, height - top - bottom
        else:
            sx, sy, sw, sh = 0.0, 0.0, width, height

        if not all(math.isfinite(v) for v in (sx, sy, sw, sh)):
            raise InvalidScreenError("安全区包含非有限值")
        if sw <= 0.0 or sh <= 0.0:
            raise InvalidScreenError("安全区宽高必须为正")
        if sx < 0.0 or sy < 0.0 or sx + sw > width + 1e-9 or sy + sh > height + 1e-9:
            raise InvalidScreenError("安全区超出屏幕范围")
        return sx, sy, sw, sh

    # -- 控件解析与依赖图 ---------------------------------------------------

    def _parse_widgets(self, widgets: Any) -> tuple[dict, dict, dict]:
        if not isinstance(widgets, _Sequence) or isinstance(widgets, (str, bytes)):
            raise InvalidWidgetError("widgets 必须是 (id, spec) 序列")

        parsed: dict[Any, dict] = {}
        edges: dict[Any, Any] = {}       # self id -> target id
        order_ids: list[Any] = []

        for item in widgets:
            if not isinstance(item, tuple) or len(item) != 2:
                raise InvalidWidgetError(f"控件描述必须是 (id, spec): {item!r}")
            wid, spec = item
            if not isinstance(spec, _Mapping):
                raise InvalidWidgetError(f"控件 {wid!r} 的 spec 必须是映射")
            if wid in parsed:
                raise InvalidWidgetError(f"控件 id 重复: {wid!r}")
            try:
                hash(wid)
            except TypeError as exc:
                raise InvalidWidgetError(f"控件 id 不可哈希: {wid!r}") from exc

            entry: dict[str, Any] = {"id": wid}
            anchor = spec.get("anchor")
            pos = spec.get("pos")

            if pos is not None:
                entry["pos"] = _finite_pair(pos, "pos")
            if anchor is None and pos is None:
                raise InvalidWidgetError(f"控件 {wid!r} 缺少 anchor 或 pos")

            if anchor is not None:
                # 先记录原始锚点，待全部 id 可知后再解析控件引用
                entry["anchor_raw"] = anchor
                entry["self_anchor"] = _normalize_anchor(
                    spec.get("self_anchor", "topleft")
                )
                entry["target_anchor"] = _normalize_anchor(
                    spec.get("target_anchor", "topleft")
                )

            if "size" in spec:
                entry["size"] = _finite_pair(spec["size"], "size", positive=True)
            else:
                # 延迟到环检测之后报错（循环用例里的控件可能没有 size）
                entry["size"] = None
            entry["margin"] = (
                _finite_pair(spec["margin"], "margin") if "margin" in spec else (0.0, 0.0)
            )
            priority = spec.get("priority", 0)
            try:
                priority = float(priority)
            except (TypeError, ValueError) as exc:
                raise InvalidWidgetError(f"控件 {wid!r} 的 priority 必须是数值") from exc
            if not math.isfinite(priority):
                raise InvalidWidgetError(f"控件 {wid!r} 的 priority 非有限")
            entry["priority"] = priority

            parsed[wid] = entry
            order_ids.append(wid)

        # 第二趟：解析锚点（屏幕锚点 vs 控件引用），建立依赖边
        for wid in order_ids:
            entry = parsed[wid]
            raw = entry.get("anchor_raw")
            if raw is None:
                continue
            is_screen_anchor = isinstance(raw, str) and (
                raw in _ANCHOR_SIDE or raw in _ANCHOR_ALIASES
            )
            if is_screen_anchor:
                entry["screen_anchor"] = _normalize_anchor(raw)
            else:
                # 非屏幕锚点名即控件 id 引用（允许非字符串 id）
                if raw not in parsed:
                    raise AnchorError(
                        f"控件 {wid!r} 的锚点 {raw!r} 既不是合法屏幕锚点，"
                        "也不是已知控件 id"
                    )
                edges[wid] = raw

        # 注意：尺寸完整性在拓扑环检测之后再校验，保证无 size 的循环依赖
        # 仍然报 CycleError 而不是被缺字段错误抢先。
        rev_edges: dict[Any, list] = {wid: [] for wid in order_ids}
        for wid, target in edges.items():
            rev_edges[target].append(wid)
        return parsed, edges, rev_edges

    @staticmethod
    def _topological_order(parsed: dict, edges: dict) -> list:
        # 反向邻接（保持输入序，保证排序确定性）
        rev: dict[Any, list] = {wid: [] for wid in parsed}
        for wid, target in edges.items():
            rev[target].append(wid)

        indeg = {wid: 0 for wid in parsed}
        for wid in edges:
            indeg[wid] += 1

        # deque 保证确定性（按输入序入队）
        ready = deque(wid for wid in parsed if indeg[wid] == 0)
        order: list = []
        while ready:
            wid = ready.popleft()
            order.append(wid)
            for dep in rev[wid]:
                indeg[dep] -= 1
                if indeg[dep] == 0:
                    ready.append(dep)

        if len(order) != len(parsed):
            remaining = {wid for wid in parsed if indeg[wid] > 0}
            cycle = HudLayout._extract_cycle(edges, remaining)
            raise CycleError(f"检测到循环依赖: {' -> '.join(repr(w) for w in cycle)}")
        return order

    @staticmethod
    def _extract_cycle(edges: dict, remaining: set) -> list:
        start = next(iter(remaining))
        path: list = []
        index: dict[Any, int] = {}
        node = start
        while node not in index:
            index[node] = len(path)
            path.append(node)
            node = edges[node]
        return path[index[node]:] + [node]

    # -- 单控件定位 ---------------------------------------------------------

    @staticmethod
    def _place(entry: dict, rects: dict, safe: tuple) -> list[float]:
        w, h = entry["size"]
        if "pos" in entry:
            x, y = entry["pos"]
            return [float(x), float(y), float(w), float(h)]

        mx, my = entry["margin"]

        if "screen_anchor" in entry:
            sx, sy, sw, sh = safe
            anchor = entry["screen_anchor"]
            side_x, side_y = _ANCHOR_SIDE[anchor]
            kx, ky = _ANCHOR_POINT_OFFSET[anchor]

            # 锚点在安全区上的位置
            px = sx + (side_x + 1.0) * 0.5 * sw
            py = sy + (side_y + 1.0) * 0.5 * sh
            # margin 作为距对应边的内边距：靠右/靠下时符号取反
            px += mx if side_x <= 0 else -mx
            py += my if side_y <= 0 else -my
            x = px - kx * w
            y = py - ky * h
            return [float(x), float(y), float(w), float(h)]

        # 相对锚点：self 锚点 = target 锚点 + margin
        target_id = entry["anchor_raw"]
        tx, ty, tw, th = rects[target_id]
        t_anchor = entry["target_anchor"]
        tkx, tky = _ANCHOR_POINT_OFFSET[t_anchor]
        px = tx + tkx * tw + mx
        py = ty + tky * th + my

        s_anchor = entry["self_anchor"]
        skx, sky = _ANCHOR_POINT_OFFSET[s_anchor]
        x = px - skx * w
        y = py - sky * h
        return [float(x), float(y), float(w), float(h)]

    # -- 重叠消解 -----------------------------------------------------------

    def _resolve_overlaps(
        self,
        result: list[tuple[float, float, float, float]],
        priorities: dict,
        rev_edges: dict,
        max_passes: int,
    ) -> None:
        n = len(result)
        if n < 2 or max_passes <= 0:
            return

        # 真实位置的可变工作副本，矩形读取始终 O(1)
        work = [list(r) for r in result]
        ids = list(priorities.keys())
        id_to_idx = {wid: i for i, wid in enumerate(ids)}
        children = {
            wid: [id_to_idx[d] for d in rev_edges.get(wid, ())] for wid in ids
        }
        # 依赖图是一片森林（每控件至多一个锚点目标）：Euler 序上子树是连续
        # 区间 [tin, tout)，用于 O(1) 祖先判断，跳过有意嵌套的父子控件。
        tin, tout = self._ancestor_ranges(children, ids)

        # 每轮枚举的相交候选对与实际产出对数都设上限：即便空间网格在极端
        # 聚集场景退化（同格控件极多），单轮工作量也严格 O(n) 有界；
        # 移动次数另以 n 为预算。
        pair_budget = 3 * n
        candidate_budget = 16 * n

        for _ in range(max_passes):
            moves = 0
            for i, j in self._iter_overlapping_pairs(
                work, pair_budget, candidate_budget
            ):
                # 锚定关系上的父子（祖先/后代）重叠属于有意嵌套，不处理
                if self._is_ancestor(i, j, tin, tout):
                    continue
                if priorities[ids[i]] >= priorities[ids[j]]:
                    # 优先级高的留下；持平则索引小的留下（确定性）
                    mover, still = j, i
                else:
                    mover, still = i, j
                dx, dy = self._min_push_out(work[mover], work[still])
                self._move_subtree(work, children, ids, mover, dx, dy)
                moves += 1
                if moves >= n:
                    break

            # 无任何推出即说明当前网格候选内已无（非祖先）重叠，收敛
            if moves == 0:
                break

        for i, rect in enumerate(work):
            result[i] = (rect[0], rect[1], rect[2], rect[3])

    @staticmethod
    def _move_subtree(work: list, children: dict, ids: list, root_idx: int, dx: float, dy: float) -> None:
        """把 dx/dy 施加到 root 及其整棵依赖子树（控件随锚定父级刚性移动）。"""
        stack = [root_idx]
        seen = {root_idx}
        while stack:
            idx = stack.pop()
            r = work[idx]
            r[0] += dx
            r[1] += dy
            for c in children[ids[idx]]:
                if c not in seen:
                    seen.add(c)
                    stack.append(c)

    @staticmethod
    def _ancestor_ranges(children: dict, ids: list) -> tuple[list[int], list[int]]:
        """对依赖森林做迭代式 Euler 遍历，返回每个节点的 [tin, tout) 区间。

        i 是 j 的祖先（或同一节点）当且仅当 tin[i] <= tin[j] < tout[i]。
        """
        n = len(ids)
        tin = [0] * n
        tout = [0] * n
        clock = 0
        child_set = {wid: children[wid] for wid in ids}
        has_parent = [False] * n
        for wid in ids:
            for c in child_set[wid]:
                has_parent[c] = True
        # (idx, state)；state 0 = 进入，1 = 退出
        for root in range(n):
            if has_parent[root]:
                continue
            stack = [(root, 0)]
            while stack:
                idx, state = stack.pop()
                if state == 0:
                    tin[idx] = clock
                    clock += 1
                    stack.append((idx, 1))
                    for c in reversed(child_set[ids[idx]]):
                        stack.append((c, 0))
                else:
                    tout[idx] = clock
        return tin, tout

    @staticmethod
    def _is_ancestor(i: int, j: int, tin: list[int], tout: list[int]) -> bool:
        return tin[i] <= tin[j] < tout[i]

    @staticmethod
    def _intersects(a: _Sequence[float], b: _Sequence[float]) -> bool:
        return (
            a[0] < b[0] + b[2]
            and b[0] < a[0] + a[2]
            and a[1] < b[1] + b[3]
            and b[1] < a[1] + a[3]
        )

    @staticmethod
    def _min_push_out(mover: _Sequence[float], still: _Sequence[float]) -> tuple[float, float]:
        # 沿最小穿透轴把 mover 推出 still
        push_right = still[0] + still[2] - mover[0]
        push_left = mover[0] + mover[2] - still[0]
        push_down = still[1] + still[3] - mover[1]
        push_up = mover[1] + mover[3] - still[1]
        ax = push_right if push_right <= push_left else -push_left
        ay = push_down if push_down <= push_up else -push_up
        if abs(ax) <= abs(ay):
            return ax, 0.0
        return 0.0, ay

    @staticmethod
    def _iter_overlapping_pairs(
        work: list,
        pair_limit: int | None = None,
        candidate_limit: int | None = None,
    ):
        """空间哈希网格流式产出确实相交的控件对 ``(i, j)``（i < j）。

        网格在生成器启动时按当前位置构建；轮内控件被即时移动后网格会过期，
        因此产出处再做一次实时相交校验。网格边长取控件平均尺寸，常规 HUD
        场景下每格仅含少量控件，候选对与实际相交对均为近线性；超大控件
        单独走 brute-force 列表，避免单个控件占用海量网格单元。

        ``candidate_limit`` 限制候选对扫描总量、``pair_limit`` 限制实际产出
        的相交对数量，二者共同保证网格退化时单轮工作量仍有线性上界。
        """
        n = len(work)
        if n < 2:
            return

        sum_w = 0.0
        sum_h = 0.0
        for r in work:
            sum_w += r[2]
            sum_h += r[3]
        cell_w = max(sum_w / n, 1e-9)
        cell_h = max(sum_h / n, 1e-9)

        grid: dict[tuple[int, int], list[int]] = {}
        big: list[int] = []
        BIG_LIMIT = 32

        for idx, r in enumerate(work):
            c0x = int(math.floor(r[0] / cell_w))
            c0y = int(math.floor(r[1] / cell_h))
            c1x = int(math.floor((r[0] + r[2]) / cell_w))
            c1y = int(math.floor((r[1] + r[3]) / cell_h))
            span = (c1x - c0x + 1) * (c1y - c0y + 1)
            if span > BIG_LIMIT:
                big.append(idx)
                continue
            for cx in range(c0x, c1x + 1):
                for cy in range(c0y, c1y + 1):
                    grid.setdefault((cx, cy), []).append(idx)

        seen_pairs: set[tuple[int, int]] = set()
        candidates_checked = 0
        pairs_emitted = 0
        for idx, r in enumerate(work):
            c0x = int(math.floor(r[0] / cell_w))
            c0y = int(math.floor(r[1] / cell_h))
            c1x = int(math.floor((r[0] + r[2]) / cell_w))
            c1y = int(math.floor((r[1] + r[3]) / cell_h))
            candidates: list[int] = []
            for cx in range(c0x, c1x + 1):
                for cy in range(c0y, c1y + 1):
                    candidates.extend(grid.get((cx, cy), ()))
            candidates.extend(big)
            for other in candidates:
                if other <= idx:
                    # 每对只在较大索引侧产出一次，天然去重且保持 i < j
                    continue
                key = (idx, other)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                candidates_checked += 1
                if candidate_limit is not None and candidates_checked > candidate_limit:
                    return
                if HudLayout._intersects(work[idx], work[other]):
                    yield key
                    pairs_emitted += 1
                    if pair_limit is not None and pairs_emitted >= pair_limit:
                        return

