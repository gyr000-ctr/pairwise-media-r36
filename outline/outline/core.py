"""outline 核心 API。

描边网格生成：沿轮廓外法线按给定宽度外扩，原网格保持不变。

- ``join='miter'``：斜接，相邻偏移边延长求交；
- ``join='round'``：圆角，凸角处用圆弧细分（矢高容差自适应），凹角仍按
  斜接交点修剪；当外扩量吞掉窄特征导致圆弧越界时，该角自动回退为斜接，
  保证外环始终简单、无自交。

``build`` 返回 ``(verts, indices)``：
``verts`` 为 ``(x, y)`` 浮点元组组成的元组，前段为外扩环、后段为与输入
一一对应的内环；``indices`` 为闭合三角形索引（首尾缝合的带状网格）。
"""

import math

__all__ = [
    "Outline",
    "OutlineError",
    "InvalidWidthError",
    "InvalidJoinError",
    "InvalidPolygonError",
    "NonFiniteError",
    "MITER",
    "ROUND",
]

MITER = "miter"
ROUND = "round"
_JOINS = (MITER, ROUND)

# 圆弧单段最大圆心角（约 0.7°）：每段弦矢高有界。90° 凸角取 128 段时，
# 单弧面积误差 ~w²·1.3e-7（对典型 w 相对带状面积优于 1e-6）；整体顶点 O(n)。
_MAX_ARC_STEP_ANGLE = math.pi / 256.0
# 单条圆弧的最大细分数（防护性上限）。
_MAX_ARC_STEPS = 1024
# 反向（180° 回头）边的防护阈值。
_HAIRPIN = 1e-12


class OutlineError(ValueError):
    """描边相关错误的基类（同时是 ValueError）。"""


class InvalidWidthError(OutlineError):
    """非法线宽。"""


class InvalidJoinError(OutlineError):
    """非法连接方式。"""


class InvalidPolygonError(OutlineError):
    """非法轮廓。"""


class NonFiniteError(OutlineError):
    """输入中存在 NaN/Inf 等非有限数值。"""


def _is_real(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_join(join):
    if not isinstance(join, str) or join not in _JOINS:
        raise InvalidJoinError(
            "join 必须是 %r 或 %r，收到 %r" % (MITER, ROUND, join)
        )
    return join


def _validate_width(width):
    if not _is_real(width):
        raise InvalidWidthError("width 必须是实数，收到 %r" % (width,))
    width = float(width)
    if not math.isfinite(width):
        raise InvalidWidthError("width 必须是有限数，收到 %r" % (width,))
    if width <= 0.0:
        raise InvalidWidthError("width 必须为正数，收到 %r" % (width,))
    return width


def _validate_poly(poly):
    """校验并拷贝轮廓，返回 list[(float, float)]。"""
    try:
        count = len(poly)
    except TypeError as exc:
        raise InvalidPolygonError("poly 必须是点序列") from exc
    if count < 3:
        raise InvalidPolygonError("轮廓至少需要 3 个顶点，收到 %d 个" % count)

    pts = []
    for index, point in enumerate(poly):
        try:
            if len(point) != 2:
                raise InvalidPolygonError(
                    "第 %d 个点不是二维点：%r" % (index, point)
                )
            x, y = point
        except (TypeError, KeyError, InvalidPolygonError) as exc:
            if isinstance(exc, InvalidPolygonError):
                raise
            raise InvalidPolygonError(
                "第 %d 个点不是二维点：%r" % (index, point)
            ) from exc
        if not (_is_real(x) and _is_real(y)):
            raise InvalidPolygonError(
                "第 %d 个点包含非数值坐标：%r" % (index, point)
            )
        x, y = float(x), float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise NonFiniteError(
                "第 %d 个点包含非有限坐标：%r" % (index, point)
            )
        pts.append((x, y))

    # 零长度边无法定义法线，直接归类为非法轮廓。
    for i in range(count):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % count]
        if ax == bx and ay == by:
            raise InvalidPolygonError("第 %d 条边长度为 0" % i)
    return pts


def _signed_area2(pts):
    """两倍有向面积（shoelace）。"""
    total = 0.0
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        total += ax * by - ay * bx
    return total


class _RingIndex:
    """简单多边形环的均匀网格索引，支持严格内部判定。

    落在环边上（边界容差内）的点不算内部——圆弧端点本就贴在斜接环的
    偏移边上，必须视为外部，才不会被误抑制。
    """

    def __init__(self, ring, tol):
        self.ring = ring
        self.tol = tol
        self.tol2 = tol * tol
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        self.minx, self.maxx = min(xs), max(xs)
        self.miny, self.maxy = min(ys), max(ys)
        span = max(self.maxx - self.minx, self.maxy - self.miny, 1.0)
        # 目标每格约容纳少量边；格长取包围盒跨度的 1/64，且不小于容差。
        self.cell = max(span / 64.0, tol * 8.0)
        self.grid = {}
        n = len(ring)
        for i in range(n):
            a = ring[i]
            b = ring[(i + 1) % n]
            if a == b:
                continue
            self._register(i, a, b)

    def _cell_keys(self, x, y):
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def _register(self, i, a, b):
        c0 = self._cell_keys(a[0], a[1])
        c1 = self._cell_keys(b[0], b[1])
        cx0, cx1 = sorted((c0[0], c1[0]))
        cy0, cy1 = sorted((c0[1], c1[1]))
        # 长边可能横跨多格；按包围盒登记，查询仍接近 O(1)。
        for gx in range(cx0, cx1 + 1):
            for gy in range(cy0, cy1 + 1):
                self.grid.setdefault((gx, gy), []).append(i)

    def strictly_inside(self, p):
        x, y = p
        if x < self.minx or x > self.maxx or y < self.miny or y > self.maxy:
            return False
        key = self._cell_keys(x, y)
        candidates = self.grid.get(key)
        if not candidates:
            return False
        ring = self.ring
        n = len(ring)
        inside = False
        for i in candidates:
            a = ring[i]
            b = ring[(i + 1) % n]
            ax, ay = a
            bx, by = b
            # 边界容差：贴边点视为外部。
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            if seg2 > 0.0:
                t = ((x - ax) * dx + (y - ay) * dy) / seg2
                if 0.0 <= t <= 1.0:
                    qx, qy = ax + t * dx, ay + t * dy
                    if (x - qx) ** 2 + (y - qy) ** 2 <= self.tol2:
                        return False
            # 射线法（水平向右）。
            if (ay > y) != (by > y):
                xint = ax + (y - ay) * dx / dy
                if xint > x:
                    inside = not inside
        return inside


class Outline:
    """描边网格生成器（无状态、可并发复用）。"""

    def build(self, poly, width, join="miter"):
        """生成闭合描边网格。

        返回 ``(verts, indices)``：``verts`` 前段为外扩环、后段为内环
        （输入顶点的浮点副本）；``indices`` 为三角形索引元组，带状网格
        在首尾处缝合。
        """
        join = _validate_join(join)
        width = _validate_width(width)
        pts = _validate_poly(poly)
        n = len(pts)

        signed2 = _signed_area2(pts)
        if signed2 == 0.0:
            raise InvalidPolygonError("轮廓退化为零面积，无法定义外法线")
        # CCW 取 +1（外法线在边右侧），CW 取 -1。
        orient = 1.0 if signed2 > 0.0 else -1.0

        # --- 边方向、单位外法线、转角（一次扫描） ---
        ux = [0.0] * n
        uy = [0.0] * n
        nx = [0.0] * n
        ny = [0.0] * n
        turn = [0.0] * n
        hypot = math.hypot
        atan2 = math.atan2
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            length = hypot(ex, ey)
            dx, dy = ex / length, ey / length
            ux[i] = dx
            uy[i] = dy
            # CCW 的外法线为右侧 (dy, -dx)；CW 取反。
            nx[i] = orient * dy
            ny[i] = orient * -dx

        for i in range(n):
            prev = (i - 1) % n
            cross = ux[prev] * uy[i] - uy[prev] * ux[i]
            dot = ux[prev] * ux[i] + uy[prev] * uy[i]
            turn[i] = atan2(cross, dot) * orient

        # --- 斜交点（凸角外延、凹角内收，凹角天然完成修剪） ---
        miter = [None] * n
        for i in range(n):
            prev = (i - 1) % n
            npx, npy = nx[prev], ny[prev]
            nnx, nny = nx[i], ny[i]
            denom = 1.0 + (npx * nnx + npy * nny)
            px, py = pts[i]
            if denom <= _HAIRPIN:
                miter[i] = (px + width * nnx, py + width * nny)
            else:
                sx = (npx + nnx) / denom
                sy = (npy + nny) / denom
                miter[i] = (px + width * sx, py + width * sy)

        clamped = self._fix_self_intersection(pts, ux, uy, miter)

        # --- 按顶点生成外环分块 ---
        if join == MITER:
            blocks = [[miter[i]] for i in range(n)]
        else:
            blocks = self._round_blocks(
                pts, nx, ny, turn, miter, width, clamped, ux, uy, orient
            )

        return self._assemble(pts, blocks, orient)

    @staticmethod
    def _fix_self_intersection(pts, ux, uy, miter):
        """修正凹角处偏移边塌陷导致的外环自交。

        偏移边 i 所在直线为 ``p_i + t*d_i``。若相邻斜交点在该直线上的
        参数区间反转，说明外扩量已吞掉该特征（自交）；将两个交点钳制到
        区间中点，保证外环简单、不再出现反向边。迭代到稳定为止。

        返回是否发生过钳制（round 模式据此决定是否做越界圆弧回退）。
        """
        n = len(pts)
        clamped = False
        for _ in range(n + 1):
            inverted = False
            for i in range(n):
                j = (i + 1) % n
                ax, ay = pts[i]
                dx, dy = ux[i], uy[i]
                mx, my = miter[i]
                jx, jy = miter[j]
                t0 = (mx - ax) * dx + (my - ay) * dy
                t1 = (jx - ax) * dx + (jy - ay) * dy
                if t1 <= t0:
                    inverted = True
                    clamped = True
                    mid = 0.5 * (t0 + t1)
                    clamp_point = (ax + mid * dx, ay + mid * dy)
                    miter[i] = clamp_point
                    miter[j] = clamp_point
            if not inverted:
                break
        return clamped

    @staticmethod
    def _round_blocks(pts, nx, ny, turn, miter, width, clamped, ux, uy, orient):
        """生成 round 模式的外环分块。

        凸角走面积误差自适应圆弧；凹角/共线走斜交点。若发生过特征塌陷，
        以钳制后的斜接环为参考简单边界：先对斜接点做一次廉价的内部预判，
        只有斜接点在外部的角才生成圆弧，再对弧点复核，任何越入边界内部的
        圆弧整体回退为斜接点，保证外环不产生自交且避免无用三角计算。

        ``orient`` 为 +1（CCW）或 -1（CW）；CW 时圆弧反向生成，使外环点序
        与轮廓绕序一致。
        """
        n = len(pts)
        sin = math.sin
        cos = math.cos
        ceil = math.ceil

        index = _RingIndex(list(miter), 1e-7) if clamped else None

        blocks = [None] * n
        for i in range(n):
            t = turn[i]
            if t <= 1e-12:
                blocks[i] = [miter[i]]
                continue

            # 塌陷场景的廉价预判：斜接点已在参考环内部则该角必回退。
            if index is not None and index.strictly_inside(miter[i]):
                blocks[i] = [miter[i]]
                continue

            prev = (i - 1) % n
            npx, npy = nx[prev], ny[prev]
            # 外法线旋转 90° 的切向（与朝向无关，纯几何量）。
            left_x, left_y = -npy, npx
            # 按单段圆心角上限分档；段数与转角成正比，整体 O(n) 顶点。
            k = int(max(1.0, ceil(t / _MAX_ARC_STEP_ANGLE)))
            if k > _MAX_ARC_STEPS:
                k = _MAX_ARC_STEPS
            px, py = pts[i]
            arc = []
            for step in range(k + 1):
                # orient=-1（CW）时 phi 取负，弧沿轮廓绕序反向扫过，
                # 起点仍是 prev 外法线方向，终点落到当前边外法线。
                phi = orient * t * step / k
                c = cos(phi)
                s = sin(phi)
                arc.append((
                    px + width * (c * npx + s * left_x),
                    py + width * (c * npy + s * left_y),
                ))
            if index is not None and any(
                    index.strictly_inside(p) for p in arc
            ):
                # 圆弧越入塌陷区：回退斜接，保持环简单。
                blocks[i] = [miter[i]]
            else:
                blocks[i] = arc
        return blocks

    @staticmethod
    def _assemble(pts, blocks, orient):
        """把外环分块与内环拼成顶点/三角形索引。

        ``orient`` 为 +1（CCW）或 -1（CW）；CW 时翻转三角形缠绕，保证索引
        朝向与轮廓朝向一致、有向面积符号正确。
        """
        n = len(pts)
        outer = []
        start = [0] * n
        length = [0] * n
        for i in range(n):
            start[i] = len(outer)
            block = blocks[i]
            length[i] = len(block)
            outer.extend(block)
        m = len(outer)

        verts = tuple(outer) + tuple(pts)

        def tri(a, b, c):
            return (a, b, c) if orient > 0.0 else (a, c, b)

        # 每个顶点处的角部扇形（圆角圆弧；斜接块只有一个点，无扇形）。
        fan = [
            tri(m + i, start[i] + k, start[i] + k + 1)
            for i in range(n)
            for k in range(length[i] - 1)
        ]
        # 每条边对应一个四边形（两个三角形），首尾取模缝合。
        quads = []
        for i in range(n):
            j = (i + 1) % n
            oi = start[i] + length[i] - 1
            oj = start[j]
            ii = m + i
            ij = m + j
            quads.append(tri(ii, oi, oj))
            quads.append(tri(ii, oj, ij))

        return verts, tuple(fan) + tuple(quads)

    def area(self, poly):
        """多边形面积（绝对值，shoelace）。"""
        pts = _validate_poly(poly)
        return abs(_signed_area2(pts)) * 0.5
