"""particles 核心 API。

提供两个能力：

* :class:`EmissionCurve` —— 归一化时间 ``[0, 1]`` 上的分段三次贝塞尔发射速率
  曲线，积分使用贝塞尔多项式的解析原函数计算；
* :class:`Particles` —— 固定容量粒子池。槽位数组、空闲链表与到期最小堆均在
  底层池构造时一次性预分配，稳态下 ``emit`` / ``update`` 只在预分配存储上原地
  复用，不再向堆申请任何容器或粒子对象（零分配复用）。

同一 ``(cap, seed)`` 构造出的 :class:`Particles` 共享同一个底层池（构造即以
``seed`` 重新播种随机源），不同 ``seed`` 之间完全隔离；每次 :meth:`emit` 开启
一个新的发射代（原地回收全部槽位后重新发射，代次与复用计数持续累积），同种子
同调用序列可复现。

所有公开方法均通过 :class:`threading.RLock` 保护，可跨线程调用。
"""

import math
import random
import threading

__all__ = [
    "Particles",
    "EmissionCurve",
    "ParticleError",
    "CapacityError",
    "RateError",
    "TimeStepError",
    "CurveError",
]


class ParticleError(ValueError):
    """粒子系统参数错误基类（同时是 ``ValueError`` 的子类）。"""


class CapacityError(ParticleError):
    """容量参数非法。"""


class RateError(ParticleError):
    """发射速率非法（类型错误、负数、NaN/Inf）。"""


class TimeStepError(ParticleError):
    """时间步长非法（负数、NaN/Inf）。"""


class CurveError(ParticleError):
    """贝塞尔曲线参数非法。"""


def _is_finite_number(x):
    # 直接类型判定（int/float 为内置最终类型），避免元组形式 isinstance
    # 触发解释器 abc 子类缓存的惰性增长。
    t = type(x)
    if t is int:
        return True
    if t is float:
        return math.isfinite(x)
    return False


def _bezier_coeffs(v0, v1, v2, v3):
    """三次贝塞尔 B(u)=sum ci*u^i 的幂基系数。"""
    return (
        v0,
        -3.0 * v0 + 3.0 * v1,
        3.0 * v0 - 6.0 * v1 + 3.0 * v2,
        -v0 + 3.0 * v1 - 3.0 * v2 + v3,
    )


class EmissionCurve:
    """分段三次贝塞尔发射速率曲线。

    参数 ``segments`` 为若干段的序列，每段包含 4 个控制点 ``(x, y)``：

    * ``x`` 为归一化时间，必须落在 ``[0, 1]`` 且段内单调不减；
    * 各段首尾相接（前一段的 p3 等于后一段的 p0），首点 x=0、末点 x=1；
    * ``y`` 为发射速率，必须非负且有限。

    曲线是参数方程 ``t = B_x(u)``、``rate = B_y(u)``，因此对**时间**的解析
    积分为 ``∫ B_y(u)·B_x'(u) du``（五次多项式，系数在构造时一次性算出），
    与对 :meth:`rate` 做高精度时间求积的结果一致。拒绝抽样使用同一被积函数，
    故发射期望严格等于 ``dt * 解析积分``。

    每段预处理结果存为一个定长元组（构造后只读），热路径不产生任何新容器。
    段元组布局：

    ``(x0, x3, cx0..cx3, cy0..cy3, d0..d2, env, aj0..aj5, full)``
    """

    # 段元组下标
    _X0, _X3 = 0, 1
    _CX = 2          # cx0..cx3 起
    _CY = 6          # cy0..cy3 起
    _D = 10          # Bx' 系数 d0..d2 起
    _ENV = 13        # ymax * span
    _AJ = 14         # 原函数系数 aj0..aj5 起
    _FULL = 20       # 段完整积分

    def __init__(self, segments):
        if not isinstance(segments, (list, tuple)) or not segments:
            raise CurveError("segments 必须是非空的控制点序列")

        norm = []
        for si, seg in enumerate(segments):
            if not isinstance(seg, (list, tuple)) or len(seg) != 4:
                raise CurveError("第 %d 段必须包含恰好 4 个控制点" % si)
            pts = []
            for pt in seg:
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    raise CurveError("控制点必须是 (x, y) 二元组")
                x, y = pt
                if not (_is_finite_number(x) and _is_finite_number(y)):
                    raise CurveError("控制点坐标必须是有限数值")
                if y < 0.0:
                    raise CurveError("速率控制点 y 不能为负")
                pts.append((float(x), float(y)))
            x0 = pts[0][0]
            if not (0.0 <= x0 <= 1.0):
                raise CurveError("控制点 x 必须落在 [0, 1]")
            for j in range(1, 4):
                if not (0.0 <= pts[j][0] <= 1.0):
                    raise CurveError("控制点 x 必须落在 [0, 1]")
                if pts[j][0] < x0:
                    raise CurveError("段内 x 必须单调不减")
            if pts[3][0] <= x0:
                raise CurveError("第 %d 段时间宽度为 0" % si)
            if norm and pts[0] != norm[-1][3]:
                raise CurveError("相邻段必须首尾相接")
            norm.append(tuple(pts))

        if norm[0][0][0] != 0.0 or norm[-1][3][0] != 1.0:
            raise CurveError("曲线必须覆盖归一化时间 [0, 1]")

        self._segments = tuple(norm)
        self._seg = []
        self._bounds = []
        self._total = 0.0
        for pts in norm:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
            cx = _bezier_coeffs(x0, x1, x2, x3)
            cy = _bezier_coeffs(y0, y1, y2, y3)
            # Bx'(u) = d0 + d1 u + d2 u^2
            d0 = cx[1]
            d1 = 2.0 * cx[2]
            d2 = 3.0 * cx[3]
            # 卷积：cj[n] = sum_{i+j=n} d_i * cy_j，n=0..5
            cj0 = d0 * cy[0]
            cj1 = d0 * cy[1] + d1 * cy[0]
            cj2 = d0 * cy[2] + d1 * cy[1] + d2 * cy[0]
            cj3 = d0 * cy[3] + d1 * cy[2] + d2 * cy[1]
            cj4 = d1 * cy[3] + d2 * cy[2]
            cj5 = d2 * cy[3]
            # 原函数 A(u) = sum aj[k] u^(k+1)
            aj = (cj0, cj1 / 2.0, cj2 / 3.0, cj3 / 4.0, cj4 / 5.0, cj5 / 6.0)
            full = aj[0] + aj[1] + aj[2] + aj[3] + aj[4] + aj[5]
            span = x3 - x0
            env = max(y0, y1, y2, y3) * span
            self._seg.append(
                (x0, x3,
                 cx[0], cx[1], cx[2], cx[3],
                 cy[0], cy[1], cy[2], cy[3],
                 d0, d1, d2,
                 env,
                 aj[0], aj[1], aj[2], aj[3], aj[4], aj[5],
                 full)
            )
            self._bounds.append(x3)
            self._total += full
        self._seg = tuple(self._seg)
        self._bounds = tuple(self._bounds)
        self._nseg = len(self._seg)

    @staticmethod
    def _eval3(c, base, u):
        return ((c[base + 3] * u + c[base + 2]) * u + c[base + 1]) * u + c[base]

    def _locate(self, t):
        """返回包含归一化时刻 t 的段下标（线性扫描，段数通常很少）。"""
        bounds = self._bounds
        i = 0
        n = self._nseg
        while i < n:
            if t <= bounds[i]:
                return i
            i += 1
        return n - 1

    def _solve_u(self, seg, t):
        """在段内二分求解 B_x(u) = t，返回 u。"""
        x0 = seg[self._X0]
        x3 = seg[self._X3]
        if t <= x0:
            return 0.0
        if t >= x3:
            return 1.0
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) * 0.5
            if self._eval3(seg, self._CX, mid) < t:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5

    def _antideriv(self, seg, u):
        b = self._AJ
        return ((((seg[b + 5] * u + seg[b + 4]) * u + seg[b + 3]) * u
                 + seg[b + 2]) * u + seg[b + 1]) * u * u + seg[b] * u

    def rate(self, t):
        """归一化时刻 ``t in [0, 1]`` 上的发射速率。"""
        if not _is_finite_number(t) or not (0.0 <= float(t) <= 1.0):
            raise CurveError("t 必须是 [0, 1] 内的有限数值")
        seg = self._seg[self._locate(float(t))]
        u = self._solve_u(seg, float(t))
        return self._eval3(seg, self._CY, u)

    def integrate(self, t0=0.0, t1=1.0):
        """速率曲线在归一化时间区间 ``[t0, t1]`` 上的解析积分。"""
        if not (_is_finite_number(t0) and _is_finite_number(t1)):
            raise CurveError("积分端点必须是有限数值")
        t0, t1 = float(t0), float(t1)
        if not (0.0 <= t0 <= 1.0 and 0.0 <= t1 <= 1.0) or t1 < t0:
            raise CurveError("积分端点必须满足 0 <= t0 <= t1 <= 1")
        if t0 == t1:
            return 0.0
        i0 = self._locate(t0)
        i1 = self._locate(t1)
        s0 = self._seg[i0]
        u0 = self._solve_u(s0, t0)
        if i0 == i1:
            u1 = self._solve_u(s0, t1)
            return self._antideriv(s0, u1) - self._antideriv(s0, u0)
        total = s0[self._FULL] - self._antideriv(s0, u0)
        i = i0 + 1
        while i < i1:
            total += self._seg[i][self._FULL]
            i += 1
        s1 = self._seg[i1]
        total += self._antideriv(s1, self._solve_u(s1, t1))
        return total

    def total_integral(self):
        """整条曲线 ``[0, 1]`` 上的解析积分。"""
        return self._total


class _Pool:
    """固定容量粒子池：所有存储在构造时一次性预分配，热路径零分配。"""

    # λ 低于该阈值用精确 Knuth 泊松采样；高于则用正态近似（λ 大时两者无偏）
    _POISSON_GAUSS_THRESHOLD = 100.0

    def __init__(self, cap, seed):
        self.cap = cap
        self.seed = seed
        self.rng = random.Random(seed)
        self.lock = threading.RLock()

        # 槽位数据（定长，构造后不再伸缩）
        self.slot_epoch = [-1] * cap
        self.used = bytearray(cap)
        self.expiry = [0.0] * cap
        self.px = [0.0] * cap
        self.py = [0.0] * cap
        # 定长空闲栈：长度恒为 cap，用 free_top 指针压入/弹出，
        # 列表本身绝不伸缩，因此热路径零分配。
        self.free = [0] * cap
        self.free_top = 0
        # 新槽位的高水位游标（仅在空闲栈为空时顺序开新槽）
        self.cursor = 0
        # 发射代编号：每开始一代自增，槽位记录所属代次实现 O(1) 代际切换
        self.epoch = 0

        # 到期最小堆（位置索引的平行数组）：三个等长定长数组的同一位置构成一个
        # 堆节点，下沉/上浮时同步交换三数组元素，没有 entry-id 间接层。因此
        # hsize=0 即完整重置，新代次从空堆开始不会读到旧数据，热路径零分配。
        self.h_exp = [0.0] * cap
        self.h_epoch = [0] * cap
        self.h_slot = [-1] * cap
        self.hsize = 0

        self.live_count = 0
        self.reuse = 0
        self.peak = 0
        self.now = 0.0
        self.life_lo = 1.0
        self.life_hi = 1.0

    def reseed(self):
        """以构造种子重新播种随机源（原地 seed，不新建 Random 对象）。"""
        self.rng.seed(self.seed)

    def begin_generation(self):
        """原地开启新发射代，O(1)：只移动游标/代次指针，不扫描槽位数组。

        历史槽位数据无需清理，代次不匹配即视为已回收；复用/峰值统计持续累积。
        """
        self.epoch += 1
        self.free_top = 0
        self.cursor = 0
        self.hsize = 0
        self.live_count = 0
        self.now = 0.0

    # ------------------------------------------------------------------ #
    # 预分配最小堆（位置索引，无 entry-id 间接层）
    # ------------------------------------------------------------------ #
    def _heap_push(self, exp, epoch, slot):
        h_exp = self.h_exp
        h_epoch = self.h_epoch
        h_slot = self.h_slot
        i = self.hsize
        self.hsize = i + 1
        h_exp[i] = exp
        h_epoch[i] = epoch
        h_slot[i] = slot
        while i > 0:
            p = (i - 1) >> 1
            if h_exp[p] <= h_exp[i]:
                break
            h_exp[p], h_exp[i] = h_exp[i], h_exp[p]
            h_epoch[p], h_epoch[i] = h_epoch[i], h_epoch[p]
            h_slot[p], h_slot[i] = h_slot[i], h_slot[p]
            i = p

    def _reap(self):
        """按到期顺序回收所有已到期粒子，槽位归还空闲栈。

        下沉完全内联（不构造返回元组），并以代次守卫丢弃过期条目。
        """
        h_exp = self.h_exp
        h_epoch = self.h_epoch
        h_slot = self.h_slot
        slot_epoch = self.slot_epoch
        free = self.free
        while self.hsize and h_exp[0] <= self.now:
            n = self.hsize - 1
            epoch = h_epoch[0]
            slot = h_slot[0]
            # 末节点搬到根
            h_exp[0] = h_exp[n]
            h_epoch[0] = h_epoch[n]
            h_slot[0] = h_slot[n]
            i = 0
            while True:
                left = i * 2 + 1
                if left >= n:
                    break
                right = left + 1
                child = left
                if right < n and h_exp[right] < h_exp[left]:
                    child = right
                if h_exp[i] <= h_exp[child]:
                    break
                h_exp[i], h_exp[child] = h_exp[child], h_exp[i]
                h_epoch[i], h_epoch[child] = h_epoch[child], h_epoch[i]
                h_slot[i], h_slot[child] = h_slot[child], h_slot[i]
                i = child
            self.hsize = n
            # 代次守卫：仅当条目仍对应当前代次的存活粒子时回收
            if slot_epoch[slot] == epoch:
                slot_epoch[slot] = -1
                top = self.free_top
                free[top] = slot
                self.free_top = top + 1
                self.live_count -= 1

    # ------------------------------------------------------------------ #
    # 随机数（泊松采样）
    # ------------------------------------------------------------------ #
    def _poisson(self, lam):
        if lam <= 0.0:
            return 0
        if lam < self._POISSON_GAUSS_THRESHOLD:
            target = math.exp(-lam)
            k = 0
            p = 1.0
            while p > target:
                k += 1
                p *= self.rng.random()
            return k - 1
        # 大 λ：正态近似。N(λ,λ) 取整并截断到非负；λ>=100 时截断概率可忽略，
        # 期望与 λ 一致，且避免 Knuth 在大 λ 下的 O(λ) 开销与下溢。
        value = int(round(self.rng.gauss(lam, math.sqrt(lam))))
        return value if value > 0 else 0

    def _spawn(self, lifetime):
        """占用一个空闲槽位；池满返回 False。

        优先复用本代已回收槽位（空闲栈，必为历史用过的槽位），其次开启高水位
        新槽位。仅当槽位在更早代次曾被占用时才计入复用次数。
        """
        if self.free_top > 0:
            self.free_top -= 1
            slot = self.free[self.free_top]
            self.reuse += 1
        else:
            if self.cursor >= self.cap:
                return False
            slot = self.cursor
            self.cursor += 1
            if self.used[slot]:
                self.reuse += 1
        self.used[slot] = 1
        self.slot_epoch[slot] = self.epoch
        self.expiry[slot] = self.now + lifetime
        self.px[slot] = self.rng.random()
        self.py[slot] = self.rng.random()
        self._heap_push(self.expiry[slot], self.epoch, slot)
        self.live_count += 1
        if self.live_count > self.peak:
            self.peak = self.live_count
        return True


class Particles:
    """固定容量粒子池句柄。

    同一 ``(cap, seed)`` 构造出的句柄共享同一个底层 :class:`_Pool`，构造时以
    ``seed`` 重新播种随机源；不同 ``seed`` 之间完全隔离。每次 :meth:`emit`
    开启一个新发射代（原地回收后重新发射），槽位只复用不新分配，池满时新粒子
    被丢弃，``peak_allocated()`` 永远不超过 ``cap``；:meth:`update` 由最小堆按
    到期时间先后回收粒子，槽位归还空闲链表，不泄漏。
    """

    # (cap, seed) -> _Pool
    _registry = {}
    _registry_lock = threading.Lock()

    def __init__(self, cap, seed):
        if type(cap) is not int or cap <= 0:
            raise CapacityError("cap 必须是正整数，收到 %r" % (cap,))
        key = (cap, seed)
        with Particles._registry_lock:
            pool = Particles._registry.get(key)
            if pool is None:
                pool = _Pool(cap, seed)
                Particles._registry[key] = pool
        self._pool = pool
        # 同 (cap, seed) 的新句柄：重新播种，保证同种子同序列可复现
        pool.reseed()

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_lifetime(pool, lifetime):
        """校验 lifetime 并把 (lo, hi) 写到 pool 上，热路径不产生元组。

        lifetime 支持正有限标量或 (lo, hi) 区间（均匀抽样）。
        """
        lt = type(lifetime)
        if lt is tuple or lt is list:
            if len(lifetime) != 2:
                raise ParticleError("lifetime 区间必须是 (lo, hi) 二元组")
            lo, hi = lifetime
            if not (_is_finite_number(lo) and _is_finite_number(hi)):
                raise ParticleError("lifetime 区间必须是有限数值")
            if lo <= 0.0 or hi < lo:
                raise ParticleError("lifetime 区间必须满足 0 < lo <= hi")
            pool.life_lo = float(lo)
            pool.life_hi = float(hi)
            return
        if not _is_finite_number(lifetime) or lifetime <= 0.0:
            raise ParticleError("lifetime 必须是正的有限数值")
        v = float(lifetime)
        pool.life_lo = v
        pool.life_hi = v

    def emit(self, rate, dt, lifetime=1.0):
        """发射粒子，返回本次实际占用槽位生成的粒子数。

        每次调用开启一个新发射代。``rate`` 可以是非负有限数值（恒速发射，期望
        数为 ``rate * dt``），或 :class:`EmissionCurve`（曲线速率在 ``dt`` 上
        积分得到期望数）。``lifetime`` 为正有限标量或 ``(lo, hi)`` 区间。采样
        使用泊松分布 + 贝塞尔段内拒绝抽样，同种子同调用序列可复现。
        """
        pool = self._pool
        with pool.lock:
            if not _is_finite_number(dt) or dt < 0.0:
                raise TimeStepError("dt 必须是非负有限数值，收到 %r" % (dt,))
            self._apply_lifetime(pool, lifetime)
            is_curve = isinstance(rate, EmissionCurve)
            if not is_curve and (not _is_finite_number(rate) or rate < 0.0):
                raise RateError("rate 必须是非负有限数值，收到 %r" % (rate,))
            if dt == 0.0:
                return 0

            # 参数全部合法后再原地开启新代（O(1)）
            pool.begin_generation()
            if is_curve:
                return self._emit_curve(rate, float(dt))
            if rate == 0.0:
                return 0

            n = pool._poisson(rate * dt)
            spawned = 0
            k = 0
            lo = pool.life_lo
            hi = pool.life_hi
            while k < n:
                life = lo if lo == hi else lo + (hi - lo) * pool.rng.random()
                if not pool._spawn(life):
                    break
                spawned += 1
                k += 1
            return spawned

    def _emit_curve(self, curve, dt):
        pool = self._pool
        lo = pool.life_lo
        hi = pool.life_hi
        spawned = 0
        segs = curve._seg
        n = curve._nseg
        s = 0
        while s < n:
            seg = segs[s]
            s += 1
            env = seg[EmissionCurve._ENV]
            if env <= 0.0:
                continue
            # 包络泊松数 + 拒绝抽样：被积强度 B_y(u)*Bx'(u)，上界 env=ymax*span，
            # 接受概率 B_y*Bx'/env，接受后的期望恰为 dt * 段解析积分。
            proposals = pool._poisson(dt * env)
            if proposals == 0:
                continue
            cyb = EmissionCurve._CY
            db = EmissionCurve._D
            k = 0
            while k < proposals:
                u = pool.rng.random()
                by = EmissionCurve._eval3(seg, cyb, u)
                dxu = (seg[db + 2] * u + seg[db + 1]) * u + seg[db]
                if pool.rng.random() * env < by * dxu:
                    life = lo if lo == hi else lo + (hi - lo) * pool.rng.random()
                    if pool._spawn(life):
                        spawned += 1
                k += 1
        return spawned

    def update(self, dt):
        """推进时间 ``dt`` 并按到期顺序回收粒子，返回自身。"""
        pool = self._pool
        with pool.lock:
            if not _is_finite_number(dt):
                raise TimeStepError("dt 必须是有限数值，收到 %r" % (dt,))
            if dt < 0.0:
                raise TimeStepError("dt 不能为负，收到 %r" % (dt,))
            pool.now += float(dt)
            pool._reap()
            return self

    def live(self):
        """当前存活粒子数。"""
        with self._pool.lock:
            return int(self._pool.live_count)

    def positions(self):
        """存活粒子的位置列表 ``[(x, y), ...]``，按槽位顺序。"""
        pool = self._pool
        with pool.lock:
            epoch = pool.epoch
            slot_epoch = pool.slot_epoch
            out = []
            i = 0
            cap = pool.cap
            while i < cap:
                if slot_epoch[i] == epoch:
                    out.append((pool.px[i], pool.py[i]))
                i += 1
            return out

    def reuse_count(self):
        """槽位被重复利用的次数（每次复用已分配过的槽位累加 1）。"""
        with self._pool.lock:
            return int(self._pool.reuse)

    def peak_allocated(self):
        """历史同时占用槽位的峰值，恒满足 ``<= cap``。"""
        with self._pool.lock:
            return int(self._pool.peak)

    def capacity(self):
        return self._pool.cap
