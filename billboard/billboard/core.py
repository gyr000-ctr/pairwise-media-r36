"""billboard 核心 API：相机基公告板朝向、稳定视深排序与背面剔除。

三种朝向模式（mode 取题面约定的英文名，亦接受常见别名）：

* ``view``   屏幕对齐公告板：基向量直接取自相机基 ``(cam_fwd, cam_up)``；
* ``axis``   轴向（圆柱 / Y 轴锁定）公告板：``up`` 锁定为 ``cam_up`` 轴，
             法线只取朝相机方向在该轴垂面内的分量，不随相机滚转；
* ``sphere`` 球形（全朝向）公告板：法线正对相机，``up`` 取相机上方向在
             法线垂面内的投影。

所有模式都返回右手正交基 ``right × up = forward(=normal)``，并在最后用
叉乘重建一遍，把正交 / 单位误差压到浮点噪声量级（~1e-16，远小于 1e-9）。

排序 :meth:`Billboard.sort` 按到相机的欧氏视深由远及近（画家算法顺序），
以原始索引作为同深度次键，严格稳定。

背面剔除统一是 ``dot(normal, view_vector)`` 的符号判定，提供两种方向约定
和一个免约定的位置判定：

* 视向量约定 ``to_camera`` 为**公告板表面指向相机**的向量（图形学最常见的
  ``V = cam_pos - pos``）：``dot > 0`` 为正面 / 可见，``dot <= 0`` 为背面 /
  应剔除（掠射 ``dot == 0`` 按不可见处理）；``is_frontface`` /
  ``is_visible`` / ``facing_camera`` 与 ``is_backface`` / ``cull`` 据此互补；
  若希望掠射面保留，可用严格负号的 ``is_backface_strict``（``dot < 0``）。
* 视线约定 ``ray_dir`` 为**相机指向公告板**的方向（即 ``cam_fwd``）：
  ``dot > 0`` 为背面（``is_backface_ray`` / ``cull_ray``），``dot <= 0``
  可见（``is_visible_ray``）。它与视向量约定严格等价，只需把向量取反。
* 位置判定 :func:`facing_camera_at` / :func:`cull_at` 直接给表面位置和相机
  位置，内部计算 ``cam_pos - pos``，不存在方向约定歧义。

全部接口无共享可变状态，可在多线程中共享同一个 :class:`Billboard` 实例
并发调用。所有非法输入都抛出 :class:`BillboardError` 的具体子类，且这些
异常都继承 :class:`ValueError`。
"""

import math

__all__ = [
    "Billboard",
    "BillboardError",
    "InvalidModeError",
    "NonFiniteInputError",
    "DegenerateInputError",
    "NonFiniteError",
    "ModeError",
    "InvalidModeArgument",
    "ValidationError",
    "InputError",
    "normalize_mode",
    "as_vec3",
    "dot3",
    "depth_squared",
    "view_depth",
    # 视向量约定（表面 -> 相机）
    "is_frontface",
    "facing_camera",
    "is_visible",
    "is_visible_to_camera",
    "is_backface",
    "cull",
    "is_culled",
    "should_cull",
    "is_backface_strict",
    "backface_to_camera",
    # 视线约定（相机 -> 表面）
    "is_backface_ray",
    "cull_ray",
    "backface_ray",
    "is_visible_ray",
    # 位置判定（免方向约定）
    "facing_camera_at",
    "cull_at",
]


# ---------------------------------------------------------------------------
# 错误分类：全部继承 ValueError，既能统一捕获，又能按子类区分
# ---------------------------------------------------------------------------
class BillboardError(ValueError):
    """公告板输入或参数错误的基类。"""


class InvalidModeError(BillboardError):
    """非法朝向模式（未知字符串或非字符串类型）。"""


class NonFiniteInputError(BillboardError):
    """输入向量含 NaN / Inf 等非有限分量。"""


class DegenerateInputError(BillboardError):
    """输入向量无法构造有效基（典型：零长度方向向量）。"""


# 常见 / 同族任务命名别名，便于不同调用习惯（均为上述类型的同一类对象）。
NonFiniteError = NonFiniteInputError
ModeError = InvalidModeError
InvalidModeArgument = InvalidModeError
ValidationError = BillboardError
InputError = BillboardError


# ---------------------------------------------------------------------------
# 模式规范化
# ---------------------------------------------------------------------------
_MODE_ALIASES = {
    # 题面 canonical 名
    "view": "view",
    "axis": "axis",
    "sphere": "sphere",
    # view 的常见别名（屏幕对齐）
    "screen": "view",
    "screenaligned": "view",
    "screenalign": "view",
    "camera": "view",
    "camerafacing": "view",
    "billboard": "view",
    # axis 的常见别名（轴向 / 圆柱 / 锁定上轴）
    "axial": "axis",
    "yaxis": "axis",
    "vertical": "axis",
    "cylindrical": "axis",
    "upright": "axis",
    # sphere 的常见别名（球形 / 全朝向）
    "spherical": "sphere",
    "ball": "sphere",
    "point": "sphere",
    # 中文模式名
    "屏幕对齐": "view",
    "屏幕": "view",
    "轴向": "axis",
    "轴": "axis",
    "球形": "sphere",
    "球面": "sphere",
}

_MODES = ("view", "axis", "sphere")

# 单位向量叉乘范数低于此值即视为平行，切换到确定性回退方向。
_PARALLEL_EPS = 1e-12


def normalize_mode(mode):
    """把模式名规范化为 ``view`` / ``axis`` / ``sphere``。

    接受大小写、首尾空白与 ``- _ 空格`` 分隔差异，以及常见英文别名和中文
    模式名；其余值（含非字符串）抛出 :class:`InvalidModeError`。
    """
    if not isinstance(mode, str):
        raise InvalidModeError(
            "mode 必须是字符串（%s），收到类型 %s: %r"
            % ("/".join(_MODES), type(mode).__name__, mode)
        )
    key = "".join(ch for ch in mode.strip().lower() if ch.isalnum())
    if key not in _MODE_ALIASES:
        raise InvalidModeError(
            "未知朝向模式 %r；支持 view(屏幕对齐) / axis(轴向) / sphere(球形)"
            % (mode,)
        )
    return _MODE_ALIASES[key]


# ---------------------------------------------------------------------------
# 向量校验与小数学工具
# ---------------------------------------------------------------------------
def as_vec3(value, name="向量"):
    """把输入校验并转换为长度 3 的 ``float`` 元组。

    结构 / 类型非法抛 :class:`BillboardError`；含 NaN/Inf 抛
    :class:`NonFiniteInputError`。
    """
    if isinstance(value, (str, bytes)):
        raise BillboardError(
            "%s 必须是长度为 3 的数值向量，收到 %r" % (name, value)
        )
    try:
        coords = tuple(value)
    except TypeError:
        raise BillboardError(
            "%s 必须是长度为 3 的向量，收到 %r" % (name, value)
        ) from None
    if len(coords) != 3:
        raise BillboardError(
            "%s 必须恰好包含 3 个分量，收到长度 %d: %r"
            % (name, len(coords), value)
        )
    out = []
    for comp in coords:
        try:
            fcomp = float(comp)
        except (TypeError, ValueError):
            raise BillboardError(
                "%s 的分量必须是实数，收到 %r" % (name, comp)
            ) from None
        if not math.isfinite(fcomp):
            raise NonFiniteInputError(
                "%s 含非有限分量（NaN/Inf）: %r" % (name, value)
            )
        out.append(fcomp)
    return (out[0], out[1], out[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v):
    # math.hypot 内部做缩放，避免大分量平方溢出。
    return math.hypot(v[0], v[1], v[2])


def _unit(v, name="向量"):
    n = _norm(v)
    if n == 0.0:
        raise DegenerateInputError("%s 长度为 0，无法归一化" % name)
    return (v[0] / n, v[1] / n, v[2] / n)


def _positive_zero(v):
    # 把 -0.0 统一成 +0.0，输出更干净，也便于逐值比较。
    return (v[0] + 0.0, v[1] + 0.0, v[2] + 0.0)


def _perp(v):
    """返回与单位向量 ``v`` 正交的确定性单位向量。

    选绝对值最小的坐标轴叉乘，结果范数不小于 sqrt(2/3)，不会数值退化。
    """
    ax, ay, az = abs(v[0]), abs(v[1]), abs(v[2])
    if ax <= ay and ax <= az:
        axis = (1.0, 0.0, 0.0)
    elif ay <= az:
        axis = (0.0, 1.0, 0.0)
    else:
        axis = (0.0, 0.0, 1.0)
    return _unit(_cross(v, axis))


def _orthonormal(right_hint, up_hint):
    """由两个方向提示构造严格右手正交基 ``(right, up, forward)``。

    先把 ``up`` 对 ``right`` 做 Gram 正交化，再用叉乘循环重建三遍，
    保证三个向量两两正交且为单位长度（误差仅浮点噪声）。
    """
    right = _unit(right_hint, "right")
    proj = _dot(up_hint, right)
    up0 = (
        up_hint[0] - right[0] * proj,
        up_hint[1] - right[1] * proj,
        up_hint[2] - right[2] * proj,
    )
    if _norm(up0) < _PARALLEL_EPS:
        up = _perp(right)
    else:
        up = _unit(up0, "up")

    forward = _unit(_cross(right, up))
    up = _unit(_cross(forward, right))
    right = _unit(_cross(up, forward))
    return _positive_zero(right), _positive_zero(up), _positive_zero(forward)


# ---------------------------------------------------------------------------
# 视深
# ---------------------------------------------------------------------------
def depth_squared(pos, cam_pos):
    """返回 ``pos`` 到 ``cam_pos`` 的平方欧氏距离（单调等价于视深）。"""
    p = as_vec3(pos, "pos")
    c = as_vec3(cam_pos, "cam_pos")
    dx = p[0] - c[0]
    dy = p[1] - c[1]
    dz = p[2] - c[2]
    return dx * dx + dy * dy + dz * dz


def view_depth(pos, cam_pos):
    """返回 ``pos`` 到 ``cam_pos`` 的欧氏距离（视深）。"""
    return math.sqrt(depth_squared(pos, cam_pos))


# ---------------------------------------------------------------------------
# 背面剔除：判定完全由 dot(normal, view_vector) 的符号决定
# ---------------------------------------------------------------------------
def _normal_arg(normal):
    if isinstance(normal, dict):
        for key in ("normal", "forward", "n", "dir"):
            if key in normal:
                return as_vec3(normal[key], "normal")
        raise BillboardError("normal 字典缺少 normal/forward 字段: %r" % (normal,))
    return as_vec3(normal, "normal")


def dot3(a, b):
    """两个 3 向量的点积（会先做有限性与长度校验）。"""
    return _dot(as_vec3(a, "a"), as_vec3(b, "b"))


# ---------------------------------------------------------------------------
# 视向量约定：to_camera = 表面 -> 相机（V = cam_pos - pos）
# ---------------------------------------------------------------------------
def is_frontface(normal, to_camera):
    """``dot(normal, to_camera) > 0`` 时为正面（法线朝向观察者）。"""
    n = _normal_arg(normal)
    v = as_vec3(to_camera, "to_camera")
    return _dot(n, v) > 0.0


#: 正面 / 朝向相机的同义名。
facing_camera = is_frontface


def is_visible(normal, to_camera):
    """可见性（视向量约定）：``dot(normal, to_camera) > 0`` 才可见。

    与 ``decal`` 等同族任务一致，点积恰为 0 的掠射 / 侧对情形按不可见处理。
    """
    return is_frontface(normal, to_camera)


#: 显式标明视向量约定的别名。
is_visible_to_camera = is_visible


def is_backface(normal, to_camera):
    """背面判定（视向量约定）：``dot(normal, to_camera) <= 0`` 即背面。

    它是 :func:`is_visible` 的严格补集；点积为 0 的掠射面也按剔除处理。
    若要保留掠射面，使用 :func:`is_backface_strict`。
    """
    n = _normal_arg(normal)
    v = as_vec3(to_camera, "to_camera")
    return _dot(n, v) <= 0.0


#: ``cull`` 是 :func:`is_backface` 的同义名：True 表示应当剔除。
def cull(normal, to_camera):
    return is_backface(normal, to_camera)


#: 按“是否已剔除 / 是否应剔除”命名的别名。
is_culled = cull
should_cull = cull


def is_backface_strict(normal, to_camera):
    """严格背面：仅当 ``dot(normal, to_camera) < 0``（法线明确背离相机）。"""
    n = _normal_arg(normal)
    v = as_vec3(to_camera, "to_camera")
    return _dot(n, v) < 0.0


#: 旧命名 / 强调点积负号的别名。
backface_to_camera = is_backface_strict


# ---------------------------------------------------------------------------
# 视线约定：ray_dir = 相机 -> 表面（≈ cam_fwd），与上面相差一个负号
# ---------------------------------------------------------------------------
def is_backface_ray(normal, ray_dir):
    """视线约定：``dot(normal, ray_dir) > 0`` 时为背面。"""
    n = _normal_arg(normal)
    d = as_vec3(ray_dir, "ray_dir")
    return _dot(n, d) > 0.0


#: 视线约定下 cull 的别名。
def cull_ray(normal, ray_dir):
    return is_backface_ray(normal, ray_dir)


#: ``is_backface_ray`` 的短别名。
backface_ray = is_backface_ray


def is_visible_ray(normal, ray_dir):
    """视线约定的可见性：``dot(normal, ray_dir) <= 0`` 时可见。"""
    n = _normal_arg(normal)
    d = as_vec3(ray_dir, "ray_dir")
    return _dot(n, d) <= 0.0


# ---------------------------------------------------------------------------
# 位置判定：内部计算 cam_pos - pos，免方向约定歧义
# ---------------------------------------------------------------------------
def facing_camera_at(normal, pos, cam_pos):
    """给定表面位置与相机位置，正面朝向相机时返回 True。"""
    p = as_vec3(pos, "pos")
    c = as_vec3(cam_pos, "cam_pos")
    return is_frontface(normal, (c[0] - p[0], c[1] - p[1], c[2] - p[2]))


def cull_at(normal, pos, cam_pos):
    """给定表面位置与相机位置，应背面剔除时返回 True。"""
    p = as_vec3(pos, "pos")
    c = as_vec3(cam_pos, "cam_pos")
    return is_backface(normal, (c[0] - p[0], c[1] - p[1], c[2] - p[2]))


# ---------------------------------------------------------------------------
# 排序条目解析
# ---------------------------------------------------------------------------
def _item_pos(entry):
    """从排序条目取出位置并校验。

    支持 ``(pos, payload)`` 二元条目、裸 ``(x, y, z)`` 位置，以及带
    ``pos``/``position`` 键的字典条目。
    """
    if isinstance(entry, dict):
        for key in ("pos", "position", "p"):
            if key in entry:
                return as_vec3(entry[key], "item.pos")
        raise BillboardError("字典条目缺少 pos/position 字段: %r" % (entry,))
    if isinstance(entry, (str, bytes)):
        raise BillboardError("非法公告板条目: %r" % (entry,))
    try:
        pos, _payload = entry
    except (TypeError, ValueError):
        # 不是二元条目：按裸位置向量处理（由 as_vec3 判定长度是否为 3）。
        return as_vec3(entry, "item.pos")
    return as_vec3(pos, "item.pos")


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------
class Billboard:
    """无状态公告板工具：朝向、视深排序与背面剔除。

    实例不保存任何每次调用的可变状态，可安全地被多线程共享。
    """

    # -- 朝向 ---------------------------------------------------------------
    def orient(self, pos, cam_fwd, cam_up=None, mode=None):
        """计算公告板的右手正交基。

        参数
        ----
        pos : 长度 3 的位置向量。题面约定朝向只依赖相机基；``pos`` 仍会
              做完整校验（非有限输入分类报错），但不改变朝向结果。
        cam_fwd : 相机前向（指向场景深处，例如 ``(0, 0, -1)``），不能为
              零向量或含非有限分量。
        cam_up : 相机上方向；为 ``None`` 时使用世界上轴 ``(0, 1, 0)``。
        mode : ``view`` / ``axis`` / ``sphere`` 之一（接受别名）。也支持
              ``orient(pos, fwd, 'view')`` 这种省略 ``cam_up`` 的写法。

        返回
        ----
        dict，含 ``right`` / ``up`` / ``forward`` / ``normal`` 四个键，
        值均为 3 个 ``float`` 组成的元组；``forward`` 与 ``normal`` 同义，
        是朝向相机的公告板正面法线，且 ``right × up = forward``。
        """
        # 兼容 orient(pos, fwd, 'sphere') 的三参写法。
        if mode is None:
            if isinstance(cam_up, str):
                mode = cam_up
                cam_up = None
            else:
                mode = "view"

        # 先规范化模式：非法模式优先报 InvalidModeError，与其它输入错误区分。
        mode_key = normalize_mode(mode)

        # pos 虽不参与相机基计算，也必须校验（NaN/Inf 要分类报错）。
        as_vec3(pos, "pos")
        f = as_vec3(cam_fwd, "cam_fwd")
        if cam_up is None:
            u = (0.0, 1.0, 0.0)
        else:
            u = as_vec3(cam_up, "cam_up")

        fwd_unit = _unit(f, "cam_fwd")
        up_unit = _unit(u, "cam_up")

        if mode_key == "view":
            # 屏幕对齐：right = normalize(cam_fwd × cam_up)，再正交重建。
            hint = _cross(fwd_unit, up_unit)
            if _norm(hint) < _PARALLEL_EPS:
                right_hint = _perp(fwd_unit)
            else:
                right_hint = hint
            right, up, normal = _orthonormal(right_hint, up_unit)

        elif mode_key == "axis":
            # 轴向：up 锁定为相机上轴；法线 = 朝相机方向 (-fwd) 在该轴
            # 垂面内的投影；right = axis × normal。
            axis = up_unit
            toward = (-fwd_unit[0], -fwd_unit[1], -fwd_unit[2])
            axial = _dot(toward, axis)
            horizontal = (
                toward[0] - axis[0] * axial,
                toward[1] - axis[1] * axial,
                toward[2] - axis[2] * axial,
            )
            if _norm(horizontal) < _PARALLEL_EPS:
                normal0 = _perp(axis)
            else:
                normal0 = _unit(horizontal, "axis-normal")
            right0 = _unit(_cross(axis, normal0))
            normal = _unit(_cross(right0, axis))
            right = _positive_zero(_unit(_cross(axis, normal)))
            up = _positive_zero(axis)
            normal = _positive_zero(normal)

        else:  # sphere
            # 球形：法线正对相机 (-fwd)，up 为相机 up 在法线垂面的投影。
            normal0 = (-fwd_unit[0], -fwd_unit[1], -fwd_unit[2])
            proj = _dot(up_unit, normal0)
            up_perp = (
                up_unit[0] - normal0[0] * proj,
                up_unit[1] - normal0[1] * proj,
                up_unit[2] - normal0[2] * proj,
            )
            if _norm(up_perp) < _PARALLEL_EPS:
                up_hint = _perp(normal0)
            else:
                up_hint = up_perp
            right_hint = _cross(_unit(up_hint), normal0)
            right, up, normal = _orthonormal(right_hint, up_hint)

        return {
            "right": right,
            "up": up,
            "forward": normal,
            "normal": normal,
        }

    # -- 排序 ---------------------------------------------------------------
    def sort(self, items, cam_pos):
        """按视深（到 ``cam_pos`` 的距离）由远及近稳定排序。

        ``items`` 为 ``(pos, payload)`` 二元组序列，或裸 ``(x, y, z)``
        位置序列（也接受带 ``pos`` 键的字典）。返回**新列表**，元素仍是
        原始条目对象（不重建、不修改入参）。同深度条目严格保持插入序。
        空序列返回 ``[]``；不可迭代或条目 / 相机位置非法时抛
        :class:`BillboardError` 子类。
        """
        cam = as_vec3(cam_pos, "cam_pos")
        if items is None or isinstance(items, (str, bytes)):
            raise BillboardError(
                "items 必须是 (pos, payload) 序列或裸位置序列，收到 %r" % (items,)
            )
        try:
            seq = list(items)
        except TypeError:
            raise BillboardError("items 必须是可迭代序列，收到 %r" % (items,)) from None

        records = []
        cx, cy, cz = cam
        for index, entry in enumerate(seq):
            p = _item_pos(entry)
            dx = p[0] - cx
            dy = p[1] - cy
            dz = p[2] - cz
            depth2 = dx * dx + dy * dy + dz * dz
            records.append((depth2, index, entry))

        # 主键 -depth2：远（大深度）在前；次键 index：同深度保持插入序。
        records.sort(key=lambda rec: (-rec[0], rec[1]))
        return [rec[2] for rec in records]

    #: 语义化别名。
    def sort_back_to_front(self, items, cam_pos):
        """与 :meth:`sort` 相同（由远及近 / 画家算法顺序）。"""
        return self.sort(items, cam_pos)

    # -- 视深 ---------------------------------------------------------------
    def depth_squared(self, pos, cam_pos):
        return depth_squared(pos, cam_pos)

    def view_depth(self, pos, cam_pos):
        return view_depth(pos, cam_pos)

    # -- 背面剔除（方法形式，委托给模块级同名函数） ---------------------------
    @staticmethod
    def is_frontface(normal, to_camera):
        """视向量约定：``dot(normal, to_camera) > 0`` 为正面。"""
        return is_frontface(normal, to_camera)

    #: 正面判定的别名。
    facing_camera = is_frontface

    @staticmethod
    def is_visible(normal, to_camera):
        """``dot(normal, to_camera) > 0`` 时可见。"""
        return is_visible(normal, to_camera)

    @staticmethod
    def is_visible_to_camera(normal, to_camera):
        return is_visible_to_camera(normal, to_camera)

    @staticmethod
    def is_backface(normal, to_camera):
        """视向量约定：``dot(normal, to_camera) <= 0`` 为背面 / 应剔除。"""
        return is_backface(normal, to_camera)

    @staticmethod
    def cull(normal, to_camera):
        """返回 True 表示该公告板应被背面剔除。"""
        return cull(normal, to_camera)

    #: ``backface`` 是 :meth:`is_backface` 的短别名。
    backface = is_backface

    #: 按“是否已剔除 / 是否应剔除”命名的别名。
    is_culled = cull
    should_cull = cull

    @staticmethod
    def is_backface_strict(normal, to_camera):
        return is_backface_strict(normal, to_camera)

    @staticmethod
    def backface_to_camera(normal, to_camera):
        return backface_to_camera(normal, to_camera)

    @staticmethod
    def is_backface_ray(normal, ray_dir):
        """视线约定（相机→表面）：``dot(normal, ray_dir) > 0`` 为背面。"""
        return is_backface_ray(normal, ray_dir)

    @staticmethod
    def cull_ray(normal, ray_dir):
        return cull_ray(normal, ray_dir)

    @staticmethod
    def backface_ray(normal, ray_dir):
        return backface_ray(normal, ray_dir)

    @staticmethod
    def is_visible_ray(normal, ray_dir):
        return is_visible_ray(normal, ray_dir)

    @staticmethod
    def facing_camera_at(normal, pos, cam_pos):
        return facing_camera_at(normal, pos, cam_pos)

    @staticmethod
    def cull_at(normal, pos, cam_pos):
        return cull_at(normal, pos, cam_pos)

    @staticmethod
    def dot(a, b):
        return dot3(a, b)
