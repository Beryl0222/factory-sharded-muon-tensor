"""物理分片轴序 RC/CR 与逻辑行列坐标之间的换轴。

约定（README）：

* 区间与 ``axis`` 永远使用**原始逻辑矩阵**的行列坐标，区间左闭右开；
* ``layout="RC"`` 表示物理第 0 轴是逻辑行、第 1 轴是逻辑列；
* ``layout="CR"`` 表示物理第 0 轴是逻辑列、第 1 轴是逻辑行。

因此二维情形下 CR 就是物理数组相对逻辑块的转置。换轴尽量返回视图，
不额外物化高维矩阵副本。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# physical axis position -> logical axis identity
_LAYOUT_TO_PHYS = {
    "RC": (0, 1),
    "R,C": (0, 1),
    "CR": (1, 0),
    "C,R": (1, 0),
}


def transpose_permutation(layout: str) -> tuple[int, ...]:
    """返回把物理轴重排为逻辑 (row, col) 顺序所需的 transpose 轴序。

    例：``CR`` 的物理轴是 (col, row)，逻辑顺序要先 row 后 col，
    所以 permutation 为 ``(1, 0)``，即 ``arr.transpose(1, 0)``。
    """
    key = str(layout).strip().upper()
    if key not in _LAYOUT_TO_PHYS:
        raise ValueError(f"未知物理布局 {layout!r}，仅支持 RC/CR")
    phys_to_logical = _LAYOUT_TO_PHYS[key]
    # 逻辑轴 l 位于哪个物理轴 p：perm[l] = p
    perm = [0] * len(phys_to_logical)
    for p, logical_axis in enumerate(phys_to_logical):
        perm[logical_axis] = p
    return tuple(perm)


def _expected_logical_shape(shape: Sequence[int], axis: int, lo: int, hi: int) -> tuple[int, int]:
    rows, cols = int(shape[0]), int(shape[1])
    if axis == 0:
        return (hi - lo, cols)
    if axis == 1:
        return (rows, hi - lo)
    raise ValueError(f"axis 只能是 0(行) 或 1(列)，收到 {axis!r}")


def unpack_physical(
    array: np.ndarray,
    layout: str,
    axis: int | None = None,
    range_: Sequence[int] | None = None,
    shape: Sequence[int] | None = None,
) -> np.ndarray:
    """把物理分片还原成逻辑坐标块（逻辑行、逻辑列顺序）。

    参数 ``axis/range_/shape`` 可选；给出时会核对还原后的逻辑形状是否等于
    该区间在完整逻辑矩阵中的期望块形状，不符抛 :class:`ValueError`。
    """
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"分片必须是二维，收到 {arr.ndim} 维")
    perm = transpose_permutation(layout)
    logical = np.transpose(arr, perm)
    if axis is not None and range_ is not None and shape is not None:
        lo, hi = int(range_[0]), int(range_[1])
        expected = _expected_logical_shape(shape, int(axis), lo, hi)
        if tuple(logical.shape) != expected:
            raise ValueError(
                f"分片逻辑形状 {tuple(logical.shape)} 与声明区间期望 {expected} 不符"
                f"（axis={axis}, range=[{lo},{hi}), shape={tuple(shape)}, layout={layout}）"
            )
    return logical


def pack_physical(logical_block: np.ndarray, layout: str) -> np.ndarray:
    """把逻辑坐标块编码回上报时的物理轴序（尽量为视图）。"""
    arr = np.asarray(logical_block)
    perm = transpose_permutation(layout)  # logical axis -> physical source axis
    # unpack: logical = physical.transpose(perm)，其中 perm[l]=p。
    # 逆变换：physical = logical.transpose(inv)，inv[p]=l。
    inv = [0] * len(perm)
    for logical_axis, physical_axis in enumerate(perm):
        inv[physical_axis] = logical_axis
    return np.transpose(arr, tuple(inv))


def physical_shape_for(
    shape: Sequence[int], axis: int, lo: int, hi: int, layout: str
) -> tuple[int, int]:
    """给定逻辑切分声明，期望的物理数组形状。"""
    logical_shape = _expected_logical_shape(shape, axis, lo, hi)
    perm = transpose_permutation(layout)
    # physical axis p corresponds to logical axis perm... 直接用逆映射
    inv = [0, 0]
    for logical_axis, physical_axis in enumerate(perm):
        inv[physical_axis] = logical_axis
    return (logical_shape[inv[0]], logical_shape[inv[1]])
