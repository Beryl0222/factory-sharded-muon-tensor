"""分片覆盖族：每个逻辑矩阵在同一步只认一套完整覆盖。

一次上报可能同时含行切分（axis=0）和列切分（axis=1）的分片。
它们形成两个互相独立的覆盖族；交付时至多选择其中**一套**完整覆盖，
绝不把两套叠起来重复计数。同一区间的重复/冲突上报由服务层先裁决。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .contract import REASON


@dataclass
class ShardPiece:
    """一块已还原为逻辑坐标的有效分片。"""

    partition: str
    axis: int
    lo: int
    hi: int
    logical: np.ndarray
    layout: str = "RC"
    version: int | None = None
    source: str | None = None
    # 上报时的物理数组（保留以便原样回传物理朝向的更新）。
    physical: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return self.hi - self.lo


@dataclass
class Coverage:
    """某条切分轴上的一套区间覆盖。"""

    axis: int
    length: int
    pieces: list[ShardPiece] = field(default_factory=list)  # 按 lo 排序
    complete: bool = False
    gaps: list[tuple[int, int]] = field(default_factory=list)
    overlaps: list[tuple[str, str]] = field(default_factory=list)

    def interval_coords(self) -> list[tuple[int, int]]:
        return [(p.lo, p.hi) for p in self.pieces]


def analyze_coverage(axis: int, length: int, pieces: Sequence[ShardPiece]) -> Coverage:
    """检查这些块是否构成 ``[0, length)`` 的精确、不相交覆盖。

    不完整时返回 gaps（左闭右开）与 overlaps 诊断，不抛异常——
    “缺席的矩阵部分”是延迟原因而不是零块。
    """
    cov = Coverage(axis=int(axis), length=int(length))
    same_axis = [p for p in pieces if int(p.axis) == int(axis)]
    same_axis.sort(key=lambda p: (p.lo, p.hi, p.partition))
    cov.pieces = same_axis

    if not same_axis:
        cov.gaps = [(0, length)] if length > 0 else []
        cov.complete = length == 0
        return cov

    gaps: list[tuple[int, int]] = []
    overlaps: list[tuple[str, str]] = []

    if same_axis[0].lo > 0:
        gaps.append((0, same_axis[0].lo))

    prev: ShardPiece | None = None
    for p in same_axis:
        if p.lo < 0 or p.hi > length or p.lo >= p.hi:
            # 越界/空区间无法形成合法覆盖；记为与前块重叠式冲突并跳过拼接。
            overlaps.append((p.partition, "<bounds>"))
        if prev is not None:
            if p.lo < prev.hi:
                overlaps.append((prev.partition, p.partition))
            elif p.lo > prev.hi:
                gaps.append((prev.hi, p.lo))
        prev = p

    if prev is not None and prev.hi < length:
        gaps.append((prev.hi, length))

    # 完全覆盖要求：无 gap、无 overlap，且首尾正好是 0/length。
    in_bounds = all(0 <= p.lo < p.hi <= length for p in same_axis)
    cov.gaps = gaps
    cov.overlaps = overlaps
    cov.complete = (
        in_bounds
        and not gaps
        and not overlaps
        and same_axis[0].lo == 0
        and same_axis[-1].hi == length
    )
    return cov


def build_coverages(
    shape: tuple[int, int], pieces: Sequence[ShardPiece]
) -> dict[int, Coverage]:
    """分别构造行轴(0)与列轴(1)覆盖族。"""
    rows, cols = int(shape[0]), int(shape[1])
    return {
        0: analyze_coverage(0, rows, list(pieces)),
        1: analyze_coverage(1, cols, list(pieces)),
    }


def choose_coverage(
    coverages: dict[int, Coverage],
    *,
    preferred_axis: int | None = 0,
) -> tuple[Coverage | None, int | None, list[int]]:
    """从两套覆盖中选一套完整的。

    返回 ``(chosen, chosen_axis, ignored_complete_axes)``。
    若指定 ``preferred_axis`` 且该轴完整，优先使用；否则取另一完整轴；
    都不完整返回 ``(None, None, [])``。两套同时完整时另一套仅记录为
    “被忽略的等价覆盖”，绝不叠加计数。
    """
    complete_axes = [ax for ax in (0, 1) if coverages.get(ax) and coverages[ax].complete]
    if not complete_axes:
        return None, None, []
    if preferred_axis in complete_axes:
        chosen_ax = int(preferred_axis)
    else:
        chosen_ax = complete_axes[0]
    ignored = [ax for ax in complete_axes if ax != chosen_ax]
    return coverages[chosen_ax], chosen_ax, ignored


def pieces_from_arrays(
    arrays: Sequence[np.ndarray] | Mapping[str, np.ndarray],
    axis: int = 0,
    ranges: Sequence[tuple[int, int]] | None = None,
    *,
    layouts: Sequence[str] | str = "RC",
    partitions: Sequence[str] | None = None,
) -> list[ShardPiece]:
    """把普通数组列表包成 :class:`ShardPiece`。

    不给 ``ranges`` 时按数组在切分轴上的长度从 0 顺序拼接。
    ``arrays`` 也可以是 ``{partition: array}`` 映射。
    """
    if isinstance(arrays, Mapping):
        partitions = list(arrays.keys()) if partitions is None else partitions
        array_list = list(arrays.values())
    else:
        array_list = list(arrays)
    n = len(array_list)
    if partitions is None:
        partitions = [f"p{i}" for i in range(n)]
    if isinstance(layouts, str):
        layout_list = [layouts] * n
    else:
        layout_list = list(layouts)
    if ranges is None:
        ranges = []
        cur = 0
        for arr in array_list:
            length = int(np.asarray(arr).shape[axis])
            ranges.append((cur, cur + length))
            cur += length
    out: list[ShardPiece] = []
    for i, arr in enumerate(array_list):
        lo, hi = int(ranges[i][0]), int(ranges[i][1])
        out.append(ShardPiece(str(partitions[i]), int(axis), lo, hi,
                              np.asarray(arr), layout=layout_list[i]))
    return out


def coverage_delay_reasons(
    coverages: dict[int, Coverage],
    *,
    stale: Sequence[dict[str, Any]] = (),
    corrupt: Sequence[dict[str, Any]] = (),
    absent: Sequence[dict[str, Any]] = (),
    conflicts: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """生成机器可读的延迟原因列表（不完整时给训练端排查）。"""
    reasons: list[dict[str, Any]] = []
    for s in stale:
        reasons.append({"code": REASON.VERSION_MISMATCH, **s})
    for c in corrupt:
        reasons.append({"code": REASON.CORRUPT_SHARD, **c})
    for a in absent:
        reasons.append({"code": REASON.ABSENT_SHARD, **a})
    for x in conflicts:
        reasons.append({"code": REASON.CONFLICTING_SHARDS, **x})

    any_complete = any(cov.complete for cov in coverages.values())
    if not any_complete:
        for ax, cov in coverages.items():
            if cov.gaps:
                reasons.append(
                    {
                        "code": REASON.MISSING_COVERAGE,
                        "axis": ax,
                        "missing_ranges": [list(g) for g in cov.gaps],
                    }
                )
            if cov.overlaps:
                reasons.append(
                    {"code": REASON.MISSING_COVERAGE, "axis": ax, "overlaps": cov.overlaps}
                )
    return reasons
