"""容量资料（data/capacity.json）：分片规模、6 MiB 暂存预算与种子回放。

大矩阵 18432x432 的薄维 432 足以放下全局 Gram（432^2），但放不下一整份
高维矩阵的 float64 副本（约 64 MiB）。本模块给出暂存估算，并可用种子
跑一次真实的分块引擎核对正确性。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contract import IterationSpec
from .coverage import ShardPiece
from .engine import assemble_logical_x, sharded_muon
from .reference import muon_reference


@dataclass
class ScratchPlan:
    shape: tuple[int, int]
    partitions: int
    split_axis: int
    mode: str
    bytes_per_element: int
    gram_bytes: int
    gram_squared_bytes: int
    largest_block_bytes: int
    estimated_peak_bytes: int
    budget_bytes: int | None
    full_matrix_copy_bytes: int
    fits_budget: bool | None
    conservative_peak_bytes: int = 0
    conservative_fits_budget: bool | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "shape": list(self.shape),
            "partitions": self.partitions,
            "split_axis": self.split_axis,
            "mode": self.mode,
            "bytes_per_element": self.bytes_per_element,
            "gram_bytes": self.gram_bytes,
            "gram_squared_bytes": self.gram_squared_bytes,
            "largest_block_bytes": self.largest_block_bytes,
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "budget_bytes": self.budget_bytes,
            "full_matrix_copy_bytes": self.full_matrix_copy_bytes,
            "fits_budget": self.fits_budget,
            "budget_margin_bytes": (
                None if self.budget_bytes is None
                else self.budget_bytes - self.estimated_peak_bytes
            ),
            "conservative_peak_bytes": self.conservative_peak_bytes,
            "conservative_fits_budget": self.conservative_fits_budget,
            "note": self.note,
        }
        if self.budget_bytes is not None:
            d["conservative_budget_margin_bytes"] = (
                self.budget_bytes - self.conservative_peak_bytes
            )
        return d


def balanced_ranges(length: int, partitions: int) -> list[tuple[int, int]]:
    """把 length 尽量均匀地切成 partitions 个左闭右开区间。"""
    partitions = min(int(partitions), int(length))
    base, rem = divmod(int(length), partitions)
    ranges: list[tuple[int, int]] = []
    cur = 0
    for i in range(partitions):
        size = base + (1 if i < rem else 0)
        ranges.append((cur, cur + size))
        cur += size
    return ranges


def scratch_plan(
    shape: tuple[int, int] = (18432, 432),
    partitions: int = 23,
    *,
    split_axis: int = 0,
    mode: str = "float64",
    budget_bytes: int | None = 6_291_456,
) -> ScratchPlan:
    """估算分块引擎的主要新分配暂存峰值（Gram 双矩阵 + 最大工作块）。

    调用方自有的只读分片输入不计入；这里估算的是额外 Gram/Gram^2(B)
    与一个最大工作块。表达式临时数组可能再增加至多一个 Gram 大小，
    因此同时给出保守口径 ``estimated_peak_with_temporaries`` 见返回 dict。
    """
    itemsize = np.dtype({"binary16": np.float16, "float32": np.float32,
                        "float64": np.float64}[mode]).itemsize
    R, C = int(shape[0]), int(shape[1])
    thin = min(R, C)
    gram = thin * thin * itemsize
    gram2 = gram  # A2/B 复用同一量级
    axis_len = R if split_axis == 0 else C
    other = C if split_axis == 0 else R
    ranges = balanced_ranges(axis_len, partitions)
    max_chunk = max(hi - lo for lo, hi in ranges)
    block_elems = max_chunk * other
    block_bytes = block_elems * itemsize
    # 主口径：A + B(A2) + 一个工作块（主要持久暂存）。
    peak = gram + gram2 + block_bytes
    # 保守口径：多项式应用时 matmul 输出需要第二个同形块，外加一个 Gram 级临时。
    conservative = gram + gram2 + 2 * block_bytes + gram
    full_copy = R * C * 8  # “高维矩阵一整份副本”按 float64 衡量
    note = (
        "全局 Gram 仅 thin^2；分片引擎不分配整份高维矩阵。"
        "6MiB 板端预算建议 binary16/float32 候选精度；float64 基准在训练侧做分块比对。"
    )
    return ScratchPlan(
        shape=(R, C), partitions=partitions, split_axis=split_axis, mode=mode,
        bytes_per_element=itemsize, gram_bytes=gram, gram_squared_bytes=gram2,
        largest_block_bytes=block_bytes, estimated_peak_bytes=peak,
        budget_bytes=budget_bytes,
        fits_budget=None if budget_bytes is None else peak <= budget_bytes,
        full_matrix_copy_bytes=full_copy,
        conservative_peak_bytes=conservative,
        conservative_fits_budget=(
            None if budget_bytes is None else conservative <= budget_bytes
        ),
        note=note,
    )


def load_capacity(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "capacity.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_capacity_replay(
    path: str | Path | None = None,
    *,
    verify_full_reference: bool = True,
) -> dict[str, Any]:
    """按 capacity.json 的种子/规模跑行切分与列切分，返回核对报告。

    这是给测试/巡检用的真实大矩阵回放；当 ``verify_full_reference=True``
    时会在测试侧装配整矩阵 float64 参考（产品引擎路径本身不分配它）。
    """
    cap = load_capacity(path)
    R, C = cap["matrix"]
    n_parts = int(cap["partitions"])
    n_iter = int(cap["iterations"])
    seed = int(cap["seed"])
    budget = int(cap["max_scratch_bytes"])
    spec = IterationSpec(iterations=n_iter)

    rng = np.random.default_rng(seed)
    # 数值尺度适中，避免低精度模式在大矩阵上必然溢出。
    V_full = rng.standard_normal((R, C)).astype(np.float64) / np.sqrt(C)

    report: dict[str, Any] = {
        "capacity": cap,
        "row_plan": scratch_plan((R, C), n_parts, split_axis=0, budget_bytes=budget).to_dict(),
        "column_plan": scratch_plan((R, C), n_parts, split_axis=1, budget_bytes=budget).to_dict(),
    }

    # 行切分（axis=0）
    row_ranges = balanced_ranges(R, n_parts)
    row_pieces = [
        ShardPiece(f"r{i}", 0, lo, hi, V_full[lo:hi, :].copy(), layout="RC")
        for i, (lo, hi) in enumerate(row_ranges)
    ]
    row_res = sharded_muon((R, C), row_pieces, spec, mode="float64")
    row_x = assemble_logical_x((R, C), row_res)

    # 列切分（axis=1）：列数 432，切成与行不同的块数也应一致。
    col_ranges = balanced_ranges(C, max(2, n_parts // 2))
    col_pieces = [
        ShardPiece(f"c{i}", 1, lo, hi, V_full[:, lo:hi].copy(), layout="RC")
        for i, (lo, hi) in enumerate(col_ranges)
    ]
    col_res = sharded_muon((R, C), col_pieces, spec, mode="float64")
    col_x = assemble_logical_x((R, C), col_res)

    report["row_vs_col_maxerr"] = float(np.abs(row_x - col_x).max())
    report["row_final_norm"] = row_res.final_norm
    report["col_final_norm"] = col_res.final_norm

    if verify_full_reference:
        ref = muon_reference(V_full, spec)
        report["row_vs_reference_maxerr"] = float(np.abs(row_x - ref).max())
        report["col_vs_reference_maxerr"] = float(np.abs(col_x - ref).max())

    # 明确记录：整矩阵 float64 副本远大于预算，而 Gram 很小。
    report["full_float64_copy_within_budget"] = (R * C * 8) <= budget
    report["gram_float64_within_budget"] = (C * C * 8) <= budget
    return report
