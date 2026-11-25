"""分片 Muon 引擎与精度门。

设计要点（对应题目约束）：

* 转置方向只由完整逻辑形状 ``rows > cols`` 决定，与调度器当前拿到的
  物理分片形状无关；
* 全局 Gram 只有薄维 ``thin x thin``，任何时候都不分配高维整矩阵的
  float64 副本；
* 行块-of-X / 列块-of-X 两种切法都走分块代数：
  ``A = X X^T`` 按坐标块聚合成全局 Gram；应用多项式时跨切分轴求和
  ``(B X)_i = sum_k B_ik X_k``；
* Gram 的无序块对 ``i <= j`` 只计算一次，对称块直接镜像；
* 上报块先按逻辑坐标排序，求和/块对顺序与到达顺序无关。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .contract import (
    PRECISION_ORDER,
    IterationSpec,
    MatrixSpec,
    normalize_mode,
)
from .coverage import Coverage, ShardPiece
from .layout import pack_physical
from .reference import muon_reference


@dataclass
class BlockUpdate:
    """一个分片坐标上的最终 X（Muon 正交化结果）。"""

    partition: str
    axis: int
    lo: int
    hi: int
    layout: str
    x_logical: np.ndarray          # 逻辑行列朝向，dtype=实际精度
    x_physical: np.ndarray         # 按上报物理轴序编码
    x_reference: np.ndarray | None = None  # 对应坐标的 float64 参考
    block_error: float = 0.0

    def to_meta(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "axis": self.axis,
            "range": [self.lo, self.hi],
            "layout": self.layout,
            "dtype": str(self.x_logical.dtype),
            "shape_logical": list(self.x_logical.shape),
            "shape_physical": list(self.x_physical.shape),
            "block_error": self.block_error,
        }


@dataclass
class ShardedMuonResult:
    mode: str
    transpose: bool
    shape: tuple[int, int]
    blocks: list[BlockUpdate]
    final_norm: float
    finite: bool
    trace: dict[str, Any] = field(default_factory=dict)

    def max_error(self) -> float:
        if not self.blocks or self.blocks[0].x_reference is None:
            return float("nan")
        return max(b.block_error for b in self.blocks)


def _sorted_v_blocks(v_pieces: Sequence[ShardPiece]):
    pieces = sorted(v_pieces, key=lambda p: (p.axis, p.lo, p.hi, p.partition))
    axes = {p.axis for p in pieces}
    if len(axes) != 1:
        raise ValueError(f"一次分片 Muon 只能使用一套切分轴，收到轴 {sorted(axes)}")
    return pieces, int(axes.pop())


def sharded_muon(
    shape: tuple[int, int],
    v_pieces: Sequence[ShardPiece],
    spec: IterationSpec | None = None,
    mode: str = "float64",
    *,
    trace: dict[str, Any] | None = None,
) -> ShardedMuonResult:
    """对一套完整覆盖的动量分块执行分片 Muon，返回各块最终 X。

    ``v_pieces`` 中每块的 ``.logical`` 必须是**逻辑朝向的动量 v 块**，
    且合起来精确覆盖 ``shape``（本函数不负责覆盖校验）。
    """
    spec = spec or IterationSpec()
    mode = normalize_mode(mode)
    dt = np.dtype({"binary16": np.float16, "float32": np.float32, "float64": np.float64}[mode])
    a0, b0, c0 = spec.abc
    a, b, c = dt.type(a0), dt.type(b0), dt.type(c0)

    R, C = int(shape[0]), int(shape[1])
    transpose = R > C  # 只看完整逻辑形状
    m = min(R, C)

    pieces, split_axis = _sorted_v_blocks(v_pieces)

    # 转成迭代朝向 X（薄维为行）。坐标区间始终索引 X 的 p_axis。
    blocks: list[list[Any]] = []
    for p in pieces:
        arr = np.asarray(p.logical)
        x = arr.T if transpose else arr
        x = np.array(x, dtype=dt, copy=True)  # X 朝向的独立工作块
        p_axis = 1 - split_axis if transpose else split_axis
        blocks.append([p.lo, p.hi, p_axis, x, p])
    p_axis = blocks[0][2]
    if any(blk[2] != p_axis for blk in blocks):
        raise RuntimeError("覆盖族内部分块轴不一致")

    tr_events: dict[str, Any] = {
        "mode": mode,
        "transpose_contract": transpose,
        "global_gram_shape": [m, m],
        "iterations": spec.iterations,
        "gram_block_pairs": [],
        "apply_contribution_pairs": [],
    }
    if trace is not None:
        trace.update(tr_events)

    # ---- 全局 Frobenius 范数（float64 求和，与到达顺序无关：已按坐标排序）----
    norm2 = 0.0
    for blk in blocks:
        x = blk[3]
        norm2 += float(np.sum(x.astype(np.float64) ** 2, dtype=np.float64))
    norm = float(np.sqrt(norm2))
    all_zero = norm == 0.0

    if all_zero:
        for blk in blocks:
            blk[3] = np.zeros_like(blk[3])
    else:
        scale = dt.type(1.0 / norm)
        for blk in blocks:
            blk[3] = (blk[3] * scale).astype(dt, copy=False)

        k = len(blocks)

        def form_gram_col() -> np.ndarray:
            # X 按列分块：A = sum_j X_j X_j^T；每个不相交坐标块恰好贡献一次。
            G = np.zeros((m, m), dtype=dt)
            pairs = 0
            for lo, hi, _, xj, _ in blocks:
                G += xj @ xj.T
                pairs += 1
            # 对称化消除浮点漂移（数学上 A 必对称）。
            G = (G + G.T) * dt.type(0.5)
            tr_events["gram_block_pairs"].append({"kind": "column_contrib", "count": pairs})
            return G

        def form_gram_row() -> np.ndarray:
            # X 按行分块：只算无序块对 i<=j 一次，(j,i) 用转置镜像。
            G = np.zeros((m, m), dtype=dt)
            pairs = 0
            for i in range(k):
                li, hi, _, xi, _ = blocks[i]
                for j in range(i, k):
                    lj, hj, _, xj, _ = blocks[j]
                    gij = xi @ xj.T
                    G[li:hi, lj:hj] = gij
                    pairs += 1
                    if i != j:
                        G[lj:hj, li:hi] = gij.T
            tr_events["gram_block_pairs"].append({"kind": "unordered_pairs_i<=j", "count": pairs})
            return G

        def apply_polynomial_col(Bm: np.ndarray) -> int:
            # 列块：X_i' = a X_i + B X_i（B 已含全部跨片 Gram 贡献）。
            for blk in blocks:
                xi = blk[3]
                blk[3] = (a * xi + Bm @ xi).astype(dt, copy=False)
            return k  # 每块一次全局 B 作用

        def apply_polynomial_row(Bm: np.ndarray) -> int:
            # 行块：(B X)_i = sum_k B_ik X_k —— 显式计算跨切分轴贡献。
            pair_count = 0
            new_blocks: list[list[Any]] = []
            for i in range(k):
                li, hi, _, xi, src = blocks[i]
                yi = (a * xi).copy()
                for lj, hj, _, xj, _ in blocks:
                    yi += Bm[li:hi, lj:hj] @ xj
                    pair_count += 1
                new_blocks.append([li, hi, 0, yi.astype(dt, copy=False), src])
            blocks[:] = new_blocks
            return pair_count

        form_gram = form_gram_col if p_axis == 1 else form_gram_row
        apply_poly = apply_polynomial_col if p_axis == 1 else apply_polynomial_row

        A = form_gram()
        for _it in range(spec.iterations):
            A2 = A @ A
            Bm = (b * A + c * A2).astype(dt, copy=False)
            n_apply_pairs = apply_poly(Bm)
            tr_events["apply_contribution_pairs"].append(n_apply_pairs)
            # 公开迭代每轮用“当前 X”重新形成 Gram（最后一轮后无需再算）。
            if _it + 1 < spec.iterations:
                A = form_gram()

    # ---- 转回逻辑朝向并打包物理布局 ----
    out_blocks: list[BlockUpdate] = []
    final_norm2 = 0.0
    finite = True
    for lo, hi, _pa, x_iter, src in blocks:
        x_logic = x_iter.T if transpose else x_iter
        x_logic = np.asarray(x_logic, dtype=dt)
        if not np.all(np.isfinite(x_logic)):
            finite = False
        final_norm2 += float(np.sum(x_logic.astype(np.float64) ** 2, dtype=np.float64))
        x_phys = pack_physical(x_logic, src.layout)
        out_blocks.append(
            BlockUpdate(
                partition=src.partition,
                axis=split_axis,
                lo=src.lo,
                hi=src.hi,
                layout=src.layout,
                x_logical=x_logic,
                x_physical=np.asarray(x_phys, dtype=dt),
            )
        )

    result = ShardedMuonResult(
        mode=mode,
        transpose=transpose,
        shape=(R, C),
        blocks=out_blocks,
        final_norm=float(np.sqrt(final_norm2)),
        finite=finite,
        trace=tr_events,
    )
    if trace is not None:
        trace["result_norm"] = result.final_norm
        trace["finite"] = finite
    return result


@dataclass
class PrecisionAttempt:
    mode: str
    max_error: float
    final_norm: float
    error_tolerance: float
    norm_limit: float
    error_passed: bool
    norm_passed: bool
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_error": self.max_error,
            "final_norm": self.final_norm,
            "error_tolerance": self.error_tolerance,
            "norm_limit": self.norm_limit,
            "error_passed": self.error_passed,
            "norm_passed": self.norm_passed,
            "finite": self.finite,
            "accepted": self.error_passed and self.norm_passed and self.finite,
        }


@dataclass
class MatrixEvaluation:
    """一个逻辑矩阵在某一步的精度门评估结论。"""

    matrix_id: str
    spec: MatrixSpec
    iteration_spec: IterationSpec
    selected_mode: str | None
    result: ShardedMuonResult | None
    reference_x: np.ndarray | None
    attempts: list[PrecisionAttempt]
    reasons: list[dict[str, Any]] = field(default_factory=list)
    chosen_axis: int | None = None

    @property
    def deliverable(self) -> bool:
        return self.selected_mode is not None and self.result is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix_id,
            "selected_precision": self.selected_mode,
            "chosen_axis": self.chosen_axis,
            "deliverable": self.deliverable,
            "attempts": [a.to_dict() for a in self.attempts],
            "reasons": self.reasons,
            "shards": [b.to_meta() for b in (self.result.blocks if self.result else [])],
            "final_norm": self.result.final_norm if self.result else None,
        }


def evaluate_matrix(
    matrix_spec: MatrixSpec,
    coverage: Coverage,
    v_pieces: Sequence[ShardPiece],
    iteration_spec: IterationSpec | None = None,
    *,
    modes: Sequence[str] = PRECISION_ORDER,
    group_mode: str | None = None,
    reference_x: np.ndarray | None = None,
) -> MatrixEvaluation:
    """按精度从低到高试候选步，用 float64 同迭代定义做误差门。

    ``group_mode`` 给定时，整组只在该精度下评估（组内精度统一）；
    否则按 ``modes`` 逐级升级直到误差与范数两道门都通过。
    """
    ispec = iteration_spec or IterationSpec()
    modes = tuple(normalize_mode(m) for m in (modes or PRECISION_ORDER))

    # 高精度基准：优先用调用方给的完整参考；否则用同一套分块代数跑 float64，
    # 它与全矩阵 muon_reference 在浮点误差内一致，且不需要整矩阵副本。
    if reference_x is None:
        ref_result = sharded_muon(matrix_spec.shape, v_pieces, ispec, mode="float64")
        reference_x = _assemble_logical(matrix_spec.shape, ref_result.blocks)
    reference_x = np.asarray(reference_x, dtype=np.float64)

    evaln = MatrixEvaluation(
        matrix_id=matrix_spec.matrix_id,
        spec=matrix_spec,
        iteration_spec=ispec,
        selected_mode=None,
        result=None,
        reference_x=reference_x,
        attempts=[],
        chosen_axis=coverage.axis,
    )

    trial_modes: Sequence[str]
    if group_mode is not None:
        trial_modes = (normalize_mode(group_mode),)
    else:
        trial_modes = modes

    last_result: ShardedMuonResult | None = None
    for mode in trial_modes:
        cand = sharded_muon(matrix_spec.shape, v_pieces, ispec, mode=mode)
        last_result = cand
        # 逐块比对同一逻辑坐标，块最大误差的最大值即整矩阵最大绝对元素误差。
        max_err = np.inf
        if cand.finite:
            for bu in cand.blocks:
                ref_slice = (
                    reference_x[bu.lo:bu.hi, :] if bu.axis == 0 else reference_x[:, bu.lo:bu.hi]
                )
                # 注意 BlockUpdate.axis 存的是原始切分轴。
                diff = np.abs(bu.x_logical.astype(np.float64) - ref_slice)
                bu.x_reference = ref_slice
                bu.block_error = float(diff.max()) if diff.size else 0.0
                max_err = bu.block_error if not np.isfinite(max_err) else max(max_err, bu.block_error)
        attempt = PrecisionAttempt(
            mode=mode,
            max_error=float(max_err),
            final_norm=cand.final_norm,
            error_tolerance=matrix_spec.error,
            norm_limit=matrix_spec.max_norm,
            error_passed=bool(np.isfinite(max_err) and max_err <= matrix_spec.error),
            norm_passed=bool(cand.final_norm <= matrix_spec.max_norm),
            finite=cand.finite,
        )
        evaln.attempts.append(attempt)
        if attempt.to_dict()["accepted"]:
            evaln.selected_mode = mode
            evaln.result = cand
            # 选中的结果里也保留参考切片与块误差。
            for bu in cand.blocks:
                ax = bu.axis
                ref_slice = (
                    reference_x[bu.lo:bu.hi, :] if ax == 0 else reference_x[:, bu.lo:bu.hi]
                )
                bu.x_reference = ref_slice
                bu.block_error = float(
                    np.abs(bu.x_logical.astype(np.float64) - ref_slice).max()
                )
            return evaln

    # 没有模式被接受：给出最贴近的失败原因（不缩放范数）。
    attempts = evaln.attempts
    if attempts and all(not a.error_passed or not a.finite for a in attempts):
        best = min(attempts, key=lambda a: (not a.finite, a.max_error if np.isfinite(a.max_error) else np.inf))
        evaln.reasons.append(
            {
                "code": "precision_error",
                "observed_max_error": best.max_error,
                "tolerance": matrix_spec.error,
                "mode": best.mode,
            }
        )
    norm_fail = [a for a in attempts if a.error_passed and not a.norm_passed]
    if norm_fail:
        a = norm_fail[-1]
        evaln.reasons.append(
            {
                "code": "norm_exceeded",
                "observed_norm": a.final_norm,
                "max_norm": matrix_spec.max_norm,
                "mode": a.mode,
                "scaled_to_comply": False,
            }
        )
    if not evaln.reasons and last_result is not None and not last_result.finite:
        evaln.reasons.append({"code": "precision_error", "observed_max_error": float("inf")})
    return evaln


def _assemble_logical(shape: tuple[int, int], blocks: Sequence[BlockUpdate]) -> np.ndarray:
    """仅用于测试/参考基准的整矩阵装配（小矩阵或训练侧使用）。"""
    R, C = int(shape[0]), int(shape[1])
    full = np.zeros((R, C), dtype=np.float64)
    for bu in blocks:
        if bu.axis == 0:
            full[bu.lo:bu.hi, :] = bu.x_logical
        else:
            full[:, bu.lo:bu.hi] = bu.x_logical
    return full


def assemble_logical_x(shape: tuple[int, int], result: ShardedMuonResult) -> np.ndarray:
    """把一次分片结果装回完整逻辑 X（测试/交接核对用）。"""
    return _assemble_logical(shape, result.blocks)


def full_reference_for(
    shape: tuple[int, int], v_full: np.ndarray, spec: IterationSpec | None = None
) -> np.ndarray:
    """完整 float64 参考的薄封装，便于调用方做独立比对。"""
    return muon_reference(v_full, spec or IterationSpec())
