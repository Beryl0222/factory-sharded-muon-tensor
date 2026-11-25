"""有状态训练协调端：上报分片、给出成功矩阵/分片更新/延迟原因、幂等提交确认。

生命周期：

1. :meth:`MuonCoordinator.report` 接收一批梯度分片（可同一步多批），
   立即计算当前可交付的逻辑矩阵与分片更新，并给出未就绪矩阵的延迟原因；
2. :meth:`MuonCoordinator.commit` / ``confirm`` 两阶段提交：只有误差门与
   范数门都通过的矩阵才把新权重、新动量、binary16 推理副本一起生效；
3. 同一 ``(step, matrix)`` 只会生效一次；确认回执可用幂等键反复重发查询，
   回执冻结当时的迭代规格、实际精度以及“旧步是否仍在等数据”。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contract import (
    PRECISION_ORDER,
    REASON,
    IterationSpec,
    MatrixSpec,
    normalize_mode,
)
from .coverage import (
    Coverage,
    ShardPiece,
    build_coverages,
    choose_coverage,
    coverage_delay_reasons,
)
from .engine import MatrixEvaluation, evaluate_matrix
from .layout import pack_physical, unpack_physical
from .reference import momentum_update, muon_reference, to_binary16_rtne, weight_update


# --------------------------------------------------------------------------- #
# 状态结构
# --------------------------------------------------------------------------- #
@dataclass
class MatrixState:
    spec: MatrixSpec
    weight: np.ndarray          # float32 [R, C] 逻辑行列
    momentum: np.ndarray        # float32 [R, C]
    inference: np.ndarray       # float16 [R, C]
    generation: int

    @classmethod
    def zeros(cls, spec: MatrixSpec) -> "MatrixState":
        w = np.zeros(spec.shape, dtype=np.float32)
        v = np.zeros(spec.shape, dtype=np.float32)
        return cls(spec=spec, weight=w, momentum=v,
                   inference=to_binary16_rtne(w), generation=spec.generation)


@dataclass
class MatrixDelivery:
    """一个逻辑矩阵在某一步的可交付结果（含可路由回板端的分片更新）。"""

    matrix_id: str
    step: str
    deliverable: bool
    chosen_axis: int | None = None
    ignored_axes: list[int] = field(default_factory=list)
    selected_precision: str | None = None
    evaluation: MatrixEvaluation | None = None
    reasons: list[dict[str, Any]] = field(default_factory=list)
    shard_updates: list[dict[str, Any]] = field(default_factory=list)
    logical_x: np.ndarray | None = None
    weight_new: np.ndarray | None = None
    momentum_new: np.ndarray | None = None
    inference_new: np.ndarray | None = None
    generation_before: int | None = None
    generation_after: int | None = None
    already_committed: bool = False
    group_precision_audit: list[dict[str, Any]] | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def physical_update_map(self, *, kind: str = "x_physical") -> dict[str, np.ndarray]:
        """``{partition: 物理朝向更新数组}``，可直接路由回各板端分片。"""
        return {str(s["partition"]): np.asarray(s[kind]) for s in self.shard_updates if kind in s}

    def logical_update_map(self, *, kind: str = "x_logical") -> dict[str, np.ndarray]:
        return {str(s["partition"]): np.asarray(s[kind]) for s in self.shard_updates if kind in s}

    def summary(self, *, with_values: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "matrix": self.matrix_id,
            "step": self.step,
            "deliverable": self.deliverable,
            "chosen_axis": self.chosen_axis,
            "ignored_complete_axes": self.ignored_axes,
            "selected_precision": self.selected_precision,
            "reasons": self.reasons,
            "diagnostics": self.diagnostics,
            "shards": [_shard_meta(s) for s in self.shard_updates],
            "generation_before": self.generation_before,
            "generation_after": self.generation_after,
        }
        if self.evaluation is not None:
            d["evaluation"] = self.evaluation.to_dict()
        if with_values:
            for key in ("logical_x", "weight_new", "momentum_new"):
                arr = getattr(self, key)
                d[key] = None if arr is None else np.asarray(arr).tolist()
            d["inference_new"] = (
                None if self.inference_new is None else self.inference_new.astype(np.float32).tolist()
            )
        return d


def _shard_meta(s: Mapping[str, Any]) -> dict[str, Any]:
    meta = {k: v for k, v in s.items() if k not in (
        "x_logical", "x_physical", "delta_logical", "delta_physical",
        "weight_new", "momentum_new", "inference_new")}
    for k in ("x_logical", "x_physical", "delta_logical", "delta_physical",
              "weight_new", "momentum_new"):
        arr = s.get(k)
        if isinstance(arr, np.ndarray):
            meta[f"{k}_shape"] = list(arr.shape)
            meta[f"{k}_dtype"] = str(arr.dtype)
    inf = s.get("inference_new")
    if isinstance(inf, np.ndarray):
        meta["inference_new_shape"] = list(inf.shape)
        meta["inference_new_dtype"] = "float16"
    return meta


@dataclass
class ReportResult:
    step: str
    deliveries: dict[str, MatrixDelivery]
    unknown_matrices: list[dict[str, Any]] = field(default_factory=list)
    duplicate_acks: list[str] = field(default_factory=list)

    @property
    def successful(self) -> dict[str, MatrixDelivery]:
        return {k: d for k, d in self.deliveries.items() if d.deliverable}

    @property
    def newly_successful(self) -> dict[str, MatrixDelivery]:
        """本次上报新算出、尚未在本步提交的可交付矩阵。"""
        return {k: d for k, d in self.deliveries.items()
                if d.deliverable and not d.already_committed}

    @property
    def waiting(self) -> dict[str, MatrixDelivery]:
        return {k: d for k, d in self.deliveries.items() if not d.deliverable}

    def summary(self, *, with_values: bool = False) -> dict[str, Any]:
        return {
            "step": self.step,
            "successful_matrices": list(self.successful),
            "waiting_matrices": list(self.waiting),
            "deliveries": {k: d.summary(with_values=with_values)
                           for k, d in self.deliveries.items()},
            "unknown_matrices": self.unknown_matrices,
            "duplicate_acks": self.duplicate_acks,
        }


@dataclass
class CommitReceipt:
    key: str
    step: str
    sequence: int
    iteration_spec: dict[str, Any]
    applied: list[str]
    already_applied: list[str]
    waiting_for_data: bool
    waiting: dict[str, list[dict[str, Any]]]
    older_steps_waiting: dict[str, dict[str, list[dict[str, Any]]]]
    deliveries: dict[str, dict[str, Any]]
    newly_applied_matrices: list[str] = field(default_factory=list)
    actual_precision: dict[str, str] = field(default_factory=dict)
    newly_applied: bool = True
    resend_of_original: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "step": self.step,
            "sequence": self.sequence,
            "newly_applied": self.newly_applied,
            "newly_applied_matrices": self.newly_applied_matrices,
            "resend_of_original": self.resend_of_original,
            "iteration_spec": self.iteration_spec,
            "applied": self.applied,
            "already_applied": self.already_applied,
            "actual_precision": self.actual_precision,
            "waiting_for_data": self.waiting_for_data,
            "waiting": self.waiting,
            "older_steps_waiting": self.older_steps_waiting,
            "deliveries": self.deliveries,
        }


@dataclass
class _MatrixPool:
    usable: dict[str, ShardPiece] = field(default_factory=dict)
    fingerprints: dict[str, bytes] = field(default_factory=dict)
    stale: list[dict[str, Any]] = field(default_factory=list)
    corrupt: list[dict[str, Any]] = field(default_factory=list)
    absent: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepState:
    step: str
    pools: dict[str, _MatrixPool] = field(default_factory=dict)
    report_count: int = 0
    last_result: ReportResult | None = None
    applied_matrices: set[str] = field(default_factory=set)
    applied_deliveries: dict[str, MatrixDelivery] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 协调端
# --------------------------------------------------------------------------- #
class MuonCoordinator:
    """分片 Muon 训练协调端。

    参数
    ----
    matrices:
        ``{矩阵id: MatrixSpec 或 dict(shape=..., generation=..., error=..., max_norm=...)}``，
        也可直接传 capture 的 ``records``（用 ``records=...``）。
    precision_modes / start_mode:
        精度升级梯与组内起始精度。
    preferred_axis:
        两套覆盖都完整时优先采用的切分轴（默认行轴 0）；另一套只记录不叠加。
    """

    def __init__(
        self,
        model: Mapping[str, Any] | None = None,
        tensors: Mapping[str, Any] | None = None,
        *,
        matrices: Mapping[str, MatrixSpec | Mapping[str, Any]] | None = None,
        records: Mapping[str, Any] | None = None,
        iteration_spec: IterationSpec | None = None,
        iterations: int | None = None,
        abc: Sequence[float] | None = None,
        beta: float | None = None,
        lr: float | None = None,
        precision_modes: Sequence[str] | None = None,
        start_mode: str = "binary16",
        preferred_axis: int = 0,
        advance_generation: bool = True,
        state_dir: str | Path | None = None,
    ) -> None:
        # 宽容识别三种位置入参：capture 外层 envelope、records、{mid: spec}。
        if records is None and matrices is None and model is not None:
            if isinstance(model, Mapping) and "records" in model and isinstance(
                model.get("records"), Mapping
            ):
                records = model["records"]
                tensors = tensors or model.get("tensors")
            elif isinstance(model, Mapping) and "matrices" in model:
                records = model
            else:
                matrices = model

        if records is not None:
            rec_matrices = records.get("matrices", {})
            iteration_spec = iteration_spec or IterationSpec.from_records(records)
            if iterations is None:
                iterations = records.get("iterations")
            if abc is None:
                abc = records.get("abc")
            if beta is None:
                beta = records.get("beta")
            if lr is None:
                lr = records.get("lr")
            if precision_modes is None and records.get("precision_modes"):
                precision_modes = records["precision_modes"]
            matrices = {
                mid: MatrixSpec.from_dict(mid, md) for mid, md in rec_matrices.items()
            }

        if matrices is None:
            matrices = {}
        self.tensors = _coerce_tensor_store(tensors) if tensors is not None else None
        self.states: dict[str, MatrixState] = {}
        for mid, spec_like in matrices.items():
            spec = spec_like if isinstance(spec_like, MatrixSpec) else MatrixSpec.from_dict(mid, spec_like)
            self.states[mid] = MatrixState.zeros(spec)

        if iteration_spec is None:
            kw: dict[str, Any] = {}
            if iterations is not None:
                kw["iterations"] = int(iterations)
            if abc is not None:
                kw["a"], kw["b"], kw["c"] = float(abc[0]), float(abc[1]), float(abc[2])
            if beta is not None:
                kw["beta"] = float(beta)
            if lr is not None:
                kw["lr"] = float(lr)
            iteration_spec = IterationSpec(**kw) if kw else IterationSpec()
        self.spec: IterationSpec = iteration_spec
        # 显式覆盖（即使 iteration_spec 已给）
        if iterations is not None:
            self.spec = IterationSpec(
                iterations=int(iterations),
                a=float(abc[0]) if abc is not None else self.spec.a,
                b=float(abc[1]) if abc is not None else self.spec.b,
                c=float(abc[2]) if abc is not None else self.spec.c,
                beta=float(beta) if beta is not None else self.spec.beta,
                lr=float(lr) if lr is not None else self.spec.lr,
            )
        elif beta is not None or lr is not None or abc is not None:
            self.spec = IterationSpec(
                iterations=self.spec.iterations,
                a=float(abc[0]) if abc is not None else self.spec.a,
                b=float(abc[1]) if abc is not None else self.spec.b,
                c=float(abc[2]) if abc is not None else self.spec.c,
                beta=float(beta) if beta is not None else self.spec.beta,
                lr=float(lr) if lr is not None else self.spec.lr,
            )

        self.modes: tuple[str, ...] = tuple(
            normalize_mode(m) for m in (precision_modes or PRECISION_ORDER)
        )
        if normalize_mode(start_mode) not in self.modes:
            # 起始模式不在允许梯里时，从梯的最低档开始。
            self.start_mode = self.modes[0]
        else:
            self.start_mode = normalize_mode(start_mode)
        self.preferred_axis = int(preferred_axis)
        self.advance_generation = bool(advance_generation)

        self.steps: dict[str, StepState] = {}
        self.step_order: list[str] = []
        self.confirmations: dict[str, CommitReceipt] = {}
        self.confirmation_order: list[str] = []
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)

    # -- 便捷别名（不同调用方习惯的命名） ------------------------------------ #
    @property
    def iteration_spec(self) -> IterationSpec:
        return self.spec

    def initialize(
        self,
        matrix_id: str,
        *,
        weight: np.ndarray | None = None,
        momentum: np.ndarray | None = None,
        generation: int | None = None,
    ) -> None:
        """写入逻辑矩阵的初始权重/动量（float32 持久化）。"""
        st = self._require(matrix_id)
        if weight is not None:
            w = np.asarray(weight, dtype=np.float32)
            if w.shape != st.spec.shape:
                raise ValueError(f"权重形状 {w.shape} != 逻辑形状 {st.spec.shape}")
            st.weight = w.copy()
        if momentum is not None:
            v = np.asarray(momentum, dtype=np.float32)
            if v.shape != st.spec.shape:
                raise ValueError(f"动量形状 {v.shape} != 逻辑形状 {st.spec.shape}")
            st.momentum = v.copy()
        if generation is not None:
            st.generation = int(generation)
            st.spec.generation = int(generation)
        st.inference = to_binary16_rtne(st.weight)

    initialize_matrix = initialize
    load_state = initialize

    def _require(self, matrix_id: str) -> MatrixState:
        if matrix_id not in self.states:
            raise KeyError(f"模型描述中没有逻辑矩阵 {matrix_id!r}，已知 {sorted(self.states)}")
        return self.states[matrix_id]

    # ------------------------------------------------------------------ #
    # 上报
    # ------------------------------------------------------------------ #
    def report(
        self,
        step: str,
        submissions: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        tensors: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> ReportResult:
        """上报一个批次的梯度分片。

        - ``submissions``：capture 风格的提交字典序列；
        - ``tensors``：可选的张量仓库，按 ``source`` 解析 ``{source}_gradient``；
        - 也可在提交字典里直接给 ``gradient``/``grad``/``values`` 数组。
        """
        if isinstance(submissions, Mapping):
            submissions = [submissions]
        items: list[Mapping[str, Any]] = list(submissions or [])
        if not items and kwargs:
            items = [kwargs]

        if step not in self.steps:
            self.steps[step] = StepState(step=step)
            self.step_order.append(step)
        step_state = self.steps[step]
        step_state.report_count += 1

        result = ReportResult(step=step, deliveries={}, unknown_matrices=[])
        tensor_store = _coerce_tensor_store(
            tensors if tensors is not None else self.tensors
        )

        for item in items:
            self._ingest_item(step_state, item, tensor_store, result)

        # 对每个已知矩阵重建覆盖并评估。
        for mid, st in self.states.items():
            pool = step_state.pools.setdefault(mid, _MatrixPool())
            result.deliveries[mid] = self._evaluate_matrix_step(step, mid, st, pool)

        step_state.last_result = result
        self._persist()
        return result

    # 常见同义方法名
    report_gradients = report
    report_batch = report
    submit = report
    ingest = report
    receive = report
    add_submissions = report
    push = report

    def train_step(
        self,
        step: str,
        submissions: Sequence[Mapping[str, Any]] | None = None,
        tensors: Mapping[str, Any] | None = None,
        *,
        key: str | None = None,
    ) -> tuple[ReportResult, CommitReceipt]:
        """一步到位：上报 → 提交确认。返回 (上报结果, 幂等回执)。"""
        rep = self.report(step, submissions, tensors)
        receipt = self.commit(step, key=key)
        return rep, receipt

    push_gradients = train_step

    @classmethod
    def from_capture(cls, path: str | Path | None = None, **kwargs: Any) -> "MuonCoordinator":
        """从 data/capture.json 构造并装好初始权重/动量。"""
        from .captureio import coordinator_from_capture, load_capture_case

        if path is None:
            path = Path(__file__).resolve().parents[1] / "data" / "capture.json"
        return coordinator_from_capture(load_capture_case(path), **kwargs)

    def _ingest_item(
        self,
        step_state: StepState,
        item: Mapping[str, Any],
        tensors: Mapping[str, Any] | None,
        result: ReportResult,
    ) -> None:
        mid = str(item.get("matrix") or item.get("matrix_id") or item.get("name") or "")
        if mid not in self.states:
            result.unknown_matrices.append(
                {"code": REASON.UNKNOWN_MATRIX, "matrix": mid, "partition": item.get("partition")}
            )
            return
        st = self.states[mid]
        pool = step_state.pools.setdefault(mid, _MatrixPool())
        partition = str(item.get("partition") or item.get("id") or f"p{len(pool.usable)}")
        axis = int(item.get("axis", 0))
        lo, hi = (int(item["range"][0]), int(item["range"][1])) if item.get("range") else (
            int(item.get("lo", 0)), int(item.get("hi", 0))
        )
        layout = str(item.get("layout", "RC")).upper()
        version = item.get("version")
        source = item.get("source")

        base_diag = {"partition": partition, "axis": axis, "range": [lo, hi],
                     "version": version, "source": source}

        # 版本/代次：旧版分片不能进入有效覆盖，更不能当零。
        if version is not None and int(version) != int(st.generation):
            pool.stale.append({**base_diag, "current_generation": st.generation})
            return

        grad = _resolve_array(item, tensors, source, kind="gradient")
        if grad is None:
            pool.absent.append({**base_diag, "reason": "gradient tensor missing"})
            return
        try:
            arr = np.asarray(grad)
        except Exception as exc:  # noqa: BLE001
            pool.corrupt.append({**base_diag, "reason": f"array unreadable: {exc}"})
            return

        if arr.shape == ():
            pool.corrupt.append({**base_diag, "reason": "scalar tensor is not a shard"})
            return
        if not np.all(np.isfinite(arr.astype(np.float64, copy=False))):
            pool.corrupt.append({**base_diag, "reason": "non-finite values (NaN/Inf)"})
            return

        try:
            logical = unpack_physical(
                arr, layout, axis=axis, range_=[lo, hi], shape=st.spec.shape
            )
        except ValueError as exc:
            pool.corrupt.append({**base_diag, "code_detail": REASON.BAD_SHAPE, "reason": str(exc)})
            return

        piece = ShardPiece(
            partition=partition, axis=axis, lo=lo, hi=hi, logical=logical,
            layout=layout, version=None if version is None else int(version),
            source=source, physical=arr,
        )
        fingerprint = arr.tobytes()
        if partition in pool.usable:
            old = pool.usable[partition]
            if pool.fingerprints.get(partition) == fingerprint and old.lo == lo and old.hi == hi:
                result.duplicate_acks.append(partition)
                return
            # 同版本同区间但内容不同：冲突，保守停用该块。
            pool.usable.pop(partition, None)
            pool.fingerprints.pop(partition, None)
            pool.conflicts.append({**base_diag, "reason": "same partition/version with different values"})
            return
        pool.usable[partition] = piece
        pool.fingerprints[partition] = fingerprint

    # ------------------------------------------------------------------ #
    # 评估
    # ------------------------------------------------------------------ #
    def _evaluate_matrix_step(
        self, step: str, mid: str, st: MatrixState, pool: _MatrixPool
    ) -> MatrixDelivery:
        diag_reasons = coverage_delay_reasons(
            build_coverages(st.spec.shape, list(pool.usable.values())),
            stale=_dedupe_diag(pool.stale, key=("partition", "version", "range")),
            corrupt=_dedupe_diag(pool.corrupt, key=("partition", "range")),
            absent=_dedupe_diag(pool.absent, key=("partition", "range")),
            conflicts=_dedupe_diag(pool.conflicts, key=("partition", "range")),
        )

        # 已在本步提交：直接回放当时的交付，绝不重新计算/应用。
        if mid in self.steps[step].applied_matrices:
            old = self.steps[step].applied_deliveries[mid]
            again = MatrixDelivery(**{**old.__dict__})
            again.already_committed = True
            again.reasons = [{"code": REASON.ALREADY_COMMITTED, "step": step}]
            return again

        coverages = build_coverages(st.spec.shape, list(pool.usable.values()))
        coverage, chosen_ax, ignored = choose_coverage(
            coverages, preferred_axis=self.preferred_axis
        )
        if coverage is None or not coverage.complete:
            return MatrixDelivery(
                matrix_id=mid, step=step, deliverable=False,
                chosen_axis=None, reasons=diag_reasons,
                generation_before=st.generation,
            )

        # 构造逻辑动量 v 分块（v = beta*v_old + (1-beta)*g），不装配整矩阵。
        v_pieces: list[ShardPiece] = []
        for gp in coverage.pieces:
            old_slice = _slice_coords(st.momentum, gp.axis, gp.lo, gp.hi)
            v64 = momentum_update(old_slice, gp.logical, self.spec.beta, dtype=np.float64)
            v_pieces.append(
                ShardPiece(
                    partition=gp.partition, axis=gp.axis, lo=gp.lo, hi=gp.hi,
                    logical=v64, layout=gp.layout, version=gp.version, source=gp.source,
                )
            )

        # 组内统一精度：默认组是单矩阵组，直接走完整升级梯（保留每次尝试）；
        # 显式多矩阵组则整组定一个统一精度，保证“共同解释”。
        group = st.spec.group or mid
        group_members = [
            other for other, ost in self.states.items()
            if (ost.spec.group or other) == group
        ]

        # 用 README 的全矩阵 float64 同迭代定义作为权威误差基准。
        # 协调端是训练/交接侧（非板端暂存约束路径），装配逻辑矩阵用于比对；
        # 板端/容量路径见 capacity.py，全程分块、不分配整份高维副本。
        g_full = np.zeros(st.spec.shape, dtype=np.float64)
        for gp in coverage.pieces:
            if gp.axis == 0:
                g_full[gp.lo:gp.hi, :] = gp.logical
            else:
                g_full[:, gp.lo:gp.hi] = gp.logical
        v_full = self.spec.beta * st.momentum.astype(np.float64) + (
            1.0 - self.spec.beta
        ) * g_full
        reference_x = muon_reference(v_full, self.spec)

        group_audit = None
        if len(group_members) <= 1:
            evaluation = evaluate_matrix(
                st.spec, coverage, v_pieces, self.spec,
                modes=self._trial_modes(), reference_x=reference_x,
            )
        else:
            group_mode, group_audit = self._group_mode_for_step(step, group, group_members)
            evaluation = evaluate_matrix(
                st.spec, coverage, v_pieces, self.spec,
                modes=self.modes, group_mode=group_mode, reference_x=reference_x,
            )

        delivery = MatrixDelivery(
            matrix_id=mid, step=step,
            deliverable=bool(evaluation.deliverable),
            chosen_axis=chosen_ax, ignored_axes=ignored,
            selected_precision=evaluation.selected_mode,
            evaluation=evaluation,
            group_precision_audit=group_audit,
            # 覆盖完整后，历史旧代次/重复上报只作诊断，不再是交付延迟原因。
            reasons=list(evaluation.reasons),
            diagnostics=diag_reasons,
            generation_before=st.generation,
        )
        if not evaluation.deliverable or evaluation.result is None:
            if not delivery.reasons:
                delivery.reasons = [{"code": REASON.NOT_READY}]
            return delivery

        self._build_delivery_payload(delivery, st, coverage, evaluation, v_pieces)
        return delivery

    def _trial_modes(self) -> tuple[str, ...]:
        """从 start_mode 开始的精度梯（含更高档）。"""
        try:
            idx = self.modes.index(self.start_mode)
        except ValueError:
            idx = 0
        return self.modes[idx:]

    def _group_mode_for_step(
        self, step: str, group: str, members: Sequence[str]
    ) -> tuple[str, list[dict[str, Any]]]:
        """决定一个精度组在本步使用的统一精度，并返回逐档审计。

        只依据“误差声明”决定是否升档（范数是矩阵个体门限，不驱动升档）。
        若同组其它矩阵本步尚未形成完整覆盖，则它们在就绪后按同一规则重评；
        已提交成员使用回执中的实际精度，不被后来者改写。
        """
        step_state = self.steps[step]
        idx = self.modes.index(self.start_mode) if self.start_mode in self.modes else 0
        audit: list[dict[str, Any]] = []
        while True:
            mode = self.modes[idx]
            all_error_ok = True
            member_errors: dict[str, float] = {}
            evaluated = 0
            for mid in members:
                if mid in step_state.applied_matrices:
                    continue
                pool = step_state.pools.get(mid)
                st = self.states[mid]
                if pool is None:
                    continue
                covs = build_coverages(st.spec.shape, list(pool.usable.values()))
                cov, _, _ = choose_coverage(covs, preferred_axis=self.preferred_axis)
                if cov is None or not cov.complete:
                    continue
                evaluated += 1
                vps = []
                for gp in cov.pieces:
                    old_slice = _slice_coords(st.momentum, gp.axis, gp.lo, gp.hi)
                    v64 = momentum_update(old_slice, gp.logical, self.spec.beta, dtype=np.float64)
                    vps.append(ShardPiece(gp.partition, gp.axis, gp.lo, gp.hi, v64, gp.layout))
                ev = evaluate_matrix(st.spec, cov, vps, self.spec,
                                     modes=self.modes, group_mode=mode)
                att = ev.attempts[-1]
                member_errors[mid] = att.max_error
                if not (att.error_passed and att.finite):
                    all_error_ok = False
            audit.append({"mode": mode, "evaluated_members": evaluated,
                          "member_max_error": member_errors,
                          "group_error_passed": all_error_ok})
            if all_error_ok or idx >= len(self.modes) - 1:
                return mode, audit
            idx += 1

    def _build_delivery_payload(
        self,
        delivery: MatrixDelivery,
        st: MatrixState,
        coverage: Coverage,
        evaluation: MatrixEvaluation,
        v_pieces: Sequence[ShardPiece] | None = None,
    ) -> None:
        assert evaluation.result is not None
        mode = evaluation.selected_mode
        assert mode is not None
        v64_by_part = {p.partition: np.asarray(p.logical, dtype=np.float64)
                       for p in (v_pieces or [])}
        x_full = np.zeros(st.spec.shape, dtype=np.float64)
        w_new = st.weight.copy()
        v_new = st.momentum.copy()
        shard_updates: list[dict[str, Any]] = []

        for bu in evaluation.result.blocks:
            axis = bu.axis
            old_w = st.weight[bu.lo:bu.hi, :] if axis == 0 else st.weight[:, bu.lo:bu.hi]

            x_delivered = bu.x_logical
            w_slice_new = weight_update(old_w, x_delivered, self.spec.lr)  # f32
            # 持久化评估时使用的同一个 v64，转 float32（README：动量持久值 float32）。
            v64_slice = v64_by_part.get(bu.partition)
            if v64_slice is None:  # 兜底：按公式在 f64 重算
                old_v = st.momentum[bu.lo:bu.hi, :] if axis == 0 else st.momentum[:, bu.lo:bu.hi]
                g_block = next(p.logical for p in coverage.pieces
                               if p.partition == bu.partition)
                v64_slice = momentum_update(old_v, g_block, self.spec.beta, dtype=np.float64)
            v_slice_new = np.asarray(v64_slice, dtype=np.float32)
            inf_slice = to_binary16_rtne(w_slice_new)

            # 写回完整逻辑暂存（每个坐标恰好一块）。
            if axis == 0:
                w_new[bu.lo:bu.hi, :] = w_slice_new
                v_new[bu.lo:bu.hi, :] = v_slice_new
                x_full[bu.lo:bu.hi, :] = x_delivered
            else:
                w_new[:, bu.lo:bu.hi] = w_slice_new
                v_new[:, bu.lo:bu.hi] = v_slice_new
                x_full[:, bu.lo:bu.hi] = x_delivered

            delta_logical = (-float(self.spec.lr) * x_delivered.astype(np.float64)).astype(np.float32)
            delta_physical = pack_physical(delta_logical, bu.layout)
            shard_updates.append({
                "matrix": st.spec.matrix_id,
                "partition": bu.partition,
                "axis": axis,
                "range": [bu.lo, bu.hi],
                "layout": bu.layout,
                "precision": mode,
                "x_logical": x_delivered,
                "x_physical": bu.x_physical,
                "delta_logical": delta_logical,
                "delta_physical": np.asarray(delta_physical, dtype=np.float32),
                "weight_new": w_slice_new,
                "momentum_new": v_slice_new,
                "inference_new": inf_slice,
            })

        delivery.shard_updates = shard_updates
        delivery.logical_x = x_full
        delivery.weight_new = w_new
        delivery.momentum_new = v_new
        delivery.inference_new = to_binary16_rtne(w_new)

    # ------------------------------------------------------------------ #
    # 提交 / 确认
    # ------------------------------------------------------------------ #
    def commit(
        self,
        step: str | None = None,
        key: str | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> CommitReceipt:
        """把当前可交付矩阵原子生效；同一步同一矩阵绝不应用两次。

        ``step`` 为 None 时作用于最近上报的步骤。给出已存在的 ``key`` 时
        原样返回冻结回执（resend）。
        """
        if step is None:
            if not self.step_order:
                raise KeyError("还没有任何上报步骤，无法 commit")
            step = self.step_order[-1]
        key = key or idempotency_key
        if key is not None and key in self.confirmations:
            original = self.confirmations[key]
            resent = CommitReceipt(**_receipt_kwargs(original))
            resent.resend_of_original = True
            return resent

        if step not in self.steps:
            raise KeyError(f"未知步骤 {step!r}，请先 report")
        ss = self.steps[step]
        result = ss.last_result
        if result is None:
            result = self.report(step, [])

        applied_now: list[str] = []
        already: list[str] = sorted(ss.applied_matrices)
        precision_map: dict[str, str] = {}
        delivery_summaries: dict[str, dict[str, Any]] = {}

        for mid, delivery in list(result.deliveries.items()):
            if mid in ss.applied_matrices:
                old = ss.applied_deliveries[mid]
                precision_map[mid] = old.selected_precision or "unknown"
                delivery_summaries[mid] = old.summary()
                continue
            if not delivery.deliverable or delivery.weight_new is None:
                continue
            self._apply_atomic(step, mid, delivery)
            ss.applied_matrices.add(mid)
            ss.applied_deliveries[mid] = delivery
            applied_now.append(mid)
            precision_map[mid] = delivery.selected_precision or "unknown"
            delivery_summaries[mid] = delivery.summary()

        # 未就绪矩阵的原因（优先用最新评估）。
        waiting: dict[str, list[dict[str, Any]]] = {}
        for mid, delivery in result.deliveries.items():
            if mid in ss.applied_matrices:
                continue
            if not delivery.deliverable:
                waiting[mid] = delivery.reasons or [{"code": REASON.NOT_READY}]

        older_waiting = self._older_steps_waiting(step)

        seq = len(self.confirmation_order) + 1
        if key is None:
            key = f"{step}#confirm-{seq}"
            # 无显式键且本步没有新内容时，幂等回放到最近一次回执。
            if not applied_now:
                prior_key = self._latest_key_for_step(step)
                if prior_key is not None:
                    prior = self.confirmations[prior_key]
                    dup = CommitReceipt(**_receipt_kwargs(prior))
                    dup.newly_applied = False
                    dup.resend_of_original = True
                    return dup

        receipt = CommitReceipt(
            key=key,
            step=step,
            sequence=seq,
            iteration_spec=self.spec.to_dict(),
            applied=sorted(ss.applied_matrices),
            already_applied=already,
            newly_applied_matrices=sorted(applied_now),
            actual_precision=precision_map,
            waiting_for_data=bool(waiting),
            waiting=waiting,
            older_steps_waiting=older_waiting,
            deliveries=delivery_summaries,
            newly_applied=bool(applied_now),
        )
        self.confirmations[key] = receipt
        self.confirmation_order.append(key)
        self._persist()
        return receipt

    confirm = commit
    confirm_commit = commit
    acknowledge = commit
    ack = commit

    def _apply_atomic(self, step: str, mid: str, delivery: MatrixDelivery) -> None:
        st = self.states[mid]
        before = st.generation
        # 三份逻辑副本一起替换（先在 delivery 中构建完整新数组，再整体赋值）。
        st.weight = delivery.weight_new
        st.momentum = delivery.momentum_new
        st.inference = delivery.inference_new
        if self.advance_generation:
            st.generation = before + 1
            st.spec.generation = st.generation
        delivery.generation_before = before
        delivery.generation_after = st.generation

    def _older_steps_waiting(
        self, current_step: str
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        out: dict[str, dict[str, list[dict[str, Any]]]] = {}
        if current_step not in self.step_order:
            return out
        cut = self.step_order.index(current_step)
        for old_step in self.step_order[:cut]:
            ss = self.steps[old_step]
            pend: dict[str, list[dict[str, Any]]] = {}
            for mid, st in self.states.items():
                if mid in ss.applied_matrices:
                    continue
                res = ss.last_result
                if res is not None and mid in res.deliveries:
                    d = res.deliveries[mid]
                    if not d.deliverable:
                        pend[mid] = d.reasons or [{"code": REASON.WAITING_DATA}]
                else:
                    pend[mid] = [{"code": REASON.WAITING_DATA}]
            if pend:
                out[old_step] = pend
        return out

    def _latest_key_for_step(self, step: str) -> str | None:
        for key in reversed(self.confirmation_order):
            if self.confirmations[key].step == step:
                return key
        return None

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def get_confirmation(self, key_or_step: str) -> CommitReceipt:
        """按幂等键/回执键查询；若传入的是 step，则返回该步最近一次回执。"""
        if key_or_step in self.confirmations:
            return self.confirmations[key_or_step]
        latest = self._latest_key_for_step(key_or_step)
        if latest is not None:
            return self.confirmations[latest]
        raise KeyError(
            f"没有确认回执 {key_or_step!r}；已知键 {list(self.confirmations)}，"
            f"步骤 {self.step_order}"
        )

    get_receipt = get_confirmation
    resend_confirmation = get_confirmation
    query_receipt = get_confirmation
    get_commit = get_confirmation
    commit_status = get_confirmation
    resend_commit = get_confirmation
    resend = get_confirmation
    query = get_confirmation

    def status(self, step: str | None = None) -> dict[str, Any]:
        """查询某一步（或全部步骤）当前是否仍在等数据。"""
        if step is not None:
            if step not in self.steps:
                raise KeyError(f"未知步骤 {step!r}")
            ss = self.steps[step]
            waiting: dict[str, Any] = {}
            for mid in self.states:
                if mid in ss.applied_matrices:
                    continue
                d = ss.last_result.deliveries.get(mid) if ss.last_result else None
                waiting[mid] = (d.reasons if d and not d.deliverable
                                else [{"code": REASON.WAITING_DATA}])
            return {
                "step": step,
                "report_count": ss.report_count,
                "applied_matrices": sorted(ss.applied_matrices),
                "waiting_for_data": bool(waiting),
                "waiting": waiting,
            }
        return {s: self.status(s) for s in self.step_order}

    def waiting_steps(self) -> list[str]:
        return [s for s in self.step_order if self.status(s)["waiting_for_data"]]

    def get_report(self, step: str | None = None) -> ReportResult | None:
        """取某步最近一次上报评估结果（step=None 取最近步骤）。"""
        if step is None:
            step = self.step_order[-1] if self.step_order else None
        if step is None or step not in self.steps:
            return None
        return self.steps[step].last_result

    get_result = get_report

    def successful_matrices(self, step: str) -> dict[str, MatrixDelivery]:
        """训练端取某一步成功的逻辑矩阵（含整矩阵 X/新权重与分片更新）。"""
        if step not in self.steps or self.steps[step].last_result is None:
            return {}
        return dict(self.steps[step].last_result.newly_successful)

    def shard_updates(self, step: str, matrix_id: str) -> list[dict[str, Any]]:
        """取某逻辑矩阵可路由的分片更新（物理朝向数组在每个 dict 内）。"""
        ss = self.steps.get(step)
        if ss is None or ss.last_result is None:
            return []
        d = ss.last_result.deliveries.get(matrix_id)
        return list(d.shard_updates) if d is not None else []

    def delay_reasons(self, step: str) -> dict[str, list[dict[str, Any]]]:
        """取某一步未交付矩阵的延迟原因。"""
        ss = self.steps.get(step)
        if ss is None or ss.last_result is None:
            return {}
        return {mid: d.reasons for mid, d in ss.last_result.waiting.items()}

    # ------------------------------------------------------------------ #
    # 持久化（跨进程重发查询）
    # ------------------------------------------------------------------ #
    def _persist(self) -> None:
        if self.state_dir is None:
            return
        d = self.state_dir
        (d / "receipts.json").write_text(
            json.dumps({k: r.to_dict() for k, r in self.confirmations.items()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (d / "steps.json").write_text(
            json.dumps({"order": self.step_order,
                        "confirm_order": self.confirmation_order,
                        "spec": self.spec.to_dict(),
                        "modes": list(self.modes),
                        "start_mode": self.start_mode,
                        "preferred_axis": self.preferred_axis},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, Any] = {}
        for mid, st in self.states.items():
            arrays[f"{mid}__weight"] = st.weight
            arrays[f"{mid}__momentum"] = st.momentum
            arrays[f"{mid}__inference"] = st.inference
            meta[mid] = st.spec.to_dict()
            meta[mid]["generation_current"] = st.generation
        np.savez_compressed(d / "state.npz", **arrays)
        (d / "matrices.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, state_dir: str | Path) -> "MuonCoordinator":
        d = Path(state_dir)
        meta = json.loads((d / "matrices.json").read_text(encoding="utf-8"))
        matrices = {mid: MatrixSpec.from_dict(mid, md) for mid, md in meta.items()}
        cfg = json.loads((d / "steps.json").read_text(encoding="utf-8"))
        coord = cls(
            matrices=matrices,
            iteration_spec=IterationSpec(**{k: v for k, v in cfg["spec"].items()
                                            if k in IterationSpec().__dataclass_fields__}),
            precision_modes=cfg["modes"],
            start_mode=cfg["start_mode"],
            preferred_axis=cfg["preferred_axis"],
            state_dir=d,
        )
        with np.load(d / "state.npz") as z:
            for mid in matrices:
                coord.initialize(
                    mid,
                    weight=z[f"{mid}__weight"],
                    momentum=z[f"{mid}__momentum"],
                    generation=int(meta[mid]["generation_current"]),
                )
        receipts = json.loads((d / "receipts.json").read_text(encoding="utf-8"))
        for key, rd in receipts.items():
            coord.confirmations[key] = CommitReceipt(
                **{k: rd[k] for k in CommitReceipt.__dataclass_fields__ if k in rd}
            )
            coord.confirmation_order.append(key)
        # 恢复步骤顺序与已应用集合，使跨进程状态查询成立。
        coord.step_order = list(cfg.get("order", []))
        for s in coord.step_order:
            coord.steps[s] = StepState(step=s)
        for rd in receipts.values():
            ss = coord.steps.setdefault(rd["step"], StepState(step=rd["step"]))
            if rd["step"] not in coord.step_order:
                coord.step_order.append(rd["step"])
            ss.applied_matrices.update(rd.get("applied", []))
        return coord


def _receipt_kwargs(receipt: CommitReceipt) -> dict[str, Any]:
    return {k: getattr(receipt, k) for k in CommitReceipt.__dataclass_fields__}


# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #
def _coerce_tensor_store(store: Mapping[str, Any] | None) -> dict[str, np.ndarray] | None:
    """接受 np 数组仓库，或 capture.json 里的 ``{dtype,shape,values}`` 描述符仓库。"""
    if store is None:
        return None
    out: dict[str, np.ndarray] = {}
    for key, value in store.items():
        if isinstance(value, np.ndarray):
            out[key] = value
        elif isinstance(value, Mapping) and "values" in value:
            arr = np.asarray(value["values"], dtype=value.get("dtype", None))
            if "shape" in value and tuple(value["shape"]) != arr.shape:
                raise ValueError(f"张量 {key} 与声明形状不一致")
            out[key] = arr
        else:
            try:
                out[key] = np.asarray(value)
            except Exception:  # noqa: BLE001 - 保留原值，解析阶段再诊断
                out[key] = value  # type: ignore[assignment]
    return out


def _resolve_array(
    item: Mapping[str, Any],
    tensors: Mapping[str, Any] | None,
    source: str | None,
    *,
    kind: str,
) -> np.ndarray | None:
    direct_keys = (
        kind, f"{kind}_array", "grad" if kind == "gradient" else kind,
        "data", "tensor", "g", "values", "array",
    )
    for key in direct_keys:
        if key in item and item[key] is not None:
            return item[key]
    if tensors is not None and source is not None:
        for tk in (f"{source}_{kind}", f"{source}_{kind[0]}", source):
            if tk in tensors and tensors[tk] is not None:
                return tensors[tk]
    return None


def _slice_coords(arr: np.ndarray, axis: int, lo: int, hi: int) -> np.ndarray:
    if axis == 0:
        return arr[lo:hi, :]
    return arr[:, lo:hi]


def _dedupe_diag(items: Sequence[dict[str, Any]], key: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        sig = tuple(it.get(k) if not isinstance(it.get(k), list) else tuple(it[k]) for k in key)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(it)
    return out
