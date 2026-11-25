"""把 data/capture.json 接到分片 Muon 协调端，并回放两批上报过程。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from caseio import load_case

from .contract import IterationSpec, MatrixSpec
from .layout import unpack_physical
from .service import MuonCoordinator


@dataclass
class CaptureCase:
    path: Path
    records: dict[str, Any]
    tensors: dict[str, np.ndarray]

    @property
    def step(self) -> str:
        return str(self.records.get("step", "train"))

    @property
    def spec(self) -> IterationSpec:
        return IterationSpec.from_records(self.records)

    @property
    def row_submissions(self) -> list[dict[str, Any]]:
        return list(self.records.get("submissions", []))

    @property
    def alternative_submissions(self) -> list[dict[str, Any]]:
        return list(self.records.get("alternative_submissions", []))


def load_capture_case(path: str | Path) -> CaptureCase:
    records, tensors = load_case(Path(path))
    return CaptureCase(path=Path(path), records=records, tensors=tensors)


def _assemble_initial_state(
    records: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray],
    submissions: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """按提交声明的逻辑坐标，从各 source 的 weight/momentum 分片装出初始状态。

    同一 partition 只取第一份（capture 中 q1 的 v11/v12 共用 source=b）。
    """
    subs = submissions if submissions is not None else list(records.get("submissions", []))
    out: dict[str, dict[str, np.ndarray]] = {}
    seen: set[tuple[str, str]] = set()
    for sub in subs:
        mid = sub["matrix"]
        part = sub["partition"]
        key = (mid, part)
        if key in seen:
            continue
        seen.add(key)
        mspec = records["matrices"][mid]
        shape = tuple(mspec["shape"])
        axis = int(sub["axis"])
        lo, hi = int(sub["range"][0]), int(sub["range"][1])
        layout = sub.get("layout", "RC")
        source = sub.get("source", part)
        bucket = out.setdefault(
            mid,
            {
                "weight": np.zeros(shape, dtype=np.float32),
                "momentum": np.zeros(shape, dtype=np.float32),
            },
        )
        for kind in ("weight", "momentum"):
            tk = f"{source}_{kind}"
            if tk not in tensors:
                continue
            logical = unpack_physical(
                tensors[tk], layout, axis=axis, range_=[lo, hi], shape=shape
            )
            logical = np.asarray(logical, dtype=np.float32)
            if axis == 0:
                bucket[kind][lo:hi, :] = logical
            else:
                bucket[kind][:, lo:hi] = logical
    return out


def coordinator_from_capture(
    case: str | Path | CaptureCase | tuple[Mapping[str, Any], Mapping[str, np.ndarray]],
    *,
    state_dir: str | Path | None = None,
    start_mode: str = "binary16",
    precision_modes=None,
) -> MuonCoordinator:
    """构造带好初始权重/动量的协调端（默认从 binary16 候选起步自动升档）。"""
    if isinstance(case, CaptureCase):
        records, tensors = case.records, case.tensors
    elif isinstance(case, (str, Path)):
        c = load_capture_case(case)
        records, tensors = c.records, c.tensors
    else:
        records, tensors = case

    coord = MuonCoordinator(
        records=records,
        state_dir=state_dir,
        start_mode=start_mode,
        precision_modes=precision_modes or records.get("precision_modes"),
    )
    initial = _assemble_initial_state(records, tensors)
    for mid, arrays in initial.items():
        coord.initialize(mid, weight=arrays["weight"], momentum=arrays["momentum"],
                         generation=int(records["matrices"][mid].get("generation", 0)))
    return coord


def split_batches(records: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    """把 submissions 分成“首批”和“另一批（补报/重发）”。

    capture 的最后一条是 q1 升到 version=12 后的补报。通用规则：
    找到同一 partition 以更新 version 再次出现的位置，从那里起作为第二批；
    找不到时按最后一条切分。
    """
    subs = list(records.get("submissions", []))
    last_idx: dict[tuple[str, str], int] = {}
    split_at = len(subs)
    max_version_seen: dict[tuple[str, str], int] = {}
    for i, s in enumerate(subs):
        key = (s["matrix"], s["partition"])
        v = int(s.get("version", 0))
        if key in max_version_seen and v > max_version_seen[key]:
            split_at = min(split_at, i)
        max_version_seen[key] = max(max_version_seen.get(key, v), v)
        last_idx[key] = i
    if split_at == len(subs) and len(subs) > 1:
        # 没有更高版本，也可能最后一条是同版本重发：仍把最后一条当第二批资料。
        split_at = len(subs) - 1
    return subs[:split_at], subs[split_at:]


@dataclass
class CapturePlay:
    coordinator: MuonCoordinator
    first_report: Any
    first_receipt: Any
    second_report: Any
    second_receipt: Any
    resend_receipt: Any
    column_report: Any = None
    column_coordinator: MuonCoordinator | None = None


def play_capture(
    path: str | Path = Path(__file__).resolve().parents[1] / "data" / "capture.json",
    *,
    state_dir: str | Path | None = None,
) -> CapturePlay:
    """完整演示：首批 → 提交 R；补报 q1 → 提交 Q；重发确认；列分片等价回放。"""
    case = load_capture_case(path)
    records, tensors = case.records, case.tensors
    step = case.step
    batch1, batch2 = split_batches(records)

    coord = coordinator_from_capture(case, state_dir=state_dir)

    rep1 = coord.report(step, batch1, tensors)
    rc1 = coord.commit(step, key=f"{step}#confirm-1")

    rep2 = coord.report(step, batch2, tensors)
    rc2 = coord.commit(step, key=f"{step}#confirm-2")

    # 用同一幂等键重发确认：返回冻结回执，不二次应用。
    resend = coord.commit(step, key=f"{step}#confirm-2")

    # 等价列分片回放：全新协调端 + alternative_submissions。
    coord_col = coordinator_from_capture(case)
    col_subs = case.alternative_submissions
    # 列分片的初始权重/动量改用 column_* source 装配，验证 CR 物理轴序。
    initial_col = _assemble_initial_state(records, tensors, col_subs)
    for mid, arrays in initial_col.items():
        coord_col.initialize(
            mid, weight=arrays["weight"], momentum=arrays["momentum"],
            generation=int(records["matrices"][mid].get("generation", 0)),
        )
    col_report = coord_col.report(step, col_subs, tensors)
    coord_col.commit(step, key=f"{step}#column-confirm")

    return CapturePlay(
        coordinator=coord,
        first_report=rep1,
        first_receipt=rc1,
        second_report=rep2,
        second_receipt=rc2,
        resend_receipt=resend,
        column_report=col_report,
        column_coordinator=coord_col,
    )
