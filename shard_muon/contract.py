"""契约类型：迭代规格、逻辑矩阵描述、延迟原因码。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

import numpy as np

# 公开 Muon/Newton-Schulz 三项多项式系数（与采集记录一致）。
ABC_MUON5 = (3.4445, -4.775, 2.0315)

# 精度从低到高。capture.json 的 precision_modes 也是这三种名字。
PRECISION_ORDER = ("binary16", "float32", "float64")
DTYPE_OF_MODE = {
    "binary16": np.float16,
    "float16": np.float16,
    "fp16": np.float16,
    "float32": np.float32,
    "fp32": np.float32,
    "float": np.float32,
    "float64": np.float64,
    "fp64": np.float64,
    "double": np.float64,
}
MODE_OF_DTYPE = {
    np.dtype(np.float16): "binary16",
    np.dtype(np.float32): "float32",
    np.dtype(np.float64): "float64",
}


def normalize_mode(mode: str | np.dtype | type) -> str:
    """把各种精度写法归一到 ``binary16/float32/float64``。"""
    if isinstance(mode, np.dtype):
        return MODE_OF_DTYPE[mode]
    if isinstance(mode, type) and issubclass(mode, np.generic):
        return MODE_OF_DTYPE[np.dtype(mode)]
    key = str(mode)
    if key in PRECISION_ORDER:
        return key
    if key in DTYPE_OF_MODE:
        dt = DTYPE_OF_MODE[key]
        return MODE_OF_DTYPE[np.dtype(dt)]
    raise ValueError(f"未知精度模式: {mode!r}，可选 {PRECISION_ORDER}")


# 延迟/失败原因码。机器可读，附带中文说明。
class REASON:
    WAITING_DATA = "waiting_data"
    VERSION_MISMATCH = "version_mismatch"
    MISSING_COVERAGE = "missing_coverage"
    CORRUPT_SHARD = "corrupt_shard"
    ABSENT_SHARD = "absent_shard"
    BAD_SHAPE = "bad_shape"
    CONFLICTING_SHARDS = "conflicting_shards"
    MIXED_AXIS = "mixed_axis"
    PRECISION_ERROR = "precision_error"
    NORM_EXCEEDED = "norm_exceeded"
    UNKNOWN_MATRIX = "unknown_matrix"
    ALREADY_COMMITTED = "already_committed"
    NOT_READY = "not_ready"


REASON_ZH = {
    REASON.WAITING_DATA: "步骤仍在等待分片数据",
    REASON.VERSION_MISMATCH: "分片版本与逻辑矩阵当前代次不一致（旧步数据不能充数）",
    REASON.MISSING_COVERAGE: "有效分片未形成完整且不相交的坐标覆盖",
    REASON.CORRUPT_SHARD: "分片损坏（含 NaN/Inf 等非有限值），不能按零处理",
    REASON.ABSENT_SHARD: "分片张量缺席，不能按零处理",
    REASON.BAD_SHAPE: "分片形状与逻辑坐标区间/物理轴序声明不符",
    REASON.CONFLICTING_SHARDS: "同一坐标区间的有效分片内容互相冲突",
    REASON.MIXED_AXIS: "同一覆盖族内混用了不同切分轴",
    REASON.PRECISION_ERROR: "所有精度下实际回传更新均未通过 float64 误差声明",
    REASON.NORM_EXCEEDED: "更新最终 Frobenius 范数超过图绑定上限（不缩放冒充合规）",
    REASON.UNKNOWN_MATRIX: "上报引用了模型描述中不存在的逻辑矩阵",
    REASON.ALREADY_COMMITTED: "同一步已经提交，重复确认只回查回执",
    REASON.NOT_READY: "矩阵尚未具备可交付结果",
}


@dataclass(frozen=True)
class IterationSpec:
    """当时使用的迭代规格快照（回执里原样保存）。"""

    iterations: int = 4
    a: float = ABC_MUON5[0]
    b: float = ABC_MUON5[1]
    c: float = ABC_MUON5[2]
    beta: float = 0.9
    lr: float = 0.004
    transpose_contract: str = (
        "if logical_rows > logical_cols: X=v.T, iterate on thin rows, transpose back"
    )
    normalization: str = "X / Frobenius_norm(v); exact zero -> zero update, no division"
    polynomial: str = "X <- a*X + b*(X X^T) X + c*(X X^T)^2 X; Gram rebuilt each iteration"
    error_metric: str = "max_abs_element_error(delivered_X, float64_same_iteration_X)"

    @property
    def abc(self) -> tuple[float, float, float]:
        return (self.a, self.b, self.c)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["abc"] = [self.a, self.b, self.c]
        return d

    @classmethod
    def from_records(cls, records: Mapping[str, Any]) -> "IterationSpec":
        abc = records.get("abc", ABC_MUON5)
        return cls(
            iterations=int(records.get("iterations", 4)),
            a=float(abc[0]),
            b=float(abc[1]),
            c=float(abc[2]),
            beta=float(records.get("beta", 0.9)),
            lr=float(records.get("lr", 0.004)),
        )


DEFAULT_ITERATION_SPEC = IterationSpec()


@dataclass
class MatrixSpec:
    """逻辑矩阵描述：完整逻辑形状、当前代次、误差声明与步范数上限。"""

    matrix_id: str
    shape: tuple[int, int]
    generation: int = 0
    error: float = np.inf
    max_norm: float = np.inf
    group: str | None = None

    def __post_init__(self) -> None:
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        if self.group is None:
            # 默认每个逻辑矩阵自成一个精度/范数组（图里逐矩阵绑定上限）。
            self.group = self.matrix_id

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def thin(self) -> int:
        return min(self.shape)

    @property
    def long(self) -> int:
        return max(self.shape)

    @property
    def transpose(self) -> bool:
        """转置契约只看完整逻辑形状：行数严格大于列数才转。"""
        return self.rows > self.cols

    @classmethod
    def from_dict(cls, matrix_id: str, d: Mapping[str, Any]) -> "MatrixSpec":
        return cls(
            matrix_id=matrix_id,
            shape=tuple(d["shape"]),
            generation=int(d.get("generation", 0)),
            error=float(d.get("error", np.inf)),
            max_norm=float(d.get("max_norm", np.inf)),
            group=d.get("group"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix_id,
            "shape": list(self.shape),
            "generation": self.generation,
            "error": self.error,
            "max_norm": self.max_norm,
            "group": self.group,
            "transpose": self.transpose,
        }
