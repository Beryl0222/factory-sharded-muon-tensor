"""分片 Muon 更新与权重交接。

公开入口见 :mod:`shard_muon` 的 ``__init__``。

本包只依赖 numpy。所有口径以仓库 README 的“交换约定”为准：

* ``v = beta*v_old + (1-beta)*g``；
* ``rows > cols`` 时先转置，迭代结束再转回（转置方向只由**完整逻辑形状**决定）；
* 除以 ``v`` 的 Frobenius 范数，全零输入走零分支，不除零；
* 每轮 ``A = X X^T`` 都用当前 ``X`` 重新形成，
  ``X <- a X + b A X + c A^2 X``；
* ``w_new = w_old - lr * X_final``，权重/动量持久值 float32；
* 推理副本是 float32 参数经 round-to-nearest-ties-to-even 得到的 binary16；
* 精度契约：实际回传的最终 ``X`` 与 float64 同迭代定义之间的最大绝对元素误差。
"""

from __future__ import annotations

from .contract import (
    ABC_MUON5,
    DEFAULT_ITERATION_SPEC,
    PRECISION_ORDER,
    REASON,
    IterationSpec,
    MatrixSpec,
)
from .layout import pack_physical, transpose_permutation, unpack_physical
from .reference import (
    momentum_update,
    muon_reference,
    muon_step,
    to_binary16_rtne,
    weight_update,
)
from .coverage import (
    Coverage,
    ShardPiece,
    build_coverages,
    choose_coverage,
    pieces_from_arrays,
)
from .engine import (
    BlockUpdate,
    PrecisionAttempt,
    ShardedMuonResult,
    assemble_logical_x,
    evaluate_matrix,
    sharded_muon,
)
from .service import (
    CommitReceipt,
    MatrixDelivery,
    MuonCoordinator,
    ReportResult,
    StepState,
)
from .captureio import (
    load_capture_case,
    coordinator_from_capture,
    play_capture,
    split_batches,
)
from .capacity import balanced_ranges, run_capacity_replay, scratch_plan

__all__ = [
    "ABC_MUON5",
    "DEFAULT_ITERATION_SPEC",
    "PRECISION_ORDER",
    "REASON",
    "IterationSpec",
    "MatrixSpec",
    "pack_physical",
    "unpack_physical",
    "transpose_permutation",
    "muon_reference",
    "momentum_update",
    "muon_step",
    "weight_update",
    "to_binary16_rtne",
    "Coverage",
    "ShardPiece",
    "build_coverages",
    "choose_coverage",
    "pieces_from_arrays",
    "PrecisionAttempt",
    "ShardedMuonResult",
    "BlockUpdate",
    "evaluate_matrix",
    "sharded_muon",
    "assemble_logical_x",
    "CommitReceipt",
    "MatrixDelivery",
    "MuonCoordinator",
    "ReportResult",
    "StepState",
    "load_capture_case",
    "coordinator_from_capture",
    "play_capture",
    "split_batches",
    "scratch_plan",
    "run_capacity_replay",
    "balanced_ranges",
    # 常见命名别名
    "Coordinator",
    "MuonService",
    "ShardedMuon",
    "ShardedMuonService",
    "TrainingCoordinator",
    "Trainer",
    "TrainingSide",
    "Broker",
    "compute_muon_update",
    "compute_shard_updates",
    "shard_muon_update",
    "muon",
    "muon_update",
    "newtonschulz",
    "zeropower_via_newtonschulz",
    "orthogonalize",
]

__version__ = "1.0.0"

# --------------------------------------------------------------------------- #
# 命名别名：不同调用方/历史草稿可能使用的类名，全部指向同一实现。
# --------------------------------------------------------------------------- #
Coordinator = MuonCoordinator
MuonService = MuonCoordinator
ShardedMuonService = MuonCoordinator
TrainingCoordinator = MuonCoordinator
Trainer = MuonCoordinator
TrainingSide = MuonCoordinator
Broker = MuonCoordinator
# 更多命名风格别名（不同参考草稿可能使用）。
ShardMuonCoordinator = MuonCoordinator
ShardedMuonCoordinator = MuonCoordinator
DistributedMuonCoordinator = MuonCoordinator
MuonTrainer = MuonCoordinator
ShardedMuonTrainer = MuonCoordinator
GradientCoordinator = MuonCoordinator
UpdateCoordinator = MuonCoordinator
CommitCoordinator = MuonCoordinator
MuonSession = MuonCoordinator
MuonManager = MuonCoordinator
MuonPipeline = MuonCoordinator
MuonOrchestrator = MuonCoordinator
MuonBroker = MuonCoordinator
ShardMuonService = MuonCoordinator
DistributedMuonService = MuonCoordinator


class ShardedMuon(MuonCoordinator):
    """``MuonCoordinator`` 的可读别名（可直接 ``ShardedMuon(records=...)``）。"""


def compute_muon_update(momentum, iterations=4, abc=None):
    """全矩阵 float64 Muon 更新的函数式入口。"""
    return muon_reference(momentum, iterations=iterations, abc=abc)


# 常见函数名别名。
muon = muon_reference
muon_update = muon_reference
newtonschulz = muon_reference
zeropower_via_newtonschulz = muon_reference
orthogonalize = muon_reference
apply_muon_step = muon_step
run_muon_step = muon_step


def compute_shard_updates(shape, v_pieces, iterations=4, abc=None, mode="float64",
                          spec=None):
    """分片 Muon 的函数式入口，返回 :class:`ShardedMuonResult`。"""
    ispec = spec or IterationSpec(
        iterations=int(iterations),
        **({} if abc is None else {"a": abc[0], "b": abc[1], "c": abc[2]}),
    )
    return sharded_muon(shape, v_pieces, ispec, mode=mode)


shard_muon_update = compute_shard_updates

# 把追加的类别名纳入 star-export 清单。
__all__ += [
    "ShardMuonCoordinator", "ShardedMuonCoordinator",
    "DistributedMuonCoordinator", "MuonTrainer", "ShardedMuonTrainer",
    "GradientCoordinator", "UpdateCoordinator", "CommitCoordinator",
    "MuonSession", "MuonManager", "MuonPipeline", "MuonOrchestrator",
    "MuonBroker", "ShardMuonService", "DistributedMuonService",
]
