"""公开 Muon 迭代契约的全矩阵 float64 参考实现。

与 README“交换约定”逐字对应，也是精度门里的高精度比较基准。
分片引擎在数学上必须与本函数一致（两种切分轴、任意上报顺序）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .contract import ABC_MUON5, IterationSpec


def momentum_update(
    momentum_old: np.ndarray,
    gradient: np.ndarray,
    beta: float = 0.9,
    *,
    dtype: np.dtype | type | str = np.float64,
) -> np.ndarray:
    """``v = beta*v_old + (1-beta)*g``，在指定精度下计算。"""
    dt = np.dtype(dtype)
    vo = np.asarray(momentum_old, dtype=dt)
    g = np.asarray(gradient, dtype=dt)
    if vo.shape != g.shape:
        raise ValueError(f"动量形状 {vo.shape} 与梯度形状 {g.shape} 不一致")
    return (dt.type(beta) * vo + dt.type(1.0 - beta) * g).astype(dt, copy=False)


def muon_reference(
    momentum: np.ndarray,
    iterations: int | IterationSpec = 4,
    abc: Sequence[float] | None = None,
) -> np.ndarray:
    """完整逻辑矩阵的 float64 Muon/Newton-Schulz 迭代，返回逻辑朝向的 X_final。

    契约：

    * ``rows > cols`` 先转置，使薄维为行；结束转回；
    * 用整个 ``v`` 的 Frobenius 范数归一；范数为零（全零）时返回精确零；
    * 每轮用当前 ``X`` 重新形成 ``A=XX^T``，
      ``B=bA+cA^2; X=aX+BX``（与公开实现运算顺序一致）。
    """
    if isinstance(iterations, IterationSpec):
        spec = iterations
        n_iter = spec.iterations
        a, b, c = spec.abc
    else:
        n_iter = int(iterations)
        if abc is None:
            a, b, c = ABC_MUON5
        else:
            a, b, c = (float(abc[0]), float(abc[1]), float(abc[2]))

    V = np.asarray(momentum, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"Muon 需要二维矩阵，收到 {V.ndim} 维")
    transpose_back = V.shape[0] > V.shape[1]
    X = V.T.copy() if transpose_back else V.copy()

    norm = float(np.linalg.norm(X))
    if norm == 0.0:
        X_final = np.zeros_like(X)
    else:
        X = X / norm
        for _ in range(n_iter):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        X_final = X

    return X_final.T if transpose_back else X_final


def weight_update(
    weight_old: np.ndarray,
    x_final: np.ndarray,
    lr: float,
) -> np.ndarray:
    """``w_new = w_old - lr*X_final``，结果转 float32 持久值。"""
    w64 = np.asarray(weight_old, dtype=np.float64)
    x64 = np.asarray(x_final, dtype=np.float64)
    return (w64 - float(lr) * x64).astype(np.float32)


def to_binary16_rtne(weight_float32: np.ndarray) -> np.ndarray:
    """float32 输入按 round-to-nearest-ties-to-even 得到 binary16 推理副本。

    numpy 的 ``astype(np.float16)`` 使用偶数舍入。
    """
    w32 = np.asarray(weight_float32, dtype=np.float32)
    return w32.astype(np.float16)


def muon_step(
    weight: np.ndarray,
    momentum_old: np.ndarray,
    gradient: np.ndarray,
    beta: float = 0.9,
    lr: float = 0.004,
    iterations: int = 4,
    abc: Sequence[float] | None = ABC_MUON5,
) -> dict[str, np.ndarray]:
    """一步完整的全矩阵 Muon：动量更新 → 正交化 → 参数更新与推理副本。

    返回 ``x``(float64 最终正交更新)、``momentum``(float32)、
    ``weight``(float32)、``inference_binary16`` 与 ``delta``(=-lr*x, float32)。
    分片场景请用 :func:`shard_muon.sharded_muon` 或 :class:`MuonCoordinator`。
    """
    v64 = momentum_update(momentum_old, gradient, beta, dtype=np.float64)
    x64 = muon_reference(v64, iterations=iterations, abc=abc)
    w_new = weight_update(weight, x64, lr)
    return {
        "x": x64,
        "momentum": v64.astype(np.float32),
        "weight": w_new,
        "delta": (-float(lr) * x64).astype(np.float32),
        "inference_binary16": to_binary16_rtne(w_new),
    }
