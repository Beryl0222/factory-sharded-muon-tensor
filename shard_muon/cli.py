"""命令行：回放 capture 故事、查询持久化回执、跑容量巡检。

用法::

    python -m shard_muon replay [data/capture.json] [--state-dir DIR] [--values]
    python -m shard_muon receipt <state_dir> <confirmation_key>
    python -m shard_muon capacity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .captureio import play_capture
from .capacity import run_capacity_replay, scratch_plan
from .service import MuonCoordinator


def _attempts(delivery) -> list[dict[str, Any]]:
    if delivery.evaluation is None:
        return []
    return [a.to_dict() for a in delivery.evaluation.attempts]


def replay_report(path: str | Path | None = None, state_dir: str | Path | None = None,
                  *, with_values: bool = False) -> dict[str, Any]:
    play = play_capture(path, state_dir=state_dir) if path else play_capture(state_dir=state_dir)
    c = play.coordinator

    def matrix_block(rep, mid: str) -> dict[str, Any]:
        d = rep.deliveries[mid]
        out = d.summary(with_values=with_values)
        out["attempts"] = _attempts(d)
        if with_values and d.logical_x is not None:
            out["logical_x_values"] = np.asarray(d.logical_x).tolist()
        return out

    q1 = play.first_report.deliveries["Q"]
    r1 = play.first_report.deliveries["R"]
    q2 = play.second_report.deliveries["Q"]

    # 列分片等价性数值核对。
    col_q = play.column_report.deliveries.get("Q")
    equiv: dict[str, Any] = {}
    if col_q is not None and col_q.logical_x is not None and q2.logical_x is not None:
        equiv["row_vs_column_maxerr"] = float(
            np.abs(col_q.logical_x - q2.logical_x).max()
        )
        equiv["row_precision"] = q2.selected_precision
        equiv["column_precision"] = col_q.selected_precision

    # 幂等：两次 confirm-2 后权重只应用一次的旁证。
    q_state = c.states["Q"]
    r_state = c.states["R"]

    return {
        "step": play.first_receipt.step,
        "iteration_spec": c.spec.to_dict(),
        "batch1": {
            "successful": list(play.first_report.successful),
            "waiting": list(play.first_report.waiting),
            "R": matrix_block(play.first_report, "R"),
            "Q": matrix_block(play.first_report, "Q"),
        },
        "commit1": play.first_receipt.to_dict(),
        "batch2": {
            "successful": list(play.second_report.successful),
            "waiting": list(play.second_report.waiting),
            "Q": matrix_block(play.second_report, "Q"),
        },
        "commit2": play.second_receipt.to_dict(),
        "resend_confirm2": play.resend_receipt.to_dict(),
        "idempotency_check": {
            "confirm2_resend_is_original_key": play.resend_receipt.key == play.second_receipt.key,
            "Q_generation_after": q_state.generation,
            "R_generation_after": r_state.generation,
            "Q_weight_f32_dtype": str(q_state.weight.dtype),
            "Q_momentum_dtype": str(q_state.momentum.dtype),
            "Q_inference_dtype": str(q_state.inference.dtype),
        },
        "column_equivalence": equiv,
        "state_dir": str(c.state_dir) if c.state_dir else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m shard_muon", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    p_replay = sub.add_parser("replay", help="回放 capture.json 的两批上报与提交")
    default_data = Path(__file__).resolve().parents[1] / "data" / "capture.json"
    p_replay.add_argument("path", nargs="?", type=Path, default=default_data)
    p_replay.add_argument("--state-dir", type=Path, default=None)
    p_replay.add_argument("--values", action="store_true", help="输出张量数值")

    p_rcpt = sub.add_parser("receipt", help="从持久化目录查询确认回执")
    p_rcpt.add_argument("state_dir", type=Path)
    p_rcpt.add_argument("key", type=str)

    p_cap = sub.add_parser("capacity", help="容量估算与可选大矩阵种子回放")
    p_cap.add_argument("--run", action="store_true", help="实际跑 18432x432 核对（数秒）")

    # 不带子命令时等价于 replay。
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = ["replay"]
    args = parser.parse_args(argv)
    cmd = args.cmd

    if cmd == "replay":
        report = replay_report(args.path, args.state_dir, with_values=args.values)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    if cmd == "receipt":
        coord = MuonCoordinator.load(args.state_dir)
        rc = coord.get_confirmation(args.key)
        print(json.dumps(rc.to_dict(), ensure_ascii=False, indent=2, default=_json_default))
        return 0
    if cmd == "capacity":
        if args.run:
            rep = run_capacity_replay()
        else:
            rep = {
                "row_plan": scratch_plan(split_axis=0).to_dict(),
                "column_plan": scratch_plan(split_axis=1).to_dict(),
            }
        print(json.dumps(rep, ensure_ascii=False, indent=2, default=_json_default))
        return 0
    parser.error(f"未知命令 {cmd}")
    return 2


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
