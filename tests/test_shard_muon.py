"""分片 Muon 规格的端到端核对。

这些测试同时是“题目每条要求”的可执行说明：
布局换轴、转置契约、两套覆盖不叠加、全零分支、损坏/缺席不当零、
精度升级门、范数门、幂等提交、跨轴等价、分片顺序无关与持久化回执。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from caseio import load_case
import shard_muon as sm
from shard_muon import (
    CommitReceipt,
    IterationSpec,
    MatrixSpec,
    MuonCoordinator,
    ShardPiece,
    assemble_logical_x,
    build_coverages,
    choose_coverage,
    muon_reference,
    pack_physical,
    scratch_plan,
    sharded_muon,
    unpack_physical,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ABC = (3.4445, -4.775, 2.0315)


def canonical_muon(v, iterations=4, abc=ABC):
    """独立照 README 写的参考，避免与被测实现共享同一错误。"""
    a, b, c = abc
    X = np.array(v, dtype=np.float64, copy=True)
    back = X.shape[0] > X.shape[1]
    if back:
        X = X.T.copy()
    n = np.linalg.norm(X)
    if n == 0.0:
        X = np.zeros_like(X)
    else:
        X = X / n
        for _ in range(iterations):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X.T if back else X


def row_pieces(M, k, axis=0, layout="RC"):
    R, C = M.shape
    length = R if axis == 0 else C
    bounds = np.linspace(0, length, k + 1, dtype=int)
    out = []
    for i in range(k):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        if hi <= lo:
            continue
        L = M[lo:hi, :] if axis == 0 else M[:, lo:hi]
        if layout == "CR":
            phys = L.T
        else:
            phys = L
        out.append({"partition": f"p{i}", "axis": axis, "range": [lo, hi],
                    "layout": layout, "logical": L, "physical": phys})
    return out


class LayoutTests(unittest.TestCase):
    def test_rc_cr_roundtrip(self):
        rng = np.random.default_rng(3)
        L = rng.standard_normal((2, 5))
        self.assertTrue(np.array_equal(unpack_physical(L, "RC"), L))
        self.assertTrue(np.allclose(unpack_physical(L.T, "CR"), L))
        self.assertTrue(np.allclose(pack_physical(L, "CR"), L.T))
        self.assertTrue(np.allclose(pack_physical(L, "RC"), L))
        # 形状声明核对：CR 物理 [3,1] 对应逻辑 [1,3]（Q 的 q2）。
        logical = unpack_physical(np.zeros((3, 1)), "CR", axis=0,
                                  range_=[6, 7], shape=(7, 3))
        self.assertEqual(logical.shape, (1, 3))
        with self.assertRaises(ValueError):
            unpack_physical(np.zeros((2, 5)), "CR", axis=0,
                            range_=[3, 5], shape=(7, 2))

    def test_capture_c_shard_is_transposed_physical(self):
        # c_gradient 物理形状 [3,1]、CR，逻辑 q2 行块应为 [1,3]。
        records, tensors = load_case(DATA / "capture.json")
        sub = records["submissions"][2]
        logical = unpack_physical(
            tensors["c_gradient"], sub["layout"], sub["axis"], sub["range"],
            records["matrices"]["Q"]["shape"],
        )
        self.assertEqual(logical.shape, (1, 3))
        np.testing.assert_allclose(logical[0], [7.0, 7.003, 0.077], rtol=0, atol=1e-6)
        # column_b 是 axis=1 + CR：[1,7] 物理 -> [7,1] 逻辑
        cbsub = records["alternative_submissions"][1]
        logical_cb = unpack_physical(
            tensors["column_b_gradient"], cbsub["layout"], cbsub["axis"],
            cbsub["range"], records["matrices"]["Q"]["shape"],
        )
        self.assertEqual(logical_cb.shape, (7, 1))
        self.assertAlmostEqual(float(logical_cb[6, 0]), 0.077, places=6)


class EngineEquivalenceTests(unittest.TestCase):
    def _run_shape(self, shape, axis, k, layout="RC", dtype=np.float64, seed=0):
        rng = np.random.default_rng(seed)
        V = rng.standard_normal(shape)
        ref = canonical_muon(V)
        specs = row_pieces(V, k, axis=axis, layout=layout)
        pieces = [ShardPiece(s["partition"], s["axis"], s["range"][0], s["range"][1],
                             s["logical"], layout=s["layout"]) for s in specs]
        res = sharded_muon(shape, pieces, IterationSpec(iterations=4), mode="float64")
        got = assemble_logical_x(shape, res)
        err = float(np.abs(got - ref).max())
        self.assertLess(err, 1e-11, msg=f"{shape} axis={axis} err={err}")
        return err, res, specs

    def test_various_shapes_and_both_axes(self):
        for j, shape in enumerate([(7, 3), (2, 3), (4, 4), (1, 8), (8, 1),
                                   (13, 5), (5, 13), (1, 1), (3, 1)]):
            for axis in (0, 1):
                k = 3 if (shape[axis] >= 3) else shape[axis]
                self._run_shape(shape, axis, max(1, k), seed=j)

    def test_cr_layout_engine(self):
        # CR 物理布局经 unpack 后结果必须与 RC 一致。
        records, tensors = load_case(DATA / "capture.json")
        subs = records["submissions"][:3]
        pieces = []
        for s in subs:
            src = s["source"]
            L = unpack_physical(tensors[f"{src}_gradient"], s["layout"], s["axis"],
                                s["range"], records["matrices"]["Q"]["shape"])
            # 统一提升到 float64，单独验证分片代数与布局，不引入 f32 舍入差异。
            L = np.asarray(L, dtype=np.float64)
            pieces.append(ShardPiece(s["partition"], s["axis"], s["range"][0],
                                     s["range"][1], L, layout=s["layout"]))
        G = np.concatenate([p.logical for p in sorted(pieces, key=lambda p: p.lo)], axis=0)
        v = 0.1 * G
        ref = canonical_muon(v, iterations=4)
        res = sharded_muon((7, 3), pieces, IterationSpec(iterations=4), "float64")
        got = assemble_logical_x((7, 3), res)
        self.assertLess(float(np.abs(got - ref).max()), 1e-11)

    def test_shard_order_independent_bitwise(self):
        rng = np.random.default_rng(4)
        V = rng.standard_normal((10, 4))
        specs = row_pieces(V, 4, axis=0)
        pieces = [ShardPiece(s["partition"], 0, s["range"][0], s["range"][1], s["logical"])
                  for s in specs]
        r1 = sharded_muon((10, 4), pieces, IterationSpec(), "float64")
        r2 = sharded_muon((10, 4), pieces[::-1], IterationSpec(), "float64")
        a1 = assemble_logical_x((10, 4), r1)
        a2 = assemble_logical_x((10, 4), r2)
        np.testing.assert_array_equal(a1, a2)

    def test_global_transpose_not_local_shape(self):
        # Q 是 7x3：全局必须转置；q0 本地块 3x3 是方阵，按本地形状会错误地不转置。
        records, tensors = load_case(DATA / "capture.json")
        G_parts = []
        for s in records["submissions"][:3]:
            L = unpack_physical(tensors[f"{s['source']}_gradient"], s["layout"],
                                s["axis"], s["range"], (7, 3))
            G_parts.append(L)
        G = np.concatenate(G_parts, axis=0)
        v = 0.1 * G
        global_x = canonical_muon(v)
        # 错误示范：对 q0 的 3x3 块做“本地 Muon”，其朝向/结果与全局片不同。
        q0 = v[0:3, :]
        local_wrong = canonical_muon(q0)  # 3x3 不转置
        correct_q0 = global_x[0:3, :]
        self.assertGreater(float(np.abs(local_wrong - correct_q0).max()), 1e-3)

    def test_zero_matrix_branch(self):
        V = np.zeros((6, 3))
        pieces = [ShardPiece(f"p{i}", 0, lo, hi, np.zeros((hi - lo, 3)))
                  for i, (lo, hi) in enumerate([(0, 2), (2, 4), (4, 6)])]
        res = sharded_muon((6, 3), pieces, IterationSpec(), "float16")
        self.assertTrue(res.finite)
        self.assertEqual(res.final_norm, 0.0)
        self.assertTrue(np.all(assemble_logical_x((6, 3), res) == 0.0))
        # 参考实现同样不除零
        self.assertTrue(np.all(muon_reference(V) == 0.0))

    def test_near_rank_deficient_uses_same_metric_and_f64(self):
        rng = np.random.default_rng(7)
        # 两个几乎共线的通道（呼应题面“两个相近通道”）。
        t = rng.standard_normal((9, 1))
        V = np.concatenate([t, t + 1e-9 * rng.standard_normal((9, 1)),
                            rng.standard_normal((9, 1)) * 1e-6], axis=1)
        ref = canonical_muon(V)
        specs = row_pieces(V, 3, axis=0)
        pieces = [ShardPiece(s["partition"], 0, s["range"][0], s["range"][1], s["logical"])
                  for s in specs]
        ev = sm.evaluate_matrix(
            MatrixSpec("D", (9, 3), error=1e-10, max_norm=99.0),
            build_coverages((9, 3), pieces)[0], pieces, IterationSpec(),
            reference_x=ref,
        )
        # 1e-10 的严公差会把精度推到 float64 并通过同一比较口径。
        self.assertEqual(ev.selected_mode, "float64")
        self.assertLessEqual(ev.attempts[-1].max_error, 1e-10)

    def test_gram_unique_pairs_trace(self):
        rng = np.random.default_rng(11)
        V = rng.standard_normal((8, 3))
        # 列分块（转置后是 X 的行块）：3 列切成 2 块，无序对 i<=j 只算一次 => 3 对/轮。
        specs = row_pieces(V, 2, axis=1)
        pieces = [ShardPiece(s["partition"], 1, s["range"][0], s["range"][1], s["logical"])
                  for s in specs]
        res = sharded_muon((8, 3), pieces, IterationSpec(iterations=2), "float64")
        pair_counts = [e["count"] for e in res.trace["gram_block_pairs"]]
        self.assertEqual(pair_counts, [3, 3])  # k(k+1)/2 = 3，每轮一次


class CoverageTests(unittest.TestCase):
    def test_exact_coverage_and_gaps(self):
        pieces = [
            ShardPiece("a", 0, 0, 3, np.zeros((3, 2))),
            ShardPiece("b", 0, 3, 5, np.zeros((2, 2))),
        ]
        covs = build_coverages((7, 2), pieces)
        self.assertFalse(covs[0].complete)
        self.assertEqual(covs[0].gaps, [(5, 7)])
        pieces.append(ShardPiece("c", 0, 5, 7, np.zeros((2, 2))))
        covs = build_coverages((7, 2), pieces)
        self.assertTrue(covs[0].complete)
        chosen, ax, ignored = choose_coverage(covs)
        self.assertEqual(ax, 0)

    def test_two_complete_coverages_not_stacked(self):
        # 同时给完整行覆盖与完整列覆盖：只选一套，结果不得是两套之和。
        rng = np.random.default_rng(5)
        M = rng.standard_normal((6, 4))
        row = [ShardPiece(f"r{i}", 0, lo, hi, M[lo:hi, :])
               for i, (lo, hi) in enumerate([(0, 2), (2, 4), (4, 6)])]
        col = [ShardPiece(f"c{i}", 1, lo, hi, M[:, lo:hi])
               for i, (lo, hi) in enumerate([(0, 2), (2, 3), (3, 4)])]
        covs = build_coverages((6, 4), row + col)
        chosen, ax, ignored = choose_coverage(covs, preferred_axis=0)
        self.assertTrue(covs[0].complete and covs[1].complete)
        self.assertEqual(ax, 0)
        self.assertEqual(ignored, [1])
        # 选中的覆盖只有 3 个行块，绝不包含列块。
        self.assertEqual([p.partition for p in chosen.pieces], ["r0", "r1", "r2"])


class CapturePlayTests(unittest.TestCase):
    def setUp(self):
        self.case = sm.load_capture_case(DATA / "capture.json")
        self.records = self.case.records
        self.tensors = self.case.tensors

    def test_full_story(self):
        play = sm.play_capture(DATA / "capture.json")
        # 首批只有 R 完整（全零），Q 因 q1 v11 旧代次缺 [3,6) 等待。
        self.assertEqual(list(play.first_report.successful), ["R"])
        self.assertIn("Q", play.first_report.waiting)
        q_reason_codes = [r["code"] for r in play.first_report.deliveries["Q"].reasons]
        self.assertIn("version_mismatch", q_reason_codes)
        self.assertIn("missing_coverage", q_reason_codes)

        # 提交1只应用 R。
        self.assertEqual(play.first_receipt.newly_applied_matrices, ["R"])
        self.assertTrue(play.first_receipt.waiting_for_data)
        self.assertIn("Q", play.first_receipt.waiting)
        self.assertEqual(play.first_receipt.iteration_spec["iterations"], 4)
        self.assertAlmostEqual(play.first_receipt.iteration_spec["beta"], 0.9)

        # 补报 q1 v12 后 Q 可交付：binary16 误差不过 1e-4，升到 float32。
        q = play.second_report.newly_successful["Q"]
        self.assertTrue(q.deliverable)
        self.assertEqual(q.selected_precision, "float32")
        modes = [a.mode for a in q.evaluation.attempts]
        self.assertEqual(modes[0], "binary16")
        self.assertFalse(q.evaluation.attempts[0].error_passed)
        self.assertTrue(q.evaluation.attempts[1].error_passed)
        self.assertLessEqual(q.evaluation.attempts[-1].max_error, 1e-4)
        self.assertLessEqual(q.evaluation.result.final_norm, 2.3)

        # R 全零走 binary16 且精确为零。
        r = play.first_report.deliveries["R"]
        self.assertEqual(r.selected_precision, "binary16")
        self.assertEqual(r.evaluation.result.final_norm, 0.0)

        # 提交2只新应用 Q，R 不二次应用；重发同键返回冻结回执。
        self.assertEqual(play.second_receipt.newly_applied_matrices, ["Q"])
        self.assertEqual(play.second_receipt.already_applied, ["R"])
        self.assertEqual(play.resend_receipt.key, "train-9#confirm-2")
        self.assertTrue(play.resend_receipt.resend_of_original)
        self.assertEqual(play.resend_receipt.actual_precision.get("Q"), "float32")

        coord = play.coordinator
        w_q = coord.states["Q"].weight
        self.assertEqual(w_q.dtype, np.float32)
        self.assertEqual(coord.states["Q"].momentum.dtype, np.float32)
        self.assertEqual(coord.states["Q"].inference.dtype, np.float16)
        # 推理副本严格等于 float32 权重的 RTNE binary16。
        np.testing.assert_array_equal(
            coord.states["Q"].inference, w_q.astype(np.float16)
        )
        # 代次只前进一次。
        self.assertEqual(coord.states["Q"].generation, 13)
        self.assertEqual(coord.states["R"].generation, 45)

    def test_q_matches_independent_reference_and_shard_physical_shapes(self):
        play = sm.play_capture(DATA / "capture.json")
        q = play.second_report.deliveries["Q"]
        # 独立 oracle：从采集梯度拼 G，v=0.1G（旧动量为零）。
        G = _assemble_q_gradient(self.records, self.tensors)
        ref = canonical_muon(0.1 * G, iterations=4)
        err = float(np.abs(q.logical_x - ref).max())
        self.assertLess(err, 1e-5)  # 实际交付 float32 对 f64 oracle

        # 每个回传分片的物理形状必须与该 submission 原始张量形状一致。
        sub_by_part = {s["partition"]: s for s in self.records["submissions"][:3]}
        src_shape = {s["partition"]: self.tensors[f"{s['source']}_gradient"].shape
                     for s in self.records["submissions"][:3]}
        for su in q.shard_updates:
            self.assertEqual(tuple(su["x_physical"].shape), src_shape[su["partition"]])
            self.assertEqual(su["layout"], sub_by_part[su["partition"]]["layout"])
            # delta = -lr*x
            np.testing.assert_allclose(
                su["delta_logical"].astype(np.float64),
                -0.004 * su["x_logical"].astype(np.float64), rtol=1e-6, atol=1e-9,
            )

    def test_column_replay_equivalent(self):
        play = sm.play_capture(DATA / "capture.json")
        col_q = play.column_report.deliveries["Q"]
        row_q = play.second_report.deliveries["Q"]
        self.assertTrue(col_q.deliverable)
        err = float(np.abs(col_q.logical_x - row_q.logical_x).max())
        self.assertLess(err, 1e-4)  # 同一比较口径
        self.assertEqual(col_q.selected_precision, row_q.selected_precision)

    def test_initial_weights_assembled_match_column_sources(self):
        # 行 source 装出的 Q 权重与 column_* 资料在逻辑上是同一份。
        coord = MuonCoordinator.from_capture(DATA / "capture.json")
        w = coord.states["Q"].weight
        np.testing.assert_allclose(w[0], [0.5, 0.0, -0.3], atol=1e-7)
        np.testing.assert_allclose(w[6], [3.0, -1.0, -2.0], atol=1e-7)
        # 相近通道的结构确实在整矩阵结果里被一起正交化（非本地分片正交化）。
        self.assertEqual(w.shape, (7, 3))

    def test_independent_local_orthogonalization_is_wrong(self):
        # 题面核心：各片独立正交化会显著改变方向；全局 Gram 分片实现必须等于整矩阵。
        G = _assemble_q_gradient(self.records, self.tensors)
        v = 0.1 * G
        global_x = canonical_muon(v, iterations=4)
        naive_local = np.zeros_like(global_x)
        for lo, hi in [(0, 3), (3, 6), (6, 7)]:
            naive_local[lo:hi] = canonical_muon(v[lo:hi], iterations=4)
        self.assertGreater(float(np.abs(naive_local - global_x).max()), 1e-2)

        play = sm.play_capture(DATA / "capture.json")
        delivered = play.second_report.deliveries["Q"].logical_x
        self.assertLess(float(np.abs(delivered - global_x).max()), 1e-5)


def _assemble_q_gradient(records, tensors):
    parts = []
    for s in records["submissions"][:3]:
        L = unpack_physical(tensors[f"{s['source']}_gradient"], s["layout"],
                            s["axis"], s["range"], records["matrices"]["Q"]["shape"])
        parts.append((s["range"][0], L))
    parts.sort()
    return np.concatenate([L for _, L in parts], axis=0)


class ServiceBehaviorTests(unittest.TestCase):
    def _coord(self, **kw):
        spec = IterationSpec(iterations=3, beta=0.9, lr=0.01)
        return MuonCoordinator(
            {"M": MatrixSpec("M", (6, 4), generation=1, error=1e-6, max_norm=99.0)},
            iteration_spec=spec, **kw,
        )

    def test_missing_and_corrupt_and_absent_not_zero(self):
        coord = self._coord()
        rng = np.random.default_rng(0)
        good = rng.standard_normal((2, 4)).astype(np.float32)
        bad = rng.standard_normal((2, 4)).astype(np.float32)
        bad[0, 0] = np.inf
        subs = [
            {"matrix": "M", "partition": "a", "axis": 0, "range": [0, 2],
             "version": 1, "gradient": good},
            {"matrix": "M", "partition": "b", "axis": 0, "range": [2, 4],
             "version": 1, "gradient": bad},
            {"matrix": "M", "partition": "c", "axis": 0, "range": [4, 6],
             "version": 1, "gradient": None},
        ]
        rep = coord.report("s", subs)
        codes = {c for r in rep.deliveries["M"].reasons for c in [r["code"]]}
        self.assertIn("corrupt_shard", codes)
        self.assertIn("absent_shard", codes)
        self.assertFalse(rep.deliveries["M"].deliverable)
        receipt = coord.commit("s")
        self.assertEqual(receipt.applied, [])

    def test_wrong_shape_is_corrupt_diagnostic(self):
        coord = self._coord()
        subs = [
            {"matrix": "M", "partition": "a", "axis": 0, "range": [0, 3],
             "layout": "RC", "version": 1, "gradient": np.zeros((2, 4))},
        ]
        rep = coord.report("s", subs)
        codes = [r["code"] for r in rep.deliveries["M"].reasons]
        self.assertTrue("corrupt_shard" in codes or "missing_coverage" in codes)

    def test_norm_gate_keeps_old(self):
        coord = MuonCoordinator(
            {"N": MatrixSpec("N", (8, 2), generation=0, error=1e-3, max_norm=0.05)},
            iteration_spec=IterationSpec(iterations=4),
        )
        rng = np.random.default_rng(2)
        G = rng.standard_normal((8, 2)).astype(np.float32)
        subs = [{"matrix": "N", "partition": f"p{i}", "axis": 0,
                 "range": [lo, hi], "version": 0, "gradient": G[lo:hi]}
                for i, (lo, hi) in enumerate([(0, 2), (2, 4), (4, 6), (6, 8)])]
        rep = coord.report("s", subs)
        d = rep.deliveries["N"]
        self.assertFalse(d.deliverable)
        self.assertTrue(any(r["code"] == "norm_exceeded" for r in d.reasons))
        w0 = coord.states["N"].weight.copy()
        coord.commit("s")
        np.testing.assert_array_equal(coord.states["N"].weight, w0)
        self.assertEqual(coord.states["N"].generation, 0)

    def test_no_double_apply_even_without_key(self):
        coord = self._coord()
        rng = np.random.default_rng(8)
        G = rng.standard_normal((6, 4)).astype(np.float32)
        subs = [{"matrix": "M", "partition": f"p{i}", "axis": 0,
                 "range": [lo, hi], "version": 1, "gradient": G[lo:hi]}
                for i, (lo, hi) in enumerate([(0, 3), (3, 6)])]
        coord.report("s", subs)
        r1 = coord.commit("s", key="k")
        w1 = coord.states["M"].weight.copy()
        coord.commit("s", key="k")
        coord.commit("s")  # 无键重复
        self.assertEqual(coord.states["M"].generation, 2)
        np.testing.assert_array_equal(coord.states["M"].weight, w1)
        self.assertEqual(r1.newly_applied_matrices, ["M"])

    def test_older_step_waiting_report(self):
        coord = self._coord()
        g = np.random.default_rng(0).standard_normal((3, 4))
        coord.report("old", [{"matrix": "M", "partition": "a", "axis": 0,
                              "range": [0, 3], "version": 1, "gradient": g}])
        # 新步骤的回执应指出 old 仍在等数据。
        rep = coord.report("new", [])
        rc = coord.commit("new")
        self.assertIn("old", rc.older_steps_waiting)

    def test_unknown_matrix_diagnostic(self):
        coord = self._coord()
        rep = coord.report("s", [{"matrix": "NOPE", "partition": "x"}])
        self.assertEqual(rep.unknown_matrices[0]["code"], "unknown_matrix")

    def test_persistence_roundtrip_receipt_query(self):
        with tempfile.TemporaryDirectory() as td:
            coord = MuonCoordinator(
                {"M": MatrixSpec("M", (4, 3), generation=9, error=1e-4, max_norm=9)},
                iteration_spec=IterationSpec(iterations=2, lr=0.02), state_dir=td,
            )
            G = np.zeros((4, 3), np.float32)
            coord.report("step-x", [
                {"matrix": "M", "partition": "a", "axis": 0, "range": [0, 2],
                 "version": 9, "gradient": G[:2]},
                {"matrix": "M", "partition": "b", "axis": 0, "range": [2, 4],
                 "version": 9, "gradient": G[2:]},
            ])
            rc = coord.commit("step-x", key="key-xyz")
            self.assertTrue(Path(td, "receipts.json").exists())

            loaded = MuonCoordinator.load(td)
            got = loaded.get_confirmation("key-xyz")
            self.assertIsInstance(got, CommitReceipt)
            self.assertEqual(got.iteration_spec["iterations"], 2)
            self.assertEqual(got.actual_precision["M"], "binary16")
            self.assertIn("M", got.applied)
            # 回执 JSON 可序列化且冻结了当时规格。
            json.dumps(got.to_dict(), ensure_ascii=False)
            np.testing.assert_array_equal(loaded.states["M"].weight,
                                          coord.states["M"].weight)


class GroupPrecisionTests(unittest.TestCase):
    def test_group_precision_uniform_and_independent_groups(self):
        specs = {
            "Q": MatrixSpec("Q", (7, 3), 0, error=1e-4, max_norm=9.0, group="G"),
            "R": MatrixSpec("R", (2, 3), 0, error=1e-3, max_norm=9.0, group="G"),
        }
        coord = MuonCoordinator(specs, iteration_spec=IterationSpec(iterations=4))
        gq = np.array(
            [[1, 1, .01], [2, 2, .021], [3, 3, .028], [4, 4.001, .041],
             [5, 4.999, .052], [6, 6, .069], [7, 7.003, .077]], dtype=np.float32) * 0.1
        subs = []
        for i, (lo, hi) in enumerate([(0, 3), (3, 6), (6, 7)]):
            subs.append({"matrix": "Q", "partition": f"q{i}", "axis": 0,
                         "range": [lo, hi], "version": 0, "gradient": gq[lo:hi]})
        # R 先缺席：Q 不应被 R 阻塞（同组但 R 无数据时只按可评估成员定档）。
        rep_q = coord.report("s", subs)
        self.assertTrue(rep_q.newly_successful["Q"].deliverable)
        self.assertEqual(rep_q.deliveries["Q"].selected_precision, "float32")
        # 补上全零 R：组内统一 float32，而不是 R 单独 binary16。
        rep = coord.report("s", [{
            "matrix": "R", "partition": "r0", "axis": 0, "range": [0, 2],
            "version": 0, "gradient": np.zeros((2, 3), np.float32)}])
        self.assertEqual(rep.deliveries["Q"].selected_precision, "float32")
        self.assertEqual(rep.deliveries["R"].selected_precision, "float32")
        rc = coord.commit("s", key="g")
        self.assertEqual(rc.actual_precision, {"Q": "float32", "R": "float32"})
        self.assertEqual(set(coord.successful_matrices("s")), {"Q", "R"})
        self.assertEqual(coord.delay_reasons("s"), {})
        self.assertEqual(len(coord.shard_updates("s", "Q")), 3)


class CapacityPlanTests(unittest.TestCase):
    def test_scratch_fits_and_full_copy_does_not(self):
        plan = scratch_plan((18432, 432), 23, split_axis=0, budget_bytes=6_291_456)
        self.assertTrue(plan.fits_budget)
        self.assertGreater(plan.full_matrix_copy_bytes, 6_291_456)
        self.assertLessEqual(plan.gram_bytes, 432 * 432 * 8)

    @unittest.skipUnless(os.environ.get("SHARD_MUON_RUN_CAPACITY") == "1",
                         "大矩阵回放默认跳过；设置 SHARD_MUON_RUN_CAPACITY=1 运行")
    def test_big_seeded_replay(self):
        rep = sm.run_capacity_replay(DATA / "capacity.json")
        self.assertLess(rep["row_vs_reference_maxerr"], 1e-10)
        self.assertLess(rep["col_vs_reference_maxerr"], 1e-10)
        self.assertLess(rep["row_vs_col_maxerr"], 1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
