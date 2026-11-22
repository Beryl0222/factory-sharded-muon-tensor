"""核对合成采集的交换记录与原始尺寸。"""
import json
from pathlib import Path
import unittest

import numpy as np

from caseio import load_case


class CaptureTests(unittest.TestCase):
    def setUp(self):
        data = Path(__file__).resolve().parents[1] / "data"
        self.records, self.tensors = load_case(data / "capture.json")
        self.capacity = json.loads((data / "capacity.json").read_text(encoding="utf-8"))

    def test_capture_contract(self):
        r, a = self.records, self.tensors
        self.assertEqual(a["b_weight"].shape, (3, 3))
        self.assertEqual(a["c_gradient"].shape, (3, 1))
        self.assertNotEqual(r["submissions"][1]["version"], r["matrices"]["Q"]["generation"])
        self.assertFalse(np.any(a["d_gradient"]))
        self.assertIn("binary16", r["precision_modes"])
