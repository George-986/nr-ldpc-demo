"""BG 选择、Kb 阈值和调制阶数测试。"""

from __future__ import annotations

import unittest

from nr_pusch_ldpc_rx.constants import (
    kb_for_bg2,
    modulation_order,
    select_base_graph,
    tb_crc_kind,
)


class TestBaseGraphSelection(unittest.TestCase):
    def test_rules(self):
        # A <= 292 总是 BG2。
        self.assertEqual(select_base_graph(1, 0.90), 2)
        self.assertEqual(select_base_graph(292, 0.90), 2)
        # 292 < A <= 3824 且 R <= 0.67 -> BG2。
        self.assertEqual(select_base_graph(293, 0.67), 2)
        self.assertEqual(select_base_graph(3824, 0.67), 2)
        # A 超过 3824 后，只有 R <= 0.25 才选 BG2。
        self.assertEqual(select_base_graph(3825, 0.67), 1)
        self.assertEqual(select_base_graph(5000, 0.25), 2)
        self.assertEqual(select_base_graph(5000, 0.251), 1)
        self.assertEqual(select_base_graph(5000, 0.80), 1)
        # 293 在 R>0.25 时不满足前两条，BG1。
        self.assertEqual(select_base_graph(293, 0.90), 1)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            select_base_graph(0, 0.5)
        with self.assertRaises(ValueError):
            select_base_graph(100, 0.0)
        with self.assertRaises(ValueError):
            select_base_graph(100, 1.0)


class TestKb(unittest.TestCase):
    def test_boundaries(self):
        """Kb 阈值按协议使用严格大于。"""
        self.assertEqual(kb_for_bg2(192), 6)
        self.assertEqual(kb_for_bg2(193), 8)
        self.assertEqual(kb_for_bg2(560), 8)
        self.assertEqual(kb_for_bg2(561), 9)
        self.assertEqual(kb_for_bg2(640), 9)
        self.assertEqual(kb_for_bg2(641), 10)
        self.assertEqual(kb_for_bg2(1), 6)
        self.assertEqual(kb_for_bg2(10000), 10)

    def test_tb_crc(self):
        self.assertEqual(tb_crc_kind(1), "16")
        self.assertEqual(tb_crc_kind(3824), "16")
        self.assertEqual(tb_crc_kind(3825), "24A")


class TestModulation(unittest.TestCase):
    def test_names(self):
        for name, qm in (
            ("BPSK", 1),
            ("pi/2-BPSK", 1),
            ("QPSK", 2),
            ("16QAM", 4),
            ("64QAM", 6),
            ("256QAM", 8),
            ("qpsk", 2),
            ("16qam", 4),
        ):
            with self.subTest(name=name):
                self.assertEqual(modulation_order(name), qm)

    def test_integer(self):
        for qm in (1, 2, 4, 6, 8):
            self.assertEqual(modulation_order(qm), qm)

    def test_invalid(self):
        for bad in (0, 3, 5, 7, 9):
            with self.assertRaises(ValueError):
                modulation_order(bad)
        with self.assertRaises(ValueError):
            modulation_order("1024QAM")
        with self.assertRaises(TypeError):
            modulation_order(True)


if __name__ == "__main__":
    unittest.main()
