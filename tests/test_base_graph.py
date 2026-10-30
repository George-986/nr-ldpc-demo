"""lifting size 表和基图结构测试。"""

from __future__ import annotations

import unittest

import numpy as np

from nr_pusch_ldpc_rx.ldpc.base_graph import (
    base_graph,
    base_graph_for_z,
    base_graph_syndrome,
    hard_syndrome_ok,
    lifting_set_index,
    shift,
)
from nr_pusch_ldpc_rx.constants import (
    BG1_COLS,
    BG1_KB,
    BG1_ROWS,
    BG2_COLS,
    BG2_ROWS,
    Z_LIST,
    Z_SETS,
    Z_TO_ILS,
)


class TestLiftingTable(unittest.TestCase):
    def test_z_table_complete_and_unique(self):
        """表 5.3.2-1 共 51 个 Z，且每个 Z 只属于一个 i_LS。"""
        self.assertEqual(len(Z_LIST), 51)
        self.assertEqual(set(Z_LIST), set(Z_TO_ILS))
        self.assertEqual(len(Z_SETS), 8)
        for i_ls, group in enumerate(Z_SETS):
            for z in group:
                self.assertEqual(Z_TO_ILS[z], i_ls)
                self.assertEqual(lifting_set_index(z), i_ls)
        self.assertEqual(Z_LIST, tuple(sorted(Z_LIST)))

    def test_base_graph_dimensions_and_entries(self):
        """BG1 为 46x68，BG2 为 42x52，已知表项与 38.212 一致。"""
        bg1 = base_graph(1, 0)
        bg2 = base_graph(2, 0)
        self.assertEqual(bg1.shape, (BG1_ROWS, BG1_COLS))
        self.assertEqual(bg2.shape, (BG2_ROWS, BG2_COLS))

        # i_LS=0 的几个表项（直接来自 Table 5.3.2-2/3）。
        self.assertEqual(int(bg1[0, 0]), 250)
        self.assertEqual(int(bg1[0, 1]), 69)
        self.assertEqual(int(bg1[0, BG1_KB]), 1)   # P_A
        self.assertEqual(int(bg1[0, 23]), 0)
        self.assertEqual(int(bg2[0, 0]), 9)
        self.assertEqual(int(bg2[0, 10]), 0)       # P_A
        self.assertEqual(int(bg2[2, 10]), 1)       # P_B
        self.assertEqual(int(bg2[0, 12]), -1)      # 表项不存在 -> 零块

        for bgn in (1, 2):
            for i_ls in range(8):
                bg = base_graph(bgn, i_ls)
                self.assertTrue(np.all(bg >= -1))
                self.assertTrue(np.all(bg < 384))
                # 返回的缓存数组应只读，防止调用方修改协议表。
                with self.assertRaises(ValueError):
                    bg[0, 0] = 7

    def test_shift_direction(self):
        """P^s * x 必须满足 H[i,(i+s) mod Z]=1 的约定。"""
        x = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        # 与 np.roll(x, -s) 等价；这正是 H 行 i 取 x[(i+s) mod Z] 的含义。
        np.testing.assert_array_equal(shift(x, 0), x)
        np.testing.assert_array_equal(shift(x, 1), [2, 3, 4, 5, 1])
        np.testing.assert_array_equal(shift(x, 2), [3, 4, 5, 1, 2])
        np.testing.assert_array_equal(shift(shift(x, 3), -3), x)

    def test_base_graph_for_z_matches_set(self):
        """base_graph_for_z 必须按 Z 正确选中 i_LS。"""
        for z in (2, 3, 5, 7, 9, 11, 13, 15, 384):
            i_ls = Z_TO_ILS[z]
            np.testing.assert_array_equal(
                base_graph_for_z(2, z), base_graph(2, i_ls)
            )

    def test_syndrome_helper(self):
        """全零码字满足 H*c^T=0；单比特错误通常不满足。"""
        z = 12
        blocks = np.zeros((BG2_COLS, z), dtype=np.uint8)
        self.assertTrue(not np.any(base_graph_syndrome(blocks, 2, z)))
        self.assertTrue(hard_syndrome_ok(blocks.reshape(-1), 2, z))

        blocks[0, 0] = 1
        self.assertTrue(np.any(base_graph_syndrome(blocks, 2, z)))
        self.assertFalse(hard_syndrome_ok(blocks.reshape(-1), 2, z))


if __name__ == "__main__":
    unittest.main()
