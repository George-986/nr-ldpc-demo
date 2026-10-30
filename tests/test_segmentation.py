"""TS 38.212 5.2.2 分割、CB CRC、TB 拼接测试。"""

from __future__ import annotations

import unittest

import numpy as np

from nr_pusch_ldpc_rx.chain.crc import crc_attach
from nr_pusch_ldpc_rx.chain.segmentation import (
    check_transport_block_crc,
    desegment_code_blocks,
    make_code_blocks,
    plan_transport_block,
)


# (A, R, BG, TB_CRC, C, Kb, Zc, K, F, zero_padding, K')
PLAN_CASES = [
    (10, 0.30, 2, "16", 1, 6, 5, 50, 24, 0, 26),
    (100, 0.30, 2, "16", 1, 6, 20, 200, 84, 0, 116),
    # B=192 时 Kb=6；B=193 时跳到 Kb=8。
    (176, 0.30, 2, "16", 1, 6, 32, 320, 128, 0, 192),
    (177, 0.30, 2, "16", 1, 8, 26, 260, 67, 0, 193),
    (300, 0.30, 2, "16", 1, 8, 40, 400, 84, 0, 316),
    # B=560/561 的 Kb 8->9 边界。
    (544, 0.30, 2, "16", 1, 8, 72, 720, 160, 0, 560),
    (545, 0.30, 2, "16", 1, 9, 64, 640, 79, 0, 561),
    # B=640/641 的 Kb 9->10 边界。
    (624, 0.30, 2, "16", 1, 9, 72, 720, 80, 0, 640),
    (625, 0.30, 2, "16", 1, 10, 72, 720, 79, 0, 641),
    (1000, 0.30, 2, "16", 1, 10, 104, 1040, 24, 0, 1016),
    # BG1、单码块。
    (5000, 0.80, 1, "24A", 1, 22, 240, 5280, 256, 0, 5024),
    # BG1、三码块、最后一个 CB 需要 1 个补零比特。
    (20000, 0.70, 1, "24A", 3, 22, 320, 7040, 341, 1, 6699),
    # BG2、两码块。
    (5000, 0.20, 2, "24A", 2, 10, 256, 2560, 24, 0, 2536),
    # BG2、三码块、Zc=384。
    (11000, 0.20, 2, "24A", 3, 10, 384, 3840, 141, 1, 3699),
    # CRC16/CRC24A 阈值。
    (3824, 0.67, 2, "16", 1, 10, 384, 3840, 0, 0, 3840),
    (3825, 0.80, 1, "24A", 1, 22, 176, 3872, 23, 0, 3849),
]


class TestTransportBlockPlan(unittest.TestCase):
    def test_plan_table(self):
        """检查 BG/Kb/Zc/K/F/补零等关键参数。"""
        for (a, rate, bgn, crc, c, kb, z, k, f, pad, kp) in PLAN_CASES:
            with self.subTest(a=a, rate=rate):
                p = plan_transport_block(a, rate)
                self.assertEqual(p.bgn, bgn)
                self.assertEqual(p.tb_crc, crc)
                self.assertEqual(p.c, c)
                self.assertEqual(p.kb, kb)
                self.assertEqual(p.z, z)
                self.assertEqual(p.k, k)
                self.assertEqual(p.f, f)
                self.assertEqual(p.zero_padding, pad)
                self.assertEqual(p.k_prime, kp)
                self.assertEqual(p.n, (66 if bgn == 1 else 50) * z)
                self.assertEqual(p.n_full, (68 if bgn == 1 else 52) * z)
                self.assertEqual(p.data_bits_per_cb, kp - (24 if c > 1 else 0))
                # filler 必须落在打孔后 d 的合法范围内。
                self.assertGreaterEqual(p.filler_start_d, 0)
                self.assertLessEqual(p.filler_end_d, p.n)
                self.assertEqual(
                    int(p.filler_mask.sum()), p.f
                )

    def test_forced_base_graph(self):
        """bgn 显式覆盖时，分段公式仍应成立。"""
        p = plan_transport_block(1000, 0.30, bgn=1)
        self.assertEqual(p.bgn, 1)
        self.assertEqual(p.kb, 22)
        self.assertEqual(p.c, 1)
        self.assertEqual(p.k, 22 * p.z)

    def test_invalid_plan_inputs(self):
        """非法 A/R/BG 应报错。"""
        with self.assertRaises(ValueError):
            plan_transport_block(0, 0.5)
        with self.assertRaises(ValueError):
            plan_transport_block(100, 0.0)
        with self.assertRaises(ValueError):
            plan_transport_block(100, 1.0)
        with self.assertRaises(ValueError):
            plan_transport_block(100, 0.5, bgn=3)


class TestSegmentationRoundTrip(unittest.TestCase):
    def test_make_and_desegment(self):
        """对所有计划做 TB -> CB -> TB 无损往返。"""
        rng = np.random.default_rng(2024)
        for (a, rate, *_rest) in PLAN_CASES:
            with self.subTest(a=a, rate=rate):
                p = plan_transport_block(a, rate)
                tb = (rng.random(a) < 0.47).astype(np.uint8)
                cbs = make_code_blocks(tb, p)
                self.assertEqual(cbs.shape, (p.c, p.k))
                # 每个 CB 的 filler 区必须全 0。
                for r in range(p.c):
                    self.assertFalse(np.any(cbs[r, p.k_prime:]))

                b_hat, checks = desegment_code_blocks(cbs, p)
                np.testing.assert_array_equal(b_hat, crc_attach(tb, p.tb_crc))
                self.assertTrue(all(not c.crc_present or c.crc_ok for c in checks))
                tb_hat, ok = check_transport_block_crc(b_hat, p)
                self.assertTrue(ok)
                np.testing.assert_array_equal(tb_hat, tb)

    def test_cb_crc_detects_error(self):
        """多码块时，任意一个 CB 的数据错误都应被 CB CRC 检出。"""
        p = plan_transport_block(20000, 0.70)
        self.assertGreater(p.c, 1)
        rng = np.random.default_rng(5)
        tb = rng.integers(0, 2, p.a, dtype=np.uint8)
        cbs = make_code_blocks(tb, p)
        bad = cbs.copy()
        bad[1, 0] ^= 1
        b_hat, checks = desegment_code_blocks(bad, p)
        self.assertTrue(checks[0].crc_ok)
        self.assertFalse(checks[1].crc_ok)
        # b_hat 仍然返回，便于上层统计；但 TB CRC 应失败。
        self.assertFalse(check_transport_block_crc(b_hat, p)[1])

    def test_single_cb_has_no_cb_crc(self):
        """C=1 时不应附加 CRC24B，cb_crc_len=0。"""
        p = plan_transport_block(1000, 0.30)
        self.assertEqual(p.c, 1)
        self.assertEqual(p.cb_crc_len, 0)
        tb = np.ones(p.a, dtype=np.uint8)
        cbs = make_code_blocks(tb, p)
        self.assertEqual(cbs.shape, (1, p.k))
        _, checks = desegment_code_blocks(cbs, p)
        self.assertFalse(checks[0].crc_present)
        self.assertTrue(checks[0].crc_ok)

    def test_make_code_blocks_rejects_bad_tb(self):
        """TB 长度或取值非法时应报错。"""
        p = plan_transport_block(100, 0.3)
        with self.assertRaises(ValueError):
            make_code_blocks(np.zeros(p.a - 1, dtype=np.uint8), p)
        with self.assertRaises(ValueError):
            make_code_blocks(np.full(p.a, 2, dtype=np.uint8), p)


if __name__ == "__main__":
    unittest.main()
