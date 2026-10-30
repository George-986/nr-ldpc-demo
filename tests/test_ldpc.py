"""LDPC 编码/分层 NMS 译码测试。"""

from __future__ import annotations

import math
import unittest

import numpy as np

from nr_pusch_ldpc_rx.ldpc.base_graph import base_graph_syndrome, hard_syndrome_ok
from nr_pusch_ldpc_rx.constants import Z_LIST
from nr_pusch_ldpc_rx.ldpc import decode, encode


def _awgn_llr(codeword: np.ndarray, ebn0_db: float, rate: float, rng) -> np.ndarray:
    """把 BPSK 码字经过 AWGN，返回 LLR（0->+1, 1->-1）。"""
    symbols = 1.0 - 2.0 * codeword.astype(np.float64)
    # Es=1，N0=2*sigma^2，Eb/N0 = Es/(rate*N0)
    sigma = math.sqrt(1.0 / (2.0 * rate * 10.0 ** (ebn0_db / 10.0)))
    received = symbols + rng.normal(0.0, sigma, symbols.size)
    return 2.0 * received / (sigma * sigma)


class TestLDPCEncode(unittest.TestCase):
    def test_all_lifting_sizes_syndrome(self):
        """遍历 51 个 Zc 和两个 BG，编码结果必须满足 H*c^T=0。"""
        rng = np.random.default_rng(2025)
        for bgn, kb in ((1, 22), (2, 10)):
            for z in Z_LIST:
                with self.subTest(bgn=bgn, z=z):
                    info = rng.integers(0, 2, kb * z, dtype=np.uint8)
                    cw = encode(info, bgn, z)
                    self.assertEqual(cw.size, (68 if bgn == 1 else 52) * z)
                    self.assertTrue(hard_syndrome_ok(cw, bgn, z))
                    blocks = cw.reshape(-1, z)
                    syn = base_graph_syndrome(blocks, bgn, z)
                    self.assertFalse(np.any(syn))

    def test_filler_is_encoded_as_zero(self):
        """输入中的 -1 filler 应与 0 编码结果完全相同。"""
        rng = np.random.default_rng(11)
        z = 104
        info = rng.integers(0, 2, 10 * z, dtype=np.uint8)
        info[-24:] = 0
        cw_a = encode(info, 2, z)
        info_b = info.copy()
        info_b[-24:] = 255  # 非 0/1 会被拒绝，先验证错误路径
        with self.assertRaises(ValueError):
            encode(info_b, 2, z)
        info_c = info.copy()
        info_c[-24:] = 0
        np.testing.assert_array_equal(cw_a, encode(info_c, 2, z))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            encode(np.zeros(100, dtype=np.uint8), 2, 9)  # K 应为 10*9
        with self.assertRaises(ValueError):
            encode(np.zeros(221, dtype=np.uint8), 1, 10)  # K 应为 22*10
        with self.assertRaises(ValueError):
            encode(np.zeros(100, dtype=np.uint8), 3, 10)


class TestLDPCDecode(unittest.TestCase):
    def test_all_lifting_sizes_noiseless(self):
        """无噪声、仅有打孔未知比特时，所有 Zc 都应正确译码。"""
        rng = np.random.default_rng(77)
        for bgn, kb in ((1, 22), (2, 10)):
            for z in Z_LIST:
                with self.subTest(bgn=bgn, z=z):
                    info = rng.integers(0, 2, kb * z, dtype=np.uint8)
                    cw = encode(info, bgn, z)
                    llr = np.where(cw == 0, 4.0, -4.0)
                    llr[:2 * z] = 0.0
                    result = decode(llr, bgn, z, max_iter=20, alpha=0.8)
                    self.assertTrue(result.converged)
                    self.assertLessEqual(result.iterations, 20)
                    np.testing.assert_array_equal(result.info_bits, info)

    def test_filler_known_zero(self):
        """filler 位置注入大正 LLR 后，译码结果应忽略这些位置。"""
        rng = np.random.default_rng(19)
        z = 104
        info = rng.integers(0, 2, 10 * z, dtype=np.uint8)
        filler = 24
        info[-filler:] = 0
        cw = encode(info, 2, z)
        llr = np.where(cw == 0, 3.0, -3.0)
        llr[:2 * z] = 0.0
        # 完整码字中的 filler 位于 [K-F, K)；打孔后对应 d 的偏移在
        # 接收链路中由 plan.filler_mask 处理。这里直接给完整码字位置。
        llr[10 * z - filler:10 * z] = 1.0e4
        result = decode(llr, 2, z, max_iter=20)
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(result.info_bits[:-filler], info[:-filler])
        np.testing.assert_array_equal(result.info_bits[-filler:], np.zeros(filler, dtype=np.uint8))

    def test_awgn_soft_llr(self):
        """中等信噪比下，软 LLR 应能纠正信道错误。"""
        rng = np.random.default_rng(2026)
        for bgn, z in ((2, 104), (1, 240)):
            with self.subTest(bgn=bgn, z=z):
                kb = 22 if bgn == 1 else 10
                info = rng.integers(0, 2, kb * z, dtype=np.uint8)
                cw = encode(info, bgn, z)
                rate = (kb * z) / ((66 if bgn == 1 else 50) * z)
                llr = _awgn_llr(cw, ebn0_db=3.0, rate=rate, rng=rng)
                llr[:2 * z] = 0.0
                result = decode(llr, bgn, z, max_iter=50, alpha=0.8)
                self.assertTrue(result.converged)
                np.testing.assert_array_equal(result.info_bits, info)

    def test_no_early_stop_still_reports_convergence(self):
        """关闭早停时应跑满 max_iter，并在最后一轮正确报告 converged。"""
        rng = np.random.default_rng(31)
        z = 20
        info = rng.integers(0, 2, 10 * z, dtype=np.uint8)
        cw = encode(info, 2, z)
        llr = np.where(cw == 0, 4.0, -4.0)
        llr[:2 * z] = 0.0
        result = decode(llr, 2, z, max_iter=5, alpha=0.8, early_stop=False)
        self.assertEqual(result.iterations, 5)
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(result.info_bits, info)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            decode(np.zeros(10), 2, 10)
        with self.assertRaises(ValueError):
            decode(np.zeros(520), 2, 10, alpha=0.0)
        with self.assertRaises(ValueError):
            decode(np.zeros(520), 2, 10, max_iter=0)
        llr = np.zeros(520)
        llr[0] = np.inf
        with self.assertRaises(ValueError):
            decode(llr, 2, 10)


if __name__ == "__main__":
    unittest.main()
