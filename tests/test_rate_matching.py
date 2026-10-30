"""TS 38.212 5.4.2 速率匹配与比特交织测试。"""

from __future__ import annotations

import unittest

import numpy as np

from nr_pusch_ldpc_rx.chain.rate_matching import (
    FILLER_LLR,
    bit_deinterleave,
    bit_interleave,
    coded_bit_lengths,
    rate_match,
    rate_match_code_blocks,
    rate_recover,
    rate_recover_code_blocks,
    starting_position,
)
from nr_pusch_ldpc_rx.chain.segmentation import plan_transport_block


def _reference_select(
    d: np.ndarray, e_len: int, k0: int, filler_mask: np.ndarray
) -> np.ndarray:
    """按 38.212 5.4.2.1 伪代码逐比特选择（测试参考实现）。"""
    out: list[int] = []
    j = 0
    while len(out) < e_len:
        pos = (k0 + j) % len(d)
        if not filler_mask[pos]:
            out.append(int(d[pos]))
        j += 1
    return np.asarray(out, dtype=np.uint8)


def _reference_interleave(e: np.ndarray, qm: int) -> np.ndarray:
    """按 5.4.2.2 双重循环直接构造 f。"""
    length = e.size // qm
    f = np.empty_like(e)
    for j in range(length):
        for i in range(qm):
            f[i + qm * j] = e[j + length * i]
    return f


class TestBitInterleaving(unittest.TestCase):
    def test_known_patterns(self):
        e = np.arange(8, dtype=np.uint8)
        np.testing.assert_array_equal(
            bit_interleave(e, 2), [0, 4, 1, 5, 2, 6, 3, 7]
        )
        np.testing.assert_array_equal(
            bit_interleave(e, 4), [0, 2, 4, 6, 1, 3, 5, 7]
        )

    def test_roundtrip_all_qm(self):
        rng = np.random.default_rng(31)
        for qm in (1, 2, 4, 6, 8):
            with self.subTest(qm=qm):
                e = rng.integers(0, 2, 48 * qm, dtype=np.uint8)
                f = bit_interleave(e, qm)
                np.testing.assert_array_equal(bit_deinterleave(f, qm), e)
                np.testing.assert_array_equal(_reference_interleave(e, qm), f)

    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            bit_interleave([0, 1, 0], 2)
        with self.assertRaises(ValueError):
            bit_deinterleave([0.0, 1.0, 2.0], 2)


class TestCodedBitLengths(unittest.TestCase):
    def test_distribution(self):
        np.testing.assert_array_equal(coded_bit_lengths(100, 1, 2), [100])
        np.testing.assert_array_equal(coded_bit_lengths(100, 3, 2), [32, 34, 34])
        np.testing.assert_array_equal(coded_bit_lengths(102, 2, 2), [50, 52])
        np.testing.assert_array_equal(coded_bit_lengths(100, 3, 1), [33, 33, 34])
        np.testing.assert_array_equal(
            coded_bit_lengths(200, 3, 2, n_layers=2), [64, 68, 68]
        )

    def test_invalid(self):
        with self.assertRaises(ValueError):
            coded_bit_lengths(0, 1, 2)
        with self.assertRaises(ValueError):
            coded_bit_lengths(101, 1, 2)
        with self.assertRaises(ValueError):
            coded_bit_lengths(100, 0, 2)


class TestStartingPosition(unittest.TestCase):
    def test_bg1(self):
        z = 10
        self.assertEqual([starting_position(1, z, rv) for rv in range(4)],
                         [0, 170, 330, 560])

    def test_bg2(self):
        z = 10
        self.assertEqual([starting_position(2, z, rv) for rv in range(4)],
                         [0, 130, 250, 430])

    def test_limited_buffer(self):
        z = 10
        n_cb = 250
        self.assertEqual(starting_position(2, z, 1, n_cb), 60)
        self.assertEqual(starting_position(2, z, 2, n_cb), 120)
        self.assertEqual(starting_position(2, z, 3, n_cb), 210)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            starting_position(2, 10, 4)
        with self.assertRaises(ValueError):
            starting_position(2, 10, 1, n_cb=501)
        with self.assertRaises(ValueError):
            starting_position(2, 10, 1, n_cb=0)


class TestRateMatchingInverse(unittest.TestCase):
    def setUp(self):
        self.plan = plan_transport_block(1000, 0.30)  # BG2, Kb=10, Z=104
        self.rng = np.random.default_rng(101)

    def _check_case(self, rv: int, qm: int, e_len: int, n_cb: int | None = None):
        plan = self.plan
        d = self.rng.integers(0, 2, plan.n, dtype=np.uint8)
        d[plan.filler_mask] = 0
        k0 = starting_position(plan.bgn, plan.z, rv, n_cb)
        buffer_len = n_cb or plan.n
        buffer_mask = plan.filler_mask[:buffer_len]
        ref = _reference_select(d[:buffer_len], e_len, k0, buffer_mask)
        ref_f = _reference_interleave(ref, qm)
        got_f = rate_match(
            d, e_len, rv, qm, plan.bgn, plan.z, plan.filler_mask, n_cb
        )
        np.testing.assert_array_equal(got_f, ref_f)

        # 用“幅度固定”的 LLR 验证解交织 + 循环缓存合并。
        amplitude = 3.0
        llr = np.where(ref_f == 0, amplitude, -amplitude)
        recovered = rate_recover(
            llr, rv, qm, plan.bgn, plan.z, plan.filler_mask, n_cb
        )
        n = plan.n
        # 复现每个 d 位置被选取的次数。
        idx = []
        j = 0
        while len(idx) < e_len:
            pos = (k0 + j) % (n_cb or n)
            if not plan.filler_mask[pos]:
                idx.append(pos)
            j += 1
        counts = np.bincount(np.asarray(idx), minlength=buffer_len)
        expected = np.zeros(n, dtype=np.float64)
        # LLR 符号：d=0 -> +amp，d=1 -> -amp，重复 count 次则相加。
        expected[:buffer_len] = (
            amplitude * counts * (1.0 - 2.0 * d[:buffer_len].astype(float))
        )
        expected[plan.filler_mask] = FILLER_LLR
        np.testing.assert_allclose(recovered, expected)
        self.assertTrue(np.all(recovered[plan.filler_mask] == FILLER_LLR))

    def test_rv_and_lengths(self):
        for rv in range(4):
            for qm in (2, 4):
                for e_len in (100, self.plan.n, 2 * self.plan.n + 38):
                    if e_len % qm:
                        continue
                    with self.subTest(rv=rv, qm=qm, e_len=e_len):
                        self._check_case(rv, qm, e_len)

    def test_limited_buffer(self):
        for rv in range(4):
            with self.subTest(rv=rv):
                self._check_case(rv, 2, 300, n_cb=600)

    def test_limited_buffer_cuts_filler(self):
        """n_cb 落在 filler 区间内时，缓存外的 filler 仍应标为已知 0。"""
        plan = self.plan
        n_cb = plan.filler_start_d + 5
        e_len = 4
        rec = rate_recover(
            np.ones(e_len), 0, 2, plan.bgn, plan.z, plan.filler_mask, n_cb
        )
        # 缓存内和缓存外的所有 filler 位置都注入 FILLER_LLR。
        self.assertTrue(np.all(rec[plan.filler_mask] == FILLER_LLR))

    def test_zero_length(self):
        d = np.zeros(self.plan.n, dtype=np.uint8)
        e = rate_match(d, 0, 0, 2, self.plan.bgn, self.plan.z, self.plan.filler_mask)
        self.assertEqual(e.size, 0)
        rec = rate_recover(e, 0, 2, self.plan.bgn, self.plan.z, self.plan.filler_mask)
        self.assertTrue(np.all(rec[~self.plan.filler_mask] == 0.0))

    def test_block_helpers(self):
        plan = self.plan
        d = self.rng.integers(0, 2, (2, plan.n), dtype=np.uint8)
        d[:, plan.filler_mask] = 0
        e_lengths = np.array([120, 80])
        coded = rate_match_code_blocks(
            d, e_lengths, 0, 2, plan.bgn, plan.z, plan.filler_mask
        )
        self.assertEqual(coded.size, int(e_lengths.sum()))
        parts = rate_recover_code_blocks(
            np.where(coded == 0, 2.0, -2.0),
            e_lengths,
            0,
            2,
            plan.bgn,
            plan.z,
            plan.filler_mask,
        )
        self.assertEqual(len(parts), 2)
        for part in parts:
            self.assertEqual(part.size, plan.n)


if __name__ == "__main__":
    unittest.main()
