"""TS 38.211 5.2.1 Gold 序列与 6.3.1.1 PUSCH 加扰/解扰测试。"""

from __future__ import annotations

import unittest

import numpy as np

from nr_pusch_ldpc_rx.chain.scrambling import (
    GOLD_NC,
    GOLD_SEQUENCE_LEN,
    PUSCHScramblingConfig,
    descramble_llr,
    descramble_pusch_llr,
    gold_sequence,
    pusch_scrambling_c_init,
    scramble_bits,
)


def _reference_gold(length: int, c_init: int) -> np.ndarray:
    """按 TS 38.211 5.2.1 伪代码逐位实现的参考 Gold 序列。

    刻意写成最直白的形式（列表 + 递推），不复用被测模块的任何代码，
    用于验证向量化实现的正确性。
    """
    nc = GOLD_NC
    total = nc + length + GOLD_SEQUENCE_LEN
    x1 = [0] * total
    x2 = [0] * total
    x1[0] = 1
    for i in range(GOLD_SEQUENCE_LEN):
        x2[i] = (c_init >> i) & 1
    for n in range(total - GOLD_SEQUENCE_LEN):
        x1[n + 31] = (x1[n + 3] + x1[n]) % 2
        x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) % 2
    return np.array(
        [(x1[n + nc] + x2[n + nc]) % 2 for n in range(length)],
        dtype=np.uint8,
    )


class TestGoldSequence(unittest.TestCase):
    def test_matches_reference_implementation(self):
        """覆盖多种 c_init、长度，与逐位参考实现逐比特一致。"""
        cases = [
            (0, 1),
            (0, 40),
            (1, 100),
            (0xFFFF, 256),
            (32768, 1000),
            (123456789, 777),
            (2147468287, 300),  # 非 msgA 公式的最大值
            (4294967295, 64),   # msgA 公式可达到 32 位上限
        ]
        for c_init, length in cases:
            with self.subTest(c_init=c_init, length=length):
                np.testing.assert_array_equal(
                    gold_sequence(length, c_init),
                    _reference_gold(length, c_init),
                )

    def test_known_prefix(self):
        """固定已知前缀，防止参考实现与主实现同错。"""
        # 由 TS 38.211 5.2.1 生成；与 py3gpp nrPRBS(c_init, len) 交叉核对。
        known = {
            0: [0, 0, 0, 0, 0, 0, 1, 0],
            1: [0, 0, 0, 0, 0, 0, 1, 0],
            1023: [0, 0, 1, 1, 1, 1, 0, 0],
        }
        for c_init, expected in known.items():
            np.testing.assert_array_equal(
                gold_sequence(len(expected), c_init), expected
            )

    def test_prefix_property(self):
        """较短序列必须是较长序列的前缀（同一 c_init）。"""
        long_seq = gold_sequence(500, 987654)
        for length in (0, 1, 7, 100, 499):
            np.testing.assert_array_equal(
                gold_sequence(length, 987654), long_seq[:length]
            )

    def test_output_type_and_values(self):
        seq = gold_sequence(300, 42)
        self.assertEqual(seq.dtype, np.uint8)
        self.assertEqual(seq.size, 300)
        self.assertTrue(set(np.unique(seq)).issubset({0, 1}))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            gold_sequence(-1, 0)
        with self.assertRaises(ValueError):
            gold_sequence(10, -5)
        with self.assertRaises(TypeError):
            gold_sequence(10.0, 0)
        with self.assertRaises(TypeError):
            gold_sequence(10, 1.5)
        with self.assertRaises(TypeError):
            gold_sequence(True, 0)


class TestScramblingConfig(unittest.TestCase):
    def test_c_init_normal(self):
        """非 msgA：c_init = n_RNTI*2^15 + q*2^14 + n_ID。"""
        cfg = PUSCHScramblingConfig(n_rnti=0x4601, n_id=321)
        self.assertEqual(cfg.c_init, 0x4601 * 2**15 + 321)
        self.assertEqual(pusch_scrambling_c_init(0x4601, 321), cfg.c_init)
        self.assertEqual(cfg.c_init, cfg.n_rnti * 32768 + cfg.n_id)

        cfg_q1 = PUSCHScramblingConfig(n_rnti=1, n_id=0, codeword_index=1)
        self.assertEqual(cfg_q1.c_init, 2**15 + 2**14)

    def test_c_init_msg_a(self):
        """msgA：c_init = n_RNTI*2^16 + n_RAPID*2^10 + n_ID。"""
        cfg = PUSCHScramblingConfig(
            n_rnti=0x1234, n_id=5, msg_a=True, n_rapid=7
        )
        self.assertEqual(cfg.c_init, 0x1234 * 2**16 + 7 * 2**10 + 5)

    def test_sequence_matches_gold(self):
        cfg = PUSCHScramblingConfig(n_rnti=65535, n_id=1023)
        np.testing.assert_array_equal(
            cfg.sequence(64), gold_sequence(64, cfg.c_init)
        )

    def test_invalid_configs(self):
        with self.assertRaises(ValueError):
            PUSCHScramblingConfig(n_rnti=-1)
        with self.assertRaises(ValueError):
            PUSCHScramblingConfig(n_rnti=65536)
        with self.assertRaises(ValueError):
            PUSCHScramblingConfig(n_rnti=1, n_id=1024)
        with self.assertRaises(ValueError):
            PUSCHScramblingConfig(n_rnti=1, codeword_index=2)
        with self.assertRaises(TypeError):
            PUSCHScramblingConfig(n_rnti=True)
        # msgA 必须给 n_rapid，且范围是 6 比特。
        with self.assertRaises(ValueError):
            PUSCHScramblingConfig(n_rnti=1, msg_a=True)
        with self.assertRaises(ValueError):
            PUSCHScramblingConfig(n_rnti=1, msg_a=True, n_rapid=64)
        # 非 msgA 时给 n_rapid 属于配置错误。
        with self.assertRaises(ValueError):
            PUSCHScramblingConfig(n_rnti=1, n_rapid=3)


class TestScramblingOperations(unittest.TestCase):
    def test_scramble_is_xor(self):
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, 256, dtype=np.uint8)
        seq = gold_sequence(256, 1234)
        np.testing.assert_array_equal(
            scramble_bits(bits, seq), (bits + seq) % 2
        )

    def test_scramble_is_self_inverse(self):
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 2, 300, dtype=np.uint8)
        seq = gold_sequence(300, 77)
        np.testing.assert_array_equal(
            scramble_bits(scramble_bits(bits, seq), seq), bits
        )

    def test_descramble_llr_flips_sign(self):
        llr = np.array([1.0, -2.0, 3.0, -4.0])
        seq = np.array([0, 1, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(
            descramble_llr(llr, seq), [1.0, 2.0, 3.0, 4.0]
        )

    def test_bits_to_llr_roundtrip(self):
        """加扰比特的 LLR 解扰后，符号必须与原始比特一致。"""
        rng = np.random.default_rng(3)
        bits = rng.integers(0, 2, 400, dtype=np.uint8)
        seq = gold_sequence(400, 555)
        scrambled = scramble_bits(bits, seq)
        llr = np.where(scrambled == 0, 2.5, -2.5)
        recovered = descramble_llr(llr, seq)
        np.testing.assert_array_equal((recovered < 0).astype(np.uint8), bits)

    def test_descramble_pusch_llr(self):
        cfg = PUSCHScramblingConfig(n_rnti=12, n_id=3)
        bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        scrambled = scramble_bits(bits, cfg.sequence(bits.size))
        llr = np.where(scrambled == 0, 1.0, -1.0)
        recovered = descramble_pusch_llr(llr, cfg)
        np.testing.assert_array_equal((recovered < 0).astype(np.uint8), bits)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            scramble_bits([0, 1, 0], gold_sequence(2, 0))
        with self.assertRaises(ValueError):
            descramble_llr([1.0, 2.0], gold_sequence(3, 0))
        with self.assertRaises(ValueError):
            descramble_llr([1.0, 2.0], np.array([1, 2], dtype=np.uint8))

    def test_non_binary_input_rejected(self):
        with self.assertRaises(ValueError):
            scramble_bits([0, 2], gold_sequence(2, 0))


if __name__ == "__main__":
    unittest.main()
