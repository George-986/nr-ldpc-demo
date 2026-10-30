"""完整 PUSCH 接收链：解扰 -> 解交织 -> 解速率匹配 -> LDPC -> CRC -> TB 拼接。"""

from __future__ import annotations

import math
import unittest

import numpy as np

from nr_pusch_ldpc_rx.link.receiver import (
    PUSCHReceiverConfig,
    decode_transport_block,
    decode_transport_block_from_bits,
)
from nr_pusch_ldpc_rx.chain.scrambling import PUSCHScramblingConfig
from nr_pusch_ldpc_rx.chain.segmentation import plan_transport_block
from nr_pusch_ldpc_rx.link.transmitter import encode_transport_block


# (A, R, Qm, rv) —— 覆盖 BG1/BG2、Kb=6/8/9/10、单/多码块、Zc=5..384。
CASES = [
    (10, 0.30, 2, 0),
    (100, 0.30, 2, 0),
    (100, 0.10, 2, 0),   # G > N，覆盖循环缓存重复读取/LLR 合并
    (177, 0.30, 2, 0),
    (545, 0.30, 2, 0),
    (1000, 0.30, 6, 0),
    (5000, 0.80, 2, 0),
    (20000, 0.70, 4, 2),
    (5000, 0.20, 2, 3),
    (11000, 0.20, 6, 1),
    (3824, 0.67, 8, 0),
    (3825, 0.80, 4, 0),
]


def _scrambling_for(a: int) -> PUSCHScramblingConfig:
    """为每个用例生成确定性的、互不相同的加扰配置。"""
    return PUSCHScramblingConfig(
        n_rnti=0x4600 + (a % 200),
        n_id=(a * 7) % 1024,
    )


def _coded_bits(plan, rv: int, qm: int) -> int:
    """给测试选择一个合适的 G：rv=0 按目标码率，其他 rv 覆盖整个缓存。"""
    unit = qm
    raw = plan.c * plan.n if rv else math.ceil(plan.b / plan.rate)
    return math.ceil(raw / unit) * unit


class TestEndToEnd(unittest.TestCase):
    def test_case_matrix(self):
        """多组 BG/Kb/Zc/Qm/rv 的无噪声端到端往返。"""
        rng = np.random.default_rng(2027)
        for (a, rate, qm, rv) in CASES:
            with self.subTest(a=a, rate=rate, qm=qm, rv=rv):
                plan = plan_transport_block(a, rate)
                g = _coded_bits(plan, rv, qm)
                scr = _scrambling_for(a)
                tb = rng.integers(0, 2, a, dtype=np.uint8)
                encoded = encode_transport_block(
                    tb, g, rv=rv, qm=qm, plan=plan, scrambling=scr
                )
                self.assertTrue(encoded.scrambled)
                # 加扰确实改变了码块拼接后的比特。
                self.assertFalse(
                    np.array_equal(encoded.coded_bits, encoded.concatenated_bits)
                )
                cfg = PUSCHReceiverConfig(
                    transport_block_size=a,
                    target_rate=rate,
                    rv=rv,
                    modulation=qm,
                    max_ldpc_iter=30,
                    scrambling=scr,
                )
                result = decode_transport_block_from_bits(
                    encoded.coded_bits, cfg, plan=plan, amplitude=4.0
                )

                self.assertEqual(result.plan, plan)
                self.assertTrue(result.tb_crc_ok)
                self.assertTrue(result.cb_crc_ok)
                self.assertTrue(result.all_ldpc_converged)
                np.testing.assert_array_equal(result.tb_bits, tb)
                self.assertEqual(result.decoded_cbs.shape, (plan.c, plan.k))
                self.assertEqual(
                    [c.e_len for c in result.code_blocks],
                    [int(e) for e in encoded.e_lengths],
                )
                for c in result.code_blocks:
                    self.assertTrue(c.ldpc.converged)
                    self.assertLessEqual(c.ldpc.iterations, 30)
                # 核对实际覆盖的关键参数，确保测试确实覆盖不同 BG/Kb/Z。
                self.assertEqual(plan.n, (66 if plan.bgn == 1 else 50) * plan.z)
                self.assertEqual(plan.k, (22 if plan.bgn == 1 else 10) * plan.z)

    def test_awgn_soft_llr(self):
        """加入 AWGN 后，接收链仍应通过 CB CRC 和 TB CRC。"""
        a, rate, qm = 1000, 0.30, 2
        plan = plan_transport_block(a, rate)
        g = _coded_bits(plan, 0, qm)
        rng = np.random.default_rng(2028)
        scr = PUSCHScramblingConfig(n_rnti=0x1234, n_id=77)
        tb = rng.integers(0, 2, a, dtype=np.uint8)
        encoded = encode_transport_block(
            tb, g, qm=qm, plan=plan, scrambling=scr
        )

        ebn0_db = 2.0
        effective_rate = a / g
        sigma = math.sqrt(1.0 / (2.0 * effective_rate * 10 ** (ebn0_db / 10.0)))
        symbols = 1.0 - 2.0 * encoded.coded_bits.astype(np.float64)
        received = symbols + rng.normal(0.0, sigma, symbols.size)
        llr = 2.0 * received / (sigma * sigma)

        cfg = PUSCHReceiverConfig(
            a, rate, modulation=qm, max_ldpc_iter=50, scrambling=scr
        )
        result = decode_transport_block(llr, cfg, plan=plan)
        self.assertTrue(result.tb_crc_ok)
        self.assertTrue(result.cb_crc_ok)
        np.testing.assert_array_equal(result.tb_bits, tb)

    def test_bg1_multicb_and_bg2_multicb(self):
        """重点验证多码块 CB CRC、去 filler 和 TB 拼接。"""
        rng = np.random.default_rng(30)
        for a, rate, qm, rv in ((20000, 0.70, 4, 2), (5000, 0.20, 2, 3)):
            with self.subTest(a=a):
                plan = plan_transport_block(a, rate)
                self.assertGreater(plan.c, 1)
                g = _coded_bits(plan, rv, qm)
                scr = _scrambling_for(a)
                tb = rng.integers(0, 2, a, dtype=np.uint8)
                enc = encode_transport_block(
                    tb, g, rv=rv, qm=qm, plan=plan, scrambling=scr
                )
                res = decode_transport_block_from_bits(
                    enc.coded_bits,
                    PUSCHReceiverConfig(
                        a, rate, rv=rv, modulation=qm, scrambling=scr
                    ),
                    plan=plan,
                    amplitude=4.0,
                )
                self.assertTrue(res.tb_crc_ok)
                self.assertTrue(all(cb.crc.crc_present for cb in res.code_blocks))
                self.assertTrue(all(cb.crc.crc_ok for cb in res.code_blocks))
                np.testing.assert_array_equal(res.tb_bits, tb)

    def test_modulation_orders(self):
        """BPSK/QPSK/16QAM/64QAM/256QAM 的 Qm 分支都应可用。"""
        a, rate = 1000, 0.30
        plan = plan_transport_block(a, rate)
        rng = np.random.default_rng(31)
        tb = rng.integers(0, 2, a, dtype=np.uint8)
        scr = PUSCHScramblingConfig(n_rnti=0x0001, n_id=0)
        for modulation, qm in (
            ("BPSK", 1),
            ("QPSK", 2),
            ("16QAM", 4),
            ("64QAM", 6),
            ("256QAM", 8),
        ):
            with self.subTest(modulation=modulation):
                g = math.ceil(math.ceil(plan.b / rate) / qm) * qm
                enc = encode_transport_block(
                    tb, g, qm=qm, plan=plan, scrambling=scr
                )
                res = decode_transport_block_from_bits(
                    enc.coded_bits,
                    PUSCHReceiverConfig(
                        a, rate, modulation=modulation, scrambling=scr
                    ),
                    plan=plan,
                )
                self.assertTrue(res.tb_crc_ok)
                np.testing.assert_array_equal(res.tb_bits, tb)

    def test_multiple_layers(self):
        """N_L>1 时 E_r 仍应覆盖整个 G，接收链正常工作。"""
        a, rate, qm, nl = 1000, 0.30, 2, 2
        plan = plan_transport_block(a, rate)
        unit = qm * nl
        g = math.ceil(math.ceil(plan.b / rate) / unit) * unit
        rng = np.random.default_rng(32)
        scr = PUSCHScramblingConfig(n_rnti=0xFFFF, n_id=1023)
        tb = rng.integers(0, 2, a, dtype=np.uint8)
        enc = encode_transport_block(
            tb, g, qm=qm, n_layers=nl, plan=plan, scrambling=scr
        )
        res = decode_transport_block_from_bits(
            enc.coded_bits,
            PUSCHReceiverConfig(
                a, rate, modulation=qm, n_layers=nl, scrambling=scr
            ),
            plan=plan,
        )
        self.assertTrue(res.tb_crc_ok)
        np.testing.assert_array_equal(res.tb_bits, tb)
        self.assertEqual(int(enc.e_lengths.sum()), g)

    def test_descrambling_is_mandatory(self):
        """未解扰或解扰 ID 错误时，TB CRC 必须失败。"""
        a, rate, qm = 1000, 0.30, 2
        plan = plan_transport_block(a, rate)
        g = _coded_bits(plan, 0, qm)
        rng = np.random.default_rng(33)
        tb = rng.integers(0, 2, a, dtype=np.uint8)
        good = PUSCHScramblingConfig(n_rnti=0x4601, n_id=321)
        enc = encode_transport_block(
            tb, g, qm=qm, plan=plan, scrambling=good
        )

        # 1) 完全不解扰：相当于把加扰比特当成原始比特。
        no_descramble = decode_transport_block_from_bits(
            enc.coded_bits,
            PUSCHReceiverConfig(a, rate, modulation=qm),
            plan=plan,
            amplitude=4.0,
        )
        self.assertFalse(no_descramble.tb_crc_ok)

        # 2) RNTI 错误：加扰序列不同，解扰后的 LLR 一半符号是反的。
        wrong_rnti = decode_transport_block_from_bits(
            enc.coded_bits,
            PUSCHReceiverConfig(
                a,
                rate,
                modulation=qm,
                scrambling=PUSCHScramblingConfig(n_rnti=0x4602, n_id=321),
            ),
            plan=plan,
            amplitude=4.0,
        )
        self.assertFalse(wrong_rnti.tb_crc_ok)

        # 3) n_ID 错误。
        wrong_id = decode_transport_block_from_bits(
            enc.coded_bits,
            PUSCHReceiverConfig(
                a,
                rate,
                modulation=qm,
                scrambling=PUSCHScramblingConfig(n_rnti=0x4601, n_id=322),
            ),
            plan=plan,
            amplitude=4.0,
        )
        self.assertFalse(wrong_id.tb_crc_ok)

        # 4) 正确配置可以通过，证明失败不是链路本身的问题。
        ok = decode_transport_block_from_bits(
            enc.coded_bits,
            PUSCHReceiverConfig(a, rate, modulation=qm, scrambling=good),
            plan=plan,
            amplitude=4.0,
        )
        self.assertTrue(ok.tb_crc_ok)
        np.testing.assert_array_equal(ok.tb_bits, tb)

    def test_msg_a_scrambling(self):
        """msgA 配置（RA-RNTI + n_RAPID）也应在端到端链路中工作。"""
        a, rate, qm = 300, 0.30, 2
        plan = plan_transport_block(a, rate)
        g = _coded_bits(plan, 0, qm)
        rng = np.random.default_rng(34)
        tb = rng.integers(0, 2, a, dtype=np.uint8)
        scr = PUSCHScramblingConfig(
            n_rnti=0x2A1, n_id=9, msg_a=True, n_rapid=17
        )
        enc = encode_transport_block(
            tb, g, qm=qm, plan=plan, scrambling=scr
        )
        res = decode_transport_block_from_bits(
            enc.coded_bits,
            PUSCHReceiverConfig(a, rate, modulation=qm, scrambling=scr),
            plan=plan,
            amplitude=4.0,
        )
        self.assertTrue(res.tb_crc_ok)
        np.testing.assert_array_equal(res.tb_bits, tb)

    def test_invalid_config_and_length(self):
        plan = plan_transport_block(100, 0.3)
        with self.assertRaises(ValueError):
            decode_transport_block(
                np.zeros(100), PUSCHReceiverConfig(100, 0.3, rv=4)
            )
        with self.assertRaises(ValueError):
            decode_transport_block(np.zeros(101), PUSCHReceiverConfig(100, 0.3))
        with self.assertRaises(ValueError):
            decode_transport_block(
                np.zeros(100), PUSCHReceiverConfig(101, 0.3), plan=plan
            )
        with self.assertRaises(ValueError):
            decode_transport_block(np.zeros(1), PUSCHReceiverConfig(0, 0.3))

    def test_result_properties(self):
        """结果对象的汇总属性应正确。"""
        a, rate = 100, 0.3
        plan = plan_transport_block(a, rate)
        tb = np.zeros(a, dtype=np.uint8)
        scr = PUSCHScramblingConfig(n_rnti=7, n_id=9)
        enc = encode_transport_block(
            tb, 400, qm=2, plan=plan, scrambling=scr
        )
        res = decode_transport_block_from_bits(
            enc.coded_bits,
            PUSCHReceiverConfig(a, rate, scrambling=scr),
            plan=plan,
        )
        self.assertTrue(res.cb_crc_ok)
        self.assertTrue(res.all_ldpc_converged)
        self.assertEqual(len(res.code_blocks), 1)


if __name__ == "__main__":
    unittest.main()
