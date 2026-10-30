"""CRC16/CRC24A/CRC24B 单元测试。"""

from __future__ import annotations

import unittest

import numpy as np

from nr_pusch_ldpc_rx.chain.crc import (
    crc_attach,
    crc_bits,
    crc_check,
    crc_remainder,
    crc_strip,
    crc_width,
)


def _ascii_bits(text: str) -> list[int]:
    """把 ASCII 字符串展开为 MSB-first 的比特列表（测试向量用）。"""
    out: list[int] = []
    for ch in text:
        value = ord(ch)
        out.extend((value >> shift) & 1 for shift in range(7, -1, -1))
    return out


class TestCRC(unittest.TestCase):
    def test_known_check_values(self):
        """CRC-16/XMODEM、CRC-24A、CRC-24B 对 "123456789" 的已知余数。"""
        data = _ascii_bits("123456789")
        self.assertEqual(crc_remainder(data, "16"), 0x31C3)
        self.assertEqual(crc_remainder(data, "24A"), 0xCDE703)
        self.assertEqual(crc_remainder(data, "24B"), 0x23EF52)

    def test_attach_yields_zero_remainder(self):
        """附加 CRC 后，整块对生成多项式的余数必须为 0。"""
        rng = np.random.default_rng(1234)
        for kind, width in (("16", 16), ("24A", 24), ("24B", 24)):
            with self.subTest(kind=kind):
                info = rng.integers(0, 2, 777, dtype=np.uint8)
                block = crc_attach(info, kind)
                self.assertEqual(block.size, info.size + width)
                self.assertEqual(crc_width(kind), width)
                self.assertTrue(crc_check(block, kind))
                self.assertTrue(np.array_equal(block[:-width], info))
                np.testing.assert_array_equal(block[-width:], crc_bits(info, kind))

    def test_single_bit_error_detected(self):
        """任意单个比特翻转都应被 CRC 检出。"""
        rng = np.random.default_rng(9)
        for kind in ("16", "24A", "24B"):
            info = rng.integers(0, 2, 64, dtype=np.uint8)
            block = crc_attach(info, kind)
            for pos in range(block.size):
                bad = block.copy()
                bad[pos] ^= 1
                with self.subTest(kind=kind, pos=pos):
                    self.assertFalse(crc_check(bad, kind))

    def test_strip(self):
        """crc_strip 返回原始信息并给出校验标志。"""
        info = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        block = crc_attach(info, "24B")
        recovered, ok = crc_strip(block, "24B")
        self.assertTrue(ok)
        np.testing.assert_array_equal(recovered, info)
        bad = block.copy()
        bad[0] ^= 1
        _, ok_bad = crc_strip(bad, "24B")
        self.assertFalse(ok_bad)

    def test_invalid_inputs(self):
        """非法 CRC 类型、非 0/1 输入、过短块应报错。"""
        with self.assertRaises(ValueError):
            crc_width("24C")
        with self.assertRaises(ValueError):
            crc_remainder([0, 1, 2], "16")
        with self.assertRaises(ValueError):
            crc_strip([0, 1, 0], "16")


if __name__ == "__main__":
    unittest.main()
