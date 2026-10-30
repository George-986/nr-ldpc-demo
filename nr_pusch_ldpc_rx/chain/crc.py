"""TS 38.212 5.1 规定的 CRC16、CRC24A、CRC24B 计算与校验。

在包内的位置
------------
本模块是传输链最底层、无状态的原语：被 :mod:`.segmentation` 用来给
TB 加 CRC24A/CRC16、给每个 CB 加 CRC24B，也被接收端用来校验。它不依赖
包内其它模块，便于独立测试。

把比特看成 GF(2) 多项式
-----------------------
协议把比特序列看成 GF(2) 上的多项式，最高位（最早发送的比特）是最高
次项。例如 ``a = [a0, a1, ..., a(A-1)]`` 对应::

    a(D) = a0*D^(A-1) + a1*D^(A-2) + ... + a(A-1)

CRC 附加比特定义为 ``a(D) * D^L mod g(D)`` 的系数，按从高次到低次的
顺序放在信息比特之后。因此把“信息 + CRC”整体再除以 ``g(D)``，余数
必为 0，这就是 :func:`crc_check` 的原理。

三个生成多项式（TS 38.212 5.1）::

    gCRC16(D)  = D^16 + D^12 + D^5 + 1
    gCRC24A(D) = D^24 + D^23 + D^18 + D^17 + D^14 + D^11 + D^10
                 + D^7 + D^6 + D^5 + D^4 + D^3 + D + 1
    gCRC24B(D) = D^24 + D^23 + D^6 + D^5 + D + 1

实现方式
--------
用逐比特 LFSR，输入/输出均为 MSB-first，和 38.212 的比特顺序一致。
数据量在本项目的测试规模下（几十万比特）足够快，且不引入第三方依赖。
LFSR 的反馈抽头与最高位异或后回馈，等价于“先给信息比特补 L 个 0 再做
多项式除法”，所以逐位处理完信息就能直接得到 ``a(D)*D^L mod g(D)``，
不需要显式追加 0 比特。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# CRC 名称 -> (宽度 L, 去掉最高次项后的生成多项式低位)
CRC_POLYNOMIALS: dict[str, tuple[int, int]] = {
    "16": (16, 0x1021),    # x^16 + x^12 + x^5 + 1
    "24A": (24, 0x864CFB),  # x^24 + x^23 + x^18 + ... + 1
    "24B": (24, 0x800063),  # x^24 + x^23 + x^6 + x^5 + x + 1
}


def crc_width(kind: str) -> int:
    """返回 CRC 类型对应的校验位数 L（16 或 24）。"""
    try:
        return CRC_POLYNOMIALS[kind][0]
    except KeyError as exc:
        raise ValueError(f"未知 CRC 类型 {kind!r}，可选 {sorted(CRC_POLYNOMIALS)}") from exc


def _as_bit_array(bits: Iterable[int]) -> np.ndarray:
    """把输入转成 uint8 一维数组，并检查每个元素只能是 0/1。"""
    arr = np.asarray(list(bits), dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError("CRC 输入必须是一维比特序列")
    if arr.size and np.any(arr > 1):
        raise ValueError("CRC 输入只能包含 0/1（或可转换为 0/1 的整数）")
    return arr


def crc_remainder(bits: Iterable[int], kind: str) -> int:
    """计算比特序列对指定生成多项式的余数（整数形式）。

    Parameters
    ----------
    bits:
        信息比特（或包含 CRC 的完整码块）序列，MSB-first。
    kind:
        ``"16"``、``"24A"`` 或 ``"24B"``。

    Returns
    -------
    int
        宽度为 L 的余数；对“信息 + CRC”整体计算时应为 0。
    """
    arr = _as_bit_array(bits)
    width, poly_low = CRC_POLYNOMIALS[kind]
    mask = (1 << width) - 1
    reg = 0
    top = width - 1

    for bit in arr:
        # 寄存器保存的是“当前余数”的低 L 位。把最高位与输入比特异或，
        # 得到该比特对当前最高次项的修正；若为 1，则左移后异或生成
        # 多项式（去掉最高次项，因为它在 L 位寄存器外被消掉）。
        feedback = ((reg >> top) & 1) ^ int(bit)
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly_low
    return int(reg)


def crc_bits(bits: Iterable[int], kind: str) -> np.ndarray:
    """返回 CRC 校验比特（长度 L，MSB-first），不附加到输入。"""
    width = crc_width(kind)
    rem = crc_remainder(bits, kind)
    return np.array([(rem >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def crc_attach(bits: Iterable[int], kind: str) -> np.ndarray:
    """在信息比特末尾附加 CRC，返回长度为 ``len(bits)+L`` 的 uint8 数组。"""
    arr = _as_bit_array(bits)
    return np.concatenate((arr, crc_bits(arr, kind)))


def crc_check(bits_with_crc: Iterable[int], kind: str) -> bool:
    """校验包含 CRC 的完整比特块。

    由于 CRC 是按 ``message*D^L mod g(D)`` 附加的，把“信息 + CRC”整体
    再除以 g(D)，余数为 0 即通过。该检查不返回原始信息。
    """
    return crc_remainder(bits_with_crc, kind) == 0


def crc_strip(bits_with_crc: Iterable[int], kind: str) -> tuple[np.ndarray, bool]:
    """去掉末尾 L 个 CRC 比特并校验，返回 ``(信息比特, 是否通过)``。"""
    arr = _as_bit_array(bits_with_crc)
    width = crc_width(kind)
    if arr.size < width:
        raise ValueError(f"输入长度 {arr.size} 小于 CRC 宽度 {width}")
    info = arr[:-width]
    return info, crc_check(arr, kind)
