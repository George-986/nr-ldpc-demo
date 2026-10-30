"""TS 38.211 5.2.1 / 6.3.1.1：PUSCH 加扰与解扰。

在链路中的位置
--------------
加扰在发送端位于**码块拼接之后、调制之前**，作用在整个 G 长码字上::

    b(0..G-1) --加扰 c(i)--> b~(0..G-1) --调制--> 符号

因此接收端必须在**按码块切分/解交织之前**先解扰（``receiver`` 的第 0
步），否则循环缓存地址和 LLR 的符号都会错。

加扰与解扰
----------
加扰是逐比特异或::

    b~(i) = (b(i) + c(i)) mod 2

异或自逆，所以解扰用同一序列再做一次异或。在 LLR 域中相当于按序列
翻转软比特的符号::

    LLR_b(i) = (1 - 2*c(i)) * LLR_b~(i)

这一步是无损的，也是它必须放在最前面的原因：后面的解交织、解速率
匹配都在做“位置置换/合并”，先解扰能让每个 LLR 对应正确的编码比特。

Gold 序列（TS 38.211 5.2.1）
----------------------------
``c(i)`` 由两条 31 位 m 序列异或而成，先各跑 ``Nc=1600`` 步预热::

    x1(n+31) = (x1(n+3) + x1(n)) mod 2
    x2(n+31) = (x2(n+3) + x2(n+2) + x2(n+1) + x2(n)) mod 2
    c(n)     = (x1(n+1600) + x2(n+1600)) mod 2

``x1`` 初值固定为 ``x1(0)=1``、其余为 0；``x2`` 的初值由 ``c_init`` 的
低 31 位逐位给出（bit0 对应 x2(0)）。

加扰序列初始化（TS 38.211 6.3.1.1）::

    c_init = n_RNTI * 2^16 + n_RAPID * 2^10 + n_ID      (msgA on PUSCH)
    c_init = n_RNTI * 2^15 + q * 2^14 + n_ID            (otherwise)

协议符号
--------
``q``        : 码字号，单码字传输时 q=0；
``n_RNTI``   : 与该 PUSCH 关联的 RNTI；msgA 时为 RA-RNTI；
``n_RAPID``  : msgA 所用随机接入前导索引（6 bit，0..63）；
``n_ID``     : ``dataScramblingIdentityPUSCH``（若配置且 RNTI 为
               C-RNTI/MCS-C-RNTI/SP-CSI-RNTI/CS-RNTI 且不是公共搜索空间
               的 DCI 0_0），msgA 时为 ``msgA-DataScramblingIndex``，
               否则为小区 ID ``N_ID^cell``；
``c_init``   : Gold 序列发生器的 31 比特初值。

协议规定 UCI 复用到 PUSCH 时会出现 ``x``/``y`` 占位比特；本模块只处理
纯 UL-SCH（数据）传输，不建模 UCI 复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# Gold 序列参数（TS 38.211 5.2.1）：两级 31 位 m 序列，输出偏移 Nc。
GOLD_SEQUENCE_LEN = 31
GOLD_NC = 1600

# c_init 只有低 31 位参与 x2 的初始化：x2(i) = bit_i(c_init), i=0..30。
_C_INIT_MASK = (1 << GOLD_SEQUENCE_LEN) - 1

# x1 序列与 c_init 无关，可以跨调用复用缓存。
_x1_cache = np.zeros(0, dtype=np.uint8)
_x1_state = 1  # x1(0)=1，其余 30 位为 0


def _x1_prefix(count: int) -> np.ndarray:
    """返回 ``x1(0..count-1)``，并在内部缓存已计算的前缀。

    x1 的递推为 ``x1(n+31) = (x1(n+3) + x1(n)) mod 2``。把最近 31 个比特
    放在整数寄存器里（bit0 对应 x1(n)，bit30 对应 x1(n+30)），每次
    右移一位并回填新算出的 x1(n+31)，避免在 Python 中逐元素维护列表。

    寄存器更新：``(state >> 3) ^ state`` 的最低位就是
    ``x1(n+3) ^ x1(n)``；``state >> 1`` 让所有下标前进 1，新比特放到
    bit30。x1 与 ``c_init`` 无关，因此缓存可以在所有调用间复用。
    """
    global _x1_cache, _x1_state
    if _x1_cache.size < count:
        need = count - _x1_cache.size
        fresh = np.empty(need, dtype=np.uint8)
        state = _x1_state
        for k in range(need):
            fresh[k] = state & 1
            nxt = ((state >> 3) ^ state) & 1
            state = (state >> 1) | (nxt << (GOLD_SEQUENCE_LEN - 1))
        _x1_cache = np.concatenate((_x1_cache, fresh))
        _x1_state = state
    return _x1_cache


def gold_sequence(length: int, c_init: int) -> np.ndarray:
    """生成 TS 38.211 5.2.1 的伪随机（Gold）序列 ``c(0..length-1)``。

    Parameters
    ----------
    length:
        需要的序列长度，必须是非负整数。长度 0 返回空数组。
    c_init:
        Gold 序列初值。协议中由 RNTI、码字号和 ID 组合得到，低位对应
        ``x2`` 的低次项。只有低 31 位参与初始化，因此这里对 2^31 取模
        （msgA 的公式可能产生 32 比特值）。

    Returns
    -------
    numpy.ndarray
        长度 ``length`` 的 uint8 数组，元素为 0/1::

            c(n) = (x1(n + 1600) + x2(n + 1600)) mod 2

    示例
    ----
    >>> seq = gold_sequence(4, 0)
    >>> seq.shape
    (4,)
    """
    if isinstance(length, bool) or not isinstance(length, (int, np.integer)):
        raise TypeError("length 必须是整数")
    if length < 0:
        raise ValueError("length 不能为负")
    if isinstance(c_init, bool) or not isinstance(c_init, (int, np.integer)):
        raise TypeError("c_init 必须是整数")
    if c_init < 0:
        raise ValueError("c_init 不能为负")
    if length == 0:
        return np.empty(0, dtype=np.uint8)

    count = int(length) + GOLD_NC
    x1 = _x1_prefix(count)[GOLD_NC:count]

    # x2 的初值由 c_init 逐位给出：x2(i) = bit_i(c_init)，i=0..30。
    # 注意是“最低位对应 x2(0)”，不是 MSB-first；写反会让序列完全不同。
    # 递推 x2(n+31) = x2(n+3)^x2(n+2)^x2(n+1)^x2(n) 同样用整数寄存器实现。
    x2 = np.empty(count, dtype=np.uint8)
    state = int(c_init) & _C_INIT_MASK
    for k in range(count):
        x2[k] = state & 1
        nxt = ((state >> 3) ^ (state >> 2) ^ (state >> 1) ^ state) & 1
        state = (state >> 1) | (nxt << (GOLD_SEQUENCE_LEN - 1))

    # c(n) 使用 n+1600 处的两级序列，故截取 [Nc, Nc+length)。x1 只取
    # 这一窗口（前面的 1600 位仅用于预热），x2 则必须从头生成。
    return (x1 ^ x2[GOLD_NC:count]).astype(np.uint8, copy=False)


@dataclass(frozen=True)
class PUSCHScramblingConfig:
    """PUSCH 加扰参数（TS 38.211 6.3.1.1）。

    Attributes
    ----------
    n_rnti:
        与该 PUSCH 关联的 RNTI，范围 0..65535；msgA 时为 RA-RNTI。
    n_id:
        ``dataScramblingIdentityPUSCH``；未配置时调用方应填入小区 ID
        ``N_ID^cell``（0..1023）。msgA 时为 ``msgA-DataScramblingIndex``。
    codeword_index:
        码字号 q，取值 0 或 1。PUSCH 单码字传输恒为 0；q=1 仅用于
        按协议公式生成不同的初值。
    msg_a:
        是否为 msgA（Type-2 随机接入）触发的 PUSCH。True 时使用
        ``n_RNTI*2^16 + n_RAPID*2^10 + n_ID``，且必须提供 ``n_rapid``。
    n_rapid:
        msgA 前导索引，0..63，仅在 ``msg_a=True`` 时使用。
    """

    n_rnti: int
    n_id: int = 0
    codeword_index: int = 0
    msg_a: bool = False
    n_rapid: int | None = None

    def __post_init__(self) -> None:
        """校验字段范围与组合关系，尽早暴露配置错误。"""
        if isinstance(self.n_rnti, bool) or not isinstance(self.n_rnti, int):
            raise TypeError("n_rnti 必须是整数")
        if not (0 <= self.n_rnti <= 65535):
            raise ValueError("n_rnti 必须在 0..65535 内")
        if isinstance(self.n_id, bool) or not isinstance(self.n_id, int):
            raise TypeError("n_id 必须是整数")
        if not (0 <= self.n_id <= 1023):
            raise ValueError("n_id 必须在 0..1023 内")
        if self.codeword_index not in (0, 1):
            raise ValueError("codeword_index 必须为 0 或 1")
        if not isinstance(self.msg_a, bool):
            raise TypeError("msg_a 必须是 bool")
        if self.msg_a:
            if self.n_rapid is None:
                raise ValueError("msg_a=True 时必须提供 n_rapid")
            if isinstance(self.n_rapid, bool) or not isinstance(self.n_rapid, int):
                raise TypeError("n_rapid 必须是整数")
            if not (0 <= self.n_rapid <= 63):
                raise ValueError("n_rapid 必须在 0..63 内")
        elif self.n_rapid is not None:
            raise ValueError("n_rapid 只在 msg_a=True 时使用")

    @property
    def c_init(self) -> int:
        """按 TS 38.211 6.3.1.1 计算 Gold 序列初值 ``c_init``。

        非 msgA 时结果不超过 2^31-1；msgA 公式可能达到 2^32-1，
        而 Gold 序列的 x2 只有 31 位初值，因此 ``gold_sequence`` 会取
        低 31 位（高位的 bit31 无法进入 x2 寄存器）。
        """
        if self.msg_a:
            # msgA on PUSCH：n_RNTI 为 RA-RNTI（16 bit），n_RAPID 为 6 bit。
            # 注意这里 n_RNTI 左移 16 位（而不是普通公式的 15 位），
            # 因此结果最高可到 2^32-1，超过 31 位；gold_sequence 会截断
            # 低 31 位，因为高位的 bit31 无法进入 x2 寄存器。
            assert self.n_rapid is not None  # __post_init__ 已保证
            return self.n_rnti * 2**16 + self.n_rapid * 2**10 + self.n_id
        return (
            self.n_rnti * 2**15
            + self.codeword_index * 2**14
            + self.n_id
        )

    def sequence(self, length: int) -> np.ndarray:
        """返回该配置下长度为 ``length`` 的加扰序列 ``c(i)``。"""
        return gold_sequence(length, self.c_init)


def pusch_scrambling_c_init(
    n_rnti: int,
    n_id: int = 0,
    codeword_index: int = 0,
    msg_a: bool = False,
    n_rapid: int | None = None,
) -> int:
    """便捷函数：直接根据参数计算 ``c_init``（等价于配置对象的属性）。"""
    return PUSCHScramblingConfig(
        n_rnti=n_rnti,
        n_id=n_id,
        codeword_index=codeword_index,
        msg_a=msg_a,
        n_rapid=n_rapid,
    ).c_init


def scramble_bits(bits: Iterable[int], sequence: np.ndarray) -> np.ndarray:
    """对比特序列加扰：``b~(i) = (b(i) + c(i)) mod 2``。

    Parameters
    ----------
    bits:
        长度与 ``sequence`` 相同的 0/1 序列。
    sequence:
        ``gold_sequence`` 或 ``PUSCHScramblingConfig.sequence`` 的结果。

    Returns
    -------
    numpy.ndarray
        加扰后的 uint8 数组。异或与加法模 2 等价，因此本函数同时是解扰。
    """
    arr = np.asarray(list(bits), dtype=np.uint8).reshape(-1)
    seq = np.asarray(sequence, dtype=np.uint8).reshape(-1)
    if arr.size != seq.size:
        raise ValueError(
            f"比特数 {arr.size} 与加扰序列长度 {seq.size} 不一致"
        )
    if arr.size and np.any(arr > 1):
        raise ValueError("bits 只能包含 0/1")
    return arr ^ seq


def descramble_llr(
    llr: Iterable[float],
    sequence: np.ndarray,
) -> np.ndarray:
    """在 LLR 域解扰：``c=1`` 的位置翻转符号，``c=0`` 的位置保持不变。

    对 BPSK/QAM 的比特级 LLR，加扰相当于把发送符号乘以 ``1-2c``，
    因此解扰就是对数似然比乘以 ``1-2c``。该操作是无损的，也是
    解扰应当在解速率匹配之前完成的原因。

    Parameters
    ----------
    llr:
        对应加扰后比特 ``b~`` 的软值，正数倾向 0，负数倾向 1。
    sequence:
        长度为 ``len(llr)`` 的加扰序列。

    Returns
    -------
    numpy.ndarray
        对应未加扰比特 ``b`` 的 LLR（float64）。
    """
    arr = np.asarray(list(llr), dtype=np.float64).reshape(-1)
    seq = np.asarray(sequence, dtype=np.uint8).reshape(-1)
    if arr.size != seq.size:
        raise ValueError(
            f"LLR 数 {arr.size} 与加扰序列长度 {seq.size} 不一致"
        )
    if seq.size and np.any(seq > 1):
        raise ValueError("加扰序列只能包含 0/1")
    return arr * (1.0 - 2.0 * seq)


def descramble_pusch_llr(
    llr: Iterable[float],
    scrambling: PUSCHScramblingConfig,
) -> np.ndarray:
    """按 PUSCH 配置生成序列并解扰，供接收链直接调用。"""
    arr = np.asarray(list(llr), dtype=np.float64).reshape(-1)
    return descramble_llr(arr, scrambling.sequence(arr.size))
