"""PUSCH 接收端链路：解扰 -> 解比特交织 -> 解速率匹配 -> LDPC -> CRC -> TB 拼接。

在包内的位置
------------
本模块是接收侧的编排层：把 :mod:`~nr_pusch_ldpc_rx.chain` 的逆过程与
:mod:`~nr_pusch_ldpc_rx.ldpc` 的译码器串起来，输入解调后的 LLR，输出
TB 比特和各级 CRC 结果。信道估计、均衡、星座解映射不在本模块内，应在
调用前完成；这样接收链可以独立测试，也便于替换译码算法。

接收顺序与规范条款
------------------
0. ``38.211 6.3.1.1`` 解扰：按 Gold 序列翻转 ``G`` 个 LLR 的符号；
1. ``38.212 5.4.2.1`` 按 ``E_r`` 把整段 LLR 切成 ``C`` 段；
2. ``38.212 5.4.2.2`` 每段解比特交织（``Qm > 1``）；
3. ``38.212 5.4.2.1`` 解速率匹配：按 ``k0`` 绕循环缓存做 LLR 软合并；
4. ``38.212 5.3.2`` LDPC 译码：头部补 ``2*Zc`` 个 0 LLR（打孔位），
   filler 位置注入大正 LLR（已知 0）；
5. ``38.212 5.2.2`` CB CRC -> 去 filler/CRC -> TB 拼接 -> TB CRC。

输入约定
--------
``llr`` 的长度必须等于 ``G``，顺序是发送端“码块拼接后”的顺序，且对应
**加扰后**的比特。LLR 符号约定：**正数倾向比特 0，负数倾向比特 1**。

为什么第一步必须是解扰
----------------------
发送端在码块拼接之后、调制之前对整个码字加扰，接收到的 G 个软值是
``b~`` 的 LLR。只有先按 ``(1-2c)*LLR`` 恢复成 ``b``，后面按 ``E_r``
切分、解交织、按 ``k0`` 填循环缓存时，每个 LLR 才落在正确的编码比特
位置上。顺序搞反不会立即报错，但译码会随机失败（等价于 RNTI 配错）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chain.rate_matching import coded_bit_lengths, rate_recover
from ..chain.scrambling import PUSCHScramblingConfig, descramble_llr
from ..chain.segmentation import (
    CodeBlockCheck,
    TransportBlockPlan,
    check_transport_block_crc,
    desegment_code_blocks,
    plan_transport_block,
)
from ..constants import modulation_order
from ..ldpc.codec import LDPCDecodeResult
from ..ldpc.codec import decode as ldpc_decode


@dataclass(frozen=True)
class PUSCHReceiverConfig:
    """PUSCH 接收配置。

    Attributes
    ----------
    transport_block_size:
        TB 信息比特数 A，必须与发送端一致。
    target_rate:
        调度目标码率 R，用于按 TS 38.212 6.2.2 选择 BG。
    rv:
        冗余版本 0..3。
    modulation:
        调制方式字符串或 Qm。
    n_layers:
        传输层数 N_L。
    bgn:
        可选的强制 base graph（测试用）。
    max_ldpc_iter:
        LDPC 最大迭代次数。
    alpha:
        归一化最小和因子。
    early_stop:
        是否启用 H*c^T=0 早停。
    scrambling:
        PUSCH 解扰配置。``None`` 表示输入 LLR 未加扰（仅用于测试/调试，
        非协议行为）；正常接收必须提供 ``PUSCHScramblingConfig``。
    """

    transport_block_size: int
    target_rate: float
    rv: int = 0
    modulation: str | int = "QPSK"
    n_layers: int = 1
    bgn: int | None = None
    max_ldpc_iter: int = 30
    alpha: float = 0.8
    early_stop: bool = True
    scrambling: PUSCHScramblingConfig | None = None

    @property
    def qm(self) -> int:
        """配置对应的调制阶数 Qm。"""
        return modulation_order(self.modulation)


@dataclass(frozen=True)
class CodeBlockResult:
    """单个码块的译码与 CRC 结果。"""

    index: int
    e_len: int
    ldpc: LDPCDecodeResult
    crc: CodeBlockCheck


@dataclass(frozen=True)
class TransportBlockResult:
    """整个 TB 的接收结果。"""

    tb_bits: np.ndarray
    tb_crc_ok: bool
    plan: TransportBlockPlan
    code_blocks: tuple[CodeBlockResult, ...]
    decoded_cbs: np.ndarray

    @property
    def cb_crc_ok(self) -> bool:
        """所有码块的 CB CRC 是否都通过（C=1 时恒为 True）。"""
        return all(cb.crc.crc_ok for cb in self.code_blocks)

    @property
    def all_ldpc_converged(self) -> bool:
        """所有码块的 LDPC 校验方程是否都在最大迭代内满足。"""
        return all(cb.ldpc.converged for cb in self.code_blocks)


def decode_transport_block(
    llr: np.ndarray,
    config: PUSCHReceiverConfig,
    plan: TransportBlockPlan | None = None,
) -> TransportBlockResult:
    """执行完整的 PUSCH 接收链。

    Parameters
    ----------
    llr:
        长度 G 的解调后 LLR 序列，已按码块顺序拼接，范围为整个 TB，
        对应**加扰后**的编码比特。正数表示比特 0，负数表示比特 1。
        若 ``config.scrambling`` 给出，会先用同一 Gold 序列解扰。
    config:
        接收配置。
    plan:
        可选的预计算计划；若给出，其 A 必须与 config 一致。

    Returns
    -------
    TransportBlockResult
        TB 比特、TB CRC 结果、每个码块的迭代次数/CB CRC、LDPC 硬判决。

    Raises
    ------
    ValueError
        长度、Qm 或配置不合法时抛出；译码失败不抛异常，而是通过 CRC 和
        ``converged`` 标志报告，便于 BLER 统计。
    """
    if not (0.0 < config.target_rate < 1.0):
        raise ValueError("target_rate 必须在 (0, 1) 内")
    if config.rv not in (0, 1, 2, 3):
        raise ValueError("rv 必须为 0/1/2/3")
    if config.n_layers < 1:
        raise ValueError("n_layers 必须 >= 1")
    if config.max_ldpc_iter < 1:
        raise ValueError("max_ldpc_iter 必须 >= 1")
    if not (0.0 < config.alpha <= 1.0):
        raise ValueError("alpha 必须在 (0, 1] 内")

    if plan is None:
        plan = plan_transport_block(
            config.transport_block_size,
            config.target_rate,
            bgn=config.bgn,
        )
    elif plan.a != config.transport_block_size:
        raise ValueError(
            f"plan.a={plan.a} 与配置的 A={config.transport_block_size} 不一致"
        )
    # 说明：plan 一旦给出，BG/Kb/Zc/C/K/F 都以 plan 为准（发送端也必须
    # 使用同一 plan）；config.bgn 只在 plan 为空时参与推导。

    arr = np.asarray(llr, dtype=np.float64).reshape(-1)
    qm = config.qm
    if arr.size == 0:
        raise ValueError("LLR 不能为空")

    # 0) 解扰：在切成码块之前对整个 G 长序列做符号翻转。
    #    c(i) 在发送端与拼接后的 b(i) 对齐，所以这里序列长度就是 arr.size。
    if config.scrambling is not None:
        arr = descramble_llr(arr, config.scrambling.sequence(arr.size))

    e_lengths = coded_bit_lengths(arr.size, plan.c, qm, config.n_layers)

    decoded_cbs = np.zeros((plan.c, plan.k), dtype=np.uint8)
    decoded_list: list[LDPCDecodeResult] = []
    e_used: list[int] = []
    start = 0
    for r in range(plan.c):
        e_len = int(e_lengths[r])
        e_llr = arr[start:start + e_len]
        start += e_len

        # 1) 解交织 + 解速率匹配 -> 长度 N 的 d 域 LLR。
        #    rate_recover 内部已经按 filler_mask 把 NULL 位置填成
        #    FILLER_LLR（已知 0），所以 d 域可以直接接 LDPC。
        d_llr = rate_recover(
            e_llr,
            config.rv,
            qm,
            plan.bgn,
            plan.z,
            filler_mask=plan.filler_mask,
        )
        # 2) 补上被打孔的 2*Zc 个系统比特。它们没有发送，信道未提供任何
        #    信息，LLR=0 表示“后验等概”，由校验方程自行恢复。
        full_llr = np.concatenate(
            (np.zeros(2 * plan.z, dtype=np.float64), d_llr)
        )
        # 3) LDPC 分层 NMS 译码。
        decoded = ldpc_decode(
            full_llr,
            plan.bgn,
            plan.z,
            max_iter=config.max_ldpc_iter,
            alpha=config.alpha,
            early_stop=config.early_stop,
        )
        decoded_cbs[r] = decoded.info_bits
        decoded_list.append(decoded)
        e_used.append(e_len)

    # 4) CB CRC、去 filler/CRC、TB 拼接，再检查 TB CRC。
    #    CB CRC 失败不抛异常：链路级仿真需要继续统计 BLER/误比特，
    #    由调用方根据 checks / tb_crc_ok 决定是否丢弃该 TB。
    b_hat, checks = desegment_code_blocks(decoded_cbs, plan)
    tb_bits, tb_ok = check_transport_block_crc(b_hat, plan)

    cb_results = tuple(
        CodeBlockResult(
            index=r,
            e_len=e_used[r],
            ldpc=decoded_list[r],
            crc=checks[r],
        )
        for r in range(plan.c)
    )
    # 防御性检查：E_r 之和应等于 G，否则说明配置或切分有 bug。
    if start != arr.size:
        raise AssertionError("未消费完全部 LLR")

    return TransportBlockResult(
        tb_bits=tb_bits,
        tb_crc_ok=tb_ok,
        plan=plan,
        code_blocks=tuple(cb_results),
        decoded_cbs=decoded_cbs,
    )


def decode_transport_block_from_bits(
    coded_bits: np.ndarray,
    config: PUSCHReceiverConfig,
    plan: TransportBlockPlan | None = None,
    amplitude: float = 1.0,
) -> TransportBlockResult:
    """把无噪声编码比特映射成 BPSK LLR 后调用接收链。

    仅用于测试/演示：``0 -> +amplitude``，``1 -> -amplitude``。
    真正的 PUSCH 接收应把解调器输出的 LLR 直接传给
    ``decode_transport_block``。
    """
    bits = np.asarray(coded_bits, dtype=np.uint8).reshape(-1)
    if bits.size and np.any(bits > 1):
        raise ValueError("coded_bits 只能包含 0/1")
    llr = np.where(bits == 0, float(amplitude), -float(amplitude))
    return decode_transport_block(llr, config, plan=plan)
