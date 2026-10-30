"""PUSCH 发送端参考链路：TB CRC -> 分割 -> LDPC -> 速率匹配 -> 加扰。

在包内的位置
------------
发送端把 :mod:`~nr_pusch_ldpc_rx.chain` 与 :mod:`~nr_pusch_ldpc_rx.ldpc`
的各级按协议顺序串起来，是接收端 :mod:`.receiver` 的严格逆过程。它存在
的目的，是生成逐比特可核对的测试向量（``EncodedTransportBlock`` 同时
保留每一级的中间结果），而不是实现完整的 PUSCH 发射机。

发送顺序（与 TS 38.212/38.211 一致）::

    TB CRC (5.1/6.2.1)
      -> CB 分割 + CB CRC + filler (5.2.2)
      -> LDPC 编码 + 打孔 2*Zc (5.3.2)
      -> 每 CB 速率匹配：E_r / k0 / 比特交织 (5.4.2)
      -> 码块拼接 (5.5)
      -> 加扰 (38.211 6.3.1.1)

调制、层映射、资源映射、DMRS 不在本模块范围内。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..chain.rate_matching import coded_bit_lengths, rate_match
from ..chain.scrambling import PUSCHScramblingConfig, scramble_bits
from ..chain.segmentation import (
    TransportBlockPlan,
    make_code_blocks,
    plan_transport_block,
)
from ..ldpc.codec import encode


@dataclass(frozen=True)
class EncodedTransportBlock:
    """一个 TB 经过编码、速率匹配和加扰后的完整中间结果。

    ``coded_bits`` 是真正送进调制器的比特；其余字段保留每一级的输出，
    便于测试逐级核对（例如用 ``full_codewords`` 独立校验 ``H*c^T=0``、
    用 ``e_lengths`` 核对 E_r 分配）。
    """

    tb_bits: np.ndarray              # 原始 TB（A）
    plan: TransportBlockPlan         # 分割/码率计划
    code_blocks: np.ndarray          # (C, K)，含 filler=0
    full_codewords: np.ndarray       # (C, N_full)，LDPC 编码输出
    punctured_codewords: np.ndarray  # (C, N)，已去掉前 2*Zc
    e_lengths: np.ndarray            # (C,)，每个 CB 的 E_r
    concatenated_bits: np.ndarray    # (G,)，码块拼接后、加扰前
    coded_bits: np.ndarray           # (G,)，加扰后（若启用）的最终比特
    scrambled: bool = False          # 是否执行了加扰

    @property
    def g(self) -> int:
        """编码比特总数 G。"""
        return int(self.coded_bits.size)


def encode_transport_block(
    tb_bits: np.ndarray,
    num_coded_bits: int,
    rv: int = 0,
    qm: int = 2,
    n_layers: int = 1,
    bgn: int | None = None,
    target_rate: float = 0.5,
    plan: TransportBlockPlan | None = None,
    scrambling: PUSCHScramblingConfig | None = None,
) -> EncodedTransportBlock:
    """执行完整的 PUSCH 发送端参考流程。

    Parameters
    ----------
    tb_bits:
        长度 A 的 0/1 序列。
    num_coded_bits:
        G，即速率匹配后整个 TB 的编码比特数（调制前）。
    rv:
        冗余版本 0..3。
    qm:
        调制阶数 1/2/4/6/8。
    n_layers:
        传输层数 N_L。
    bgn:
        可选的强制 base graph；默认按 A/R 选择。
    target_rate:
        调度码率 R，仅用于默认 BG 选择。
    plan:
        可选的已计算计划；提供时必须与 tb_bits 长度一致。
    scrambling:
        PUSCH 加扰配置。``None`` 表示不做加扰（非协议行为，仅用于
        单元测试）；正常链路应提供。加扰在码块拼接之后、调制之前进行。

    Returns
    -------
    EncodedTransportBlock
        含每个码块的完整码字、打孔码字、E_r 和最终 G 长比特序列。
    """
    tb = np.asarray(tb_bits, dtype=np.uint8).reshape(-1)
    if plan is None:
        plan = plan_transport_block(tb.size, target_rate, bgn=bgn)
    if plan.a != tb.size:
        raise ValueError(f"plan.a={plan.a} 与 TB 长度 {tb.size} 不一致")

    e_lengths = coded_bit_lengths(num_coded_bits, plan.c, qm, n_layers)
    cbs = make_code_blocks(tb, plan)

    # 每个码块独立完成 LDPC 编码 -> 打孔 -> 速率匹配；E_r 由整个 TB 的
    # G 统一分配（见 coded_bit_lengths），所以先算 e_lengths 再循环。
    full = np.zeros((plan.c, plan.n_full), dtype=np.uint8)
    punctured = np.zeros((plan.c, plan.n), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for r in range(plan.c):
        full[r] = encode(cbs[r], plan.bgn, plan.z)
        # TS 38.212 5.3.2：前 2*Zc 个系统比特打孔，不进入循环缓存。
        # 这 2*Zc 个比特在接收端没有观测，只能补 0 LLR（完全未知）。
        punctured[r] = full[r, 2 * plan.z:]
        parts.append(
            rate_match(
                punctured[r],
                int(e_lengths[r]),
                rv,
                qm,
                plan.bgn,
                plan.z,
                filler_mask=plan.filler_mask,
            )
        )

    concatenated = (
        np.concatenate(parts) if parts else np.empty(0, dtype=np.uint8)
    )
    if concatenated.size != num_coded_bits:
        raise AssertionError("速率匹配输出长度与 G 不一致")

    # TS 38.211 6.3.1.1：码块拼接之后、调制之前对整个码字加扰。
    # 注意作用域是拼接后的 G 长序列，而不是每个码块内部；接收端也必须
    # 在切分码块之前对整段 LLR 解扰。
    if scrambling is None:
        coded = concatenated.copy()
    else:
        coded = scramble_bits(concatenated, scrambling.sequence(concatenated.size))

    return EncodedTransportBlock(
        tb_bits=tb,
        plan=plan,
        code_blocks=cbs,
        full_codewords=full,
        punctured_codewords=punctured,
        e_lengths=e_lengths,
        concatenated_bits=concatenated,
        coded_bits=coded,
        scrambled=scrambling is not None,
    )
