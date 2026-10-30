"""传输链处理级：CRC、码块分割、速率匹配、加扰。

这一层按 TS 38.212/38.211 的处理顺序划分模块，每个模块只做一件事，
既被 :mod:`nr_pusch_ldpc_rx.link` 的收发链调用，也可以单独做单元测试：

====================  ==================================================
模块                  职责（规范条款）
====================  ==================================================
:mod:`.crc`           CRC16/24A/24B 计算、附加、校验（38.212 5.1）
:mod:`.segmentation`  TB CRC、CB 分割、CB CRC、filler、TB 拼接（5.2.2）
:mod:`.rate_matching` E_r、k0、循环缓存、比特交织（5.4.2）
:mod:`.scrambling`    Gold 序列、c_init、加扰/解扰（38.211 5.2.1/6.3.1.1）
====================  ==================================================
"""

from .crc import (
    CRC_POLYNOMIALS,
    crc_attach,
    crc_bits,
    crc_check,
    crc_remainder,
    crc_strip,
    crc_width,
)
from .rate_matching import (
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
from .scrambling import (
    GOLD_NC,
    GOLD_SEQUENCE_LEN,
    PUSCHScramblingConfig,
    descramble_llr,
    descramble_pusch_llr,
    gold_sequence,
    pusch_scrambling_c_init,
    scramble_bits,
)
from .segmentation import (
    CodeBlockCheck,
    TransportBlockPlan,
    check_transport_block_crc,
    desegment_code_blocks,
    make_code_blocks,
    plan_transport_block,
    tb_crc_ok,
)

__all__ = [
    "CRC_POLYNOMIALS",
    "CodeBlockCheck",
    "FILLER_LLR",
    "GOLD_NC",
    "GOLD_SEQUENCE_LEN",
    "PUSCHScramblingConfig",
    "TransportBlockPlan",
    "bit_deinterleave",
    "bit_interleave",
    "check_transport_block_crc",
    "coded_bit_lengths",
    "crc_attach",
    "crc_bits",
    "crc_check",
    "crc_remainder",
    "crc_strip",
    "crc_width",
    "descramble_llr",
    "descramble_pusch_llr",
    "desegment_code_blocks",
    "gold_sequence",
    "make_code_blocks",
    "plan_transport_block",
    "pusch_scrambling_c_init",
    "rate_match",
    "rate_match_code_blocks",
    "rate_recover",
    "rate_recover_code_blocks",
    "scramble_bits",
    "starting_position",
    "tb_crc_ok",
]
