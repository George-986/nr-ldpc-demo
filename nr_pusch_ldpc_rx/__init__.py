"""NR PUSCH LDPC 收发链参考实现（Python / NumPy）。

本包按 3GPP TS 38.212 / TS 38.211 实现 PUSCH（UL-SCH）的
“TB CRC -> 码块分割 -> LDPC -> 速率匹配 -> 比特交织 -> 加扰”发送链与
严格互逆的接收链，只依赖 NumPy。

包内按功能分为四层，依赖方向自上而下::

    app/      配置解析、端到端运行、报告、CLI
    link/     发送端参考链路 / 接收端主入口
    chain/    CRC、分割、速率匹配、加扰（逐级处理）
    ldpc/     基图表、lifting、LDPC 编码与分层 NMS 译码
    constants.py  全层共用的协议常量与选择规则

最常用的入口
------------
``PUSCHReceiverConfig`` + ``decode_transport_block``:
    从解调后的 LLR 恢复 TB。
``encode_transport_block``:
    生成与接收端严格互逆的测试向量。
``plan_transport_block``:
    查看 A/R 对应的 BG、Kb、Zc、C、K、F 等协议参数。
``run_config``:
    读取 JSON 模板，直接运行整个发送/接收流程并返回报告（见
    ``app.pipeline`` 与 ``templates/``）。
"""

from .app import (
    CaseResult,
    PUSCHCase,
    RunReport,
    load_config,
    parse_case,
    parse_config,
    run_case,
    run_config,
)
from .chain import (
    FILLER_LLR,
    GOLD_NC,
    GOLD_SEQUENCE_LEN,
    CodeBlockCheck,
    PUSCHScramblingConfig,
    TransportBlockPlan,
    bit_deinterleave,
    bit_interleave,
    check_transport_block_crc,
    coded_bit_lengths,
    crc_attach,
    crc_bits,
    crc_check,
    crc_remainder,
    crc_strip,
    crc_width,
    descramble_llr,
    descramble_pusch_llr,
    desegment_code_blocks,
    gold_sequence,
    make_code_blocks,
    plan_transport_block,
    pusch_scrambling_c_init,
    rate_match,
    rate_match_code_blocks,
    rate_recover,
    rate_recover_code_blocks,
    scramble_bits,
    starting_position,
)
from .constants import (
    MODULATION_ORDER,
    Z_LIST,
    Z_SETS,
    Z_TO_ILS,
    modulation_order,
    select_base_graph,
)
from .ldpc import (
    LDPCDecodeResult,
    base_graph,
    base_graph_for_z,
    base_graph_syndrome,
    hard_syndrome_ok,
    lifting_set_index,
    shift,
)
from .ldpc import decode as ldpc_decode
from .ldpc import encode as ldpc_encode
from .link import (
    CodeBlockResult,
    EncodedTransportBlock,
    PUSCHReceiverConfig,
    TransportBlockResult,
    decode_transport_block,
    decode_transport_block_from_bits,
    encode_transport_block,
)

__version__ = "1.1.0"

__all__ = [
    # 常量与选择规则
    "MODULATION_ORDER",
    "Z_LIST",
    "Z_SETS",
    "Z_TO_ILS",
    "modulation_order",
    "select_base_graph",
    # CRC
    "crc_attach",
    "crc_bits",
    "crc_check",
    "crc_remainder",
    "crc_strip",
    "crc_width",
    # 码块分割
    "CodeBlockCheck",
    "TransportBlockPlan",
    "check_transport_block_crc",
    "desegment_code_blocks",
    "make_code_blocks",
    "plan_transport_block",
    # LDPC
    "LDPCDecodeResult",
    "base_graph",
    "base_graph_for_z",
    "base_graph_syndrome",
    "hard_syndrome_ok",
    "lifting_set_index",
    "ldpc_decode",
    "ldpc_encode",
    "shift",
    # 速率匹配
    "FILLER_LLR",
    "bit_deinterleave",
    "bit_interleave",
    "coded_bit_lengths",
    "rate_match",
    "rate_match_code_blocks",
    "rate_recover",
    "rate_recover_code_blocks",
    "starting_position",
    # 加扰
    "GOLD_NC",
    "GOLD_SEQUENCE_LEN",
    "PUSCHScramblingConfig",
    "descramble_llr",
    "descramble_pusch_llr",
    "gold_sequence",
    "pusch_scrambling_c_init",
    "scramble_bits",
    # 收发链路
    "CodeBlockResult",
    "EncodedTransportBlock",
    "PUSCHReceiverConfig",
    "TransportBlockResult",
    "decode_transport_block",
    "decode_transport_block_from_bits",
    "encode_transport_block",
    # 应用层
    "CaseResult",
    "PUSCHCase",
    "RunReport",
    "load_config",
    "parse_case",
    "parse_config",
    "run_case",
    "run_config",
]
