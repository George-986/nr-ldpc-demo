"""LDPC 码层：TS 38.212 5.3.2 的基图、编码与译码。

分层关系::

    tables.py      协议表 5.3.2-2/3 的原始表项（纯数据）
        |
    base_graph.py  基图构造、lifting 移位、H*c^T 校验
        |
    codec.py       LDPC 编码器 + 分层归一化最小和译码器

对外只需 ``from nr_pusch_ldpc_rx.ldpc import encode, decode``；需要检查
码字是否满足校验方程时使用 :func:`hard_syndrome_ok` 或
:func:`base_graph_syndrome`。
"""

from .base_graph import (
    base_graph,
    base_graph_for_z,
    base_graph_syndrome,
    hard_syndrome_ok,
    lifting_set_index,
    shift,
)
from .codec import LDPCDecodeResult, decode, encode
from .tables import BASE_GRAPH_1_EDGES, BASE_GRAPH_2_EDGES

__all__ = [
    "BASE_GRAPH_1_EDGES",
    "BASE_GRAPH_2_EDGES",
    "LDPCDecodeResult",
    "base_graph",
    "base_graph_for_z",
    "base_graph_syndrome",
    "decode",
    "encode",
    "hard_syndrome_ok",
    "lifting_set_index",
    "shift",
]
