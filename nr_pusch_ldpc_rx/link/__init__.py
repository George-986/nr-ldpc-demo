"""收发链路层：发送端参考链路与接收端主入口。

* :mod:`.transmitter` -- TB CRC -> 分割 -> LDPC -> 速率匹配 -> 加扰；
* :mod:`.receiver`    -- 解扰 -> 解交织 -> 解速率匹配 -> LDPC -> CRC -> 拼接。

发送端存在的目的是给接收端提供逐比特可核对的测试向量；生产接收机可以
只使用 :mod:`.receiver`。
"""

from .receiver import (
    CodeBlockResult,
    PUSCHReceiverConfig,
    TransportBlockResult,
    decode_transport_block,
    decode_transport_block_from_bits,
)
from .transmitter import EncodedTransportBlock, encode_transport_block

__all__ = [
    "CodeBlockResult",
    "EncodedTransportBlock",
    "PUSCHReceiverConfig",
    "TransportBlockResult",
    "decode_transport_block",
    "decode_transport_block_from_bits",
    "encode_transport_block",
]
