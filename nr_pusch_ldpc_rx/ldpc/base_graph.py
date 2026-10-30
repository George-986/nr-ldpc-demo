"""LDPC 基图查表与 lifting（提升）操作。

在包内的位置
------------
:mod:`~nr_pusch_ldpc_rx.ldpc.tables` 提供 TS 38.212 表 5.3.2-2/3 的原始
三元组；本模块把表项提升为实际的奇偶校验矩阵，并把“按块循环移位”封装
成向量操作；编码/译码算法在 :mod:`.codec` 中，不在这里。

lifting（提升）的含义
---------------------
TS 38.212 5.3.2 用基图 BG 和 lifting size Zc 生成完整的 H 矩阵：

1. 基图中的一个表项 ``V_{i,j}`` 替换为 ``Zc x Zc`` 的循环移位单位阵
   ``I(P_{i,j})``，其中 ``P_{i,j} = V_{i,j} mod Zc``；
2. 基图中未列出的位置（本实现记为 -1）是 ``Zc x Zc`` 零矩阵；
3. 因此 ``(46*Zc) x (68*Zc)``（BG1）的稀疏 H 只需用一个 46x68 的整数
   矩阵表示，编码/译码时对长度为 Zc 的块做异或和循环移位即可。

循环移位的方向
--------------
本实现固定 ``P^s`` 为“列下标加 s”，即 ``H[i, (i+s) mod Zc] = 1``；对
列向量 ``x`` 有 ``P^s x = np.roll(x, -s)``。这个约定必须同时用于编码、
译码和校验子计算。方向搞反时收发可能同时错位而在无噪声自测中“自洽”，
却与协议码字不一致，因此测试用独立的 ``H*c^T=0`` 校验约束方向。
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .tables import BASE_GRAPH_1_EDGES, BASE_GRAPH_2_EDGES
from ..constants import (
    BG1_COLS,
    BG1_ROWS,
    BG2_COLS,
    BG2_ROWS,
    Z_TO_ILS,
)


def lifting_set_index(z: int) -> int:
    """返回 lifting size ``z`` 对应的表 5.3.2-1 集合下标 i_LS。

    Parameters
    ----------
    z:
        合法 lifting size（2..384 中表 5.3.2-1 的离散取值）。

    Returns
    -------
    int
        i_LS，范围 0..7。
    """
    try:
        return Z_TO_ILS[z]
    except KeyError as exc:
        raise ValueError(f"Zc={z} 不是合法的 lifting size") from exc


@lru_cache(maxsize=None)
def _base_graph_cached(bgn: int, i_ls: int) -> np.ndarray:
    """构造并缓存指定基图/集合的基图矩阵（-1 表示零矩阵）。"""
    if bgn not in (1, 2):
        raise ValueError(f"bgn 必须为 1 或 2，收到 {bgn}")
    if not (0 <= i_ls <= 7):
        raise ValueError(f"i_LS 必须在 0..7，收到 {i_ls}")

    rows, cols = (BG1_ROWS, BG1_COLS) if bgn == 1 else (BG2_ROWS, BG2_COLS)
    edges = BASE_GRAPH_1_EDGES if bgn == 1 else BASE_GRAPH_2_EDGES

    # 用 -1 填充“全零矩阵”的位置；非负值表示循环移位量。
    bg = np.full((rows, cols), -1, dtype=np.int16)
    for i, j, shifts in edges:
        # 表值 V_{i,j} 的范围是 0..383，对不同的 Zc 可能大于 Zc；
        # 协议要求实际移位 P = V mod Zc。这里保留原始 V，由 shift()
        # 里的 np.roll 自动按 Zc 取模（np.roll 的移位量天然对长度取模），
        # 既省一次取模，也方便直接用表值核对协议。
        bg[i, j] = int(shifts[i_ls])
    bg.setflags(write=False)
    return bg


def base_graph(bgn: int, i_ls: int) -> np.ndarray:
    """返回 ``i_LS`` 对应的基图矩阵，形状 BG1=(46,68)、BG2=(42,52)。

    矩阵元素为 -1（全零块）或 0..383（循环移位量）。返回的数组是只读的，
    避免调用方意外修改缓存。
    """
    return _base_graph_cached(bgn, i_ls)


def base_graph_for_z(bgn: int, z: int) -> np.ndarray:
    """根据 Zc 自动查 i_LS 并返回基图。"""
    return base_graph(bgn, lifting_set_index(z))


def shift(x: np.ndarray, amount: int) -> np.ndarray:
    """按 TS 38.212 的循环移位方向计算 ``P^amount * x``。

    本实现把 TS 38.212 的 ``P^s`` 落实为
    ``H[i, (i+s) mod Zc] = 1``（i 为行号）。对长度为 Zc 的列向量 ``x``，
    与 ``P^s`` 相乘等价于 ``np.roll(x, -s)``：结果的第 n 个分量等于
    ``x[(n + s) mod Zc]``。这个方向必须与 H 矩阵的定义一致，否则编码和
    译码会同时错位而在无噪声测试中看似“自洽”，却不符合协议。

    Parameters
    ----------
    x:
        长度为 Zc 的一维数组（一个提升块）。
    amount:
        移位量 s；负数也允许，表示反方向移位。

    Returns
    -------
    numpy.ndarray
        移位后的数组（不修改输入）。
    """
    return np.roll(x, -int(amount))


def base_graph_syndrome(
    blocks: np.ndarray,
    bgn: int,
    z: int,
) -> np.ndarray:
    """用基图计算提升后码字的校验子（每行一个 Zc 长向量）。

    Parameters
    ----------
    blocks:
        形状 ``(N_full/Zc, Zc)`` 的码字块，按基图列顺序排列。
    bgn, z:
        基图编号和 lifting size。

    Returns
    -------
    numpy.ndarray
        形状 ``(BG_rows, Zc)``；全零表示满足 H*c^T=0。
    """
    bg = base_graph_for_z(bgn, z)
    n_blocks = blocks.shape[0]
    if n_blocks != bg.shape[1]:
        raise ValueError(
            f"码字块数 {n_blocks} 与基图列数 {bg.shape[1]} 不一致"
        )

    # 第 i 行校验方程：sum_j P_{i,j} * c_j = 0（GF(2) 加法即异或）。
    # blocks[j] 是第 j 个 Zc 长码字块，shift(blocks[j], P) 完成 P^{P} * c_j。
    syndrome = np.zeros((bg.shape[0], z), dtype=np.uint8)
    for i in range(bg.shape[0]):
        for j in np.flatnonzero(bg[i] >= 0):
            syndrome[i] ^= shift(blocks[j], int(bg[i, j]))
    return syndrome


def hard_syndrome_ok(bits: np.ndarray, bgn: int, z: int) -> bool:
    """判断硬判决码字是否满足所有校验方程（用于 LDPC 早停）。"""
    blocks = np.asarray(bits, dtype=np.uint8).reshape(-1, z)
    return not np.any(base_graph_syndrome(blocks, bgn, z))
