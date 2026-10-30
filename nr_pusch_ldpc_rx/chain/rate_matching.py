"""TS 38.212 5.4.2：LDPC 速率匹配（bit selection + bit interleaving）及逆过程。

在链路中的位置
--------------
发送端每个码块执行::

    LDPC 码字 c --打孔 2*Zc--> d(N) --bit selection--> e(E_r)
                --bit interleaving--> f(E_r)

接收端顺序相反::

    LLR f --解交织--> e --循环缓存软合并--> d(N) --补打孔位--> LDPC 输入

本模块只做“选择/交织”及其逆过程，不涉及 LDPC 译码；但接收端返回值
的坐标已经是 LDPC 译码器需要的 d 域，前面补 ``2*Zc`` 个 0 LLR 即可。

循环缓存与 NULL
---------------
打孔后的码字 ``d`` 放进长度 ``N_cb`` 的循环缓存（PUSCH 未配置 LBRM 时
``N_cb=N``）。读取从 ``k0`` 开始绕圈，**跳过 filler（协议中的 NULL）**；
如果要读的比特数超过非 NULL 的缓存长度，就绕回开头重复读取。接收端
把落在同一缓存地址的多个 LLR 直接相加——这是独立观测下的最优软合并。

filler 在 d 域的下标是 ``[K-F-2*Zc, K-2*Zc)``（filler 本来在完整码字
的块尾，打孔后整体左移 ``2*Zc``），由 ``TransportBlockPlan.filler_mask``
提供；本模块不自己推导这个位置。

协议符号
--------
``N``    : 打孔后的码字长度，66*Zc（BG1）或 50*Zc（BG2）；
``N_cb`` : 循环缓存长度，PUSCH 未配置 LBRM 时等于 N；
``E_r``  : 第 r 个码块速率匹配后的比特数；
``G``    : 整个 TB 的编码比特总数；
``Qm``   : 调制阶数；
``N_L``  : 传输层数；
``rv``   : 冗余版本 0..3；
``k0``   : 循环缓存读取起点。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..constants import Z_TO_ILS, codeword_length

# filler 是协议中的 NULL，译码器需要把它当作“已知 0”。这里用有限大值，
# 避免 inf 在 min-sum 中产生 inf-inf 或 nan。
FILLER_LLR: float = 1.0e4


def bit_interleave(bits: Iterable[int], qm: int) -> np.ndarray:
    """TS 38.212 5.4.2.2 的比特交织。

    对 ``E = Qm * L`` 个比特，按列写入 ``Qm x L`` 矩阵再按行读出::

        f[i + Qm*j] = e[j + L*i],  0<=i<Qm, 0<=j<L

    这等价于 ``e.reshape(Qm, L).ravel(order="F")``。

    直观理解：把 ``e`` 按行填进一个 ``Qm x L`` 矩阵，再按列读出，使
    原本相邻的 ``Qm`` 个编码比特被分散到不同调制符号的相同比特位置上，
    避免某个符号的深衰落/干扰集中打在一段连续编码比特上。
    """
    arr = np.asarray(list(bits))
    if qm < 1:
        raise ValueError("Qm 必须 >= 1")
    if arr.size % qm:
        raise ValueError(f"比特数 {arr.size} 不是 Qm={qm} 的整数倍")
    if qm == 1:
        return arr.copy()
    return arr.reshape(qm, arr.size // qm).ravel(order="F")


def bit_deinterleave(llr: Iterable[float], qm: int) -> np.ndarray:
    """比特交织的逆过程。

    发送端把 ``Qm x L`` 矩阵按列展开，接收端先按列还原矩阵，再按行展开。
    LLR 与比特一样是逐位置对应的，所以直接对软值做同样的置换即可；
    该操作是纯置换，不改变信息量。
    """
    arr = np.asarray(list(llr))
    if qm < 1:
        raise ValueError("Qm 必须 >= 1")
    if arr.size % qm:
        raise ValueError(f"LLR 数 {arr.size} 不是 Qm={qm} 的整数倍")
    if qm == 1:
        return arr.copy()
    return arr.reshape(qm, arr.size // qm, order="F").ravel(order="C")


def coded_bit_lengths(g: int, c: int, qm: int, n_layers: int = 1) -> np.ndarray:
    """按 TS 38.212 5.4.2.1 计算每个码块的 E_r。

    令 ``Gp = G/(N_L*Qm)``，则前 ``C - (Gp mod C)`` 个码块取
    ``N_L*Qm*floor(Gp/C)``，其余码块取 ``N_L*Qm*ceil(Gp/C)``。
    这样所有 E_r 之和严格等于 G，且各码块长度最多相差一个 Qm*N_L。

    实现上先算 ``base, rem = divmod(Gp, C)``，前 ``C-rem`` 块为 ``base``
    个单位，最后 ``rem`` 块多 1 个单位（再乘 ``Qm*N_L``）。余数放在
    后面而不是前面，是 TS 38.212 5.4.2.1 的原文顺序，接收端必须与
    发送端一致；否则多码块场景下 E_r 切分会整体错位。
    """
    if g <= 0:
        raise ValueError("G 必须大于 0")
    if c <= 0:
        raise ValueError("C 必须大于 0")
    if qm < 1 or n_layers < 1:
        raise ValueError("Qm 和 N_L 必须 >= 1")
    unit = qm * n_layers
    if g % unit:
        raise ValueError(f"G={g} 必须是 Qm*N_L={unit} 的整数倍")

    gp = g // unit
    base, rem = divmod(gp, c)
    lengths = np.full(c, base * unit, dtype=np.int64)
    if rem:
        lengths[c - rem:] = (base + 1) * unit
    if int(lengths.sum()) != g:
        raise AssertionError("E_r 之和不等于 G")
    return lengths


def starting_position(bgn: int, z: int, rv: int, n_cb: int | None = None) -> int:
    """返回 TS 38.212 Table 5.4.2.1-2 的循环缓存起点 k0。

    BG1 的分子为 17/33/56（rv=1/2/3），分母为 66*Zc；
    BG2 的分子为 13/25/43，分母为 50*Zc。公式为::

        k0 = floor(factor * N_cb / (den*Zc)) * Zc

    ``k0`` 一定是 ``Zc`` 的整数倍，这样读取起点刚好落在提升块的边界上，
    接收端才能按块对齐做循环移位。``rv=0`` 时直接返回 0（不乘任何因子）。
    """
    if bgn not in (1, 2):
        raise ValueError("bgn 必须为 1 或 2")
    if z not in Z_TO_ILS:
        raise ValueError(f"Zc={z} 不是合法 lifting size")
    if rv not in (0, 1, 2, 3):
        raise ValueError("rv 必须为 0/1/2/3")
    n = codeword_length(bgn, z)
    if n_cb is None:
        n_cb = n
    if not (0 < n_cb <= n):
        raise ValueError(f"N_cb 必须在 (0, {n}] 内")

    if rv == 0:
        return 0
    if bgn == 1:
        factor, den = {1: (17, 66), 2: (33, 66), 3: (56, 66)}[rv]
    else:
        factor, den = {1: (13, 50), 2: (25, 50), 3: (43, 50)}[rv]
    return (factor * n_cb) // (den * z) * z


def _validate_filler_mask(filler_mask: np.ndarray | None, n_cb: int) -> np.ndarray:
    """把 filler 掩码规范化成长度 n_cb 的 bool 数组。"""
    if filler_mask is None:
        return np.zeros(n_cb, dtype=bool)
    mask = np.asarray(filler_mask, dtype=bool)
    if mask.ndim != 1 or mask.size < n_cb:
        raise ValueError(f"filler_mask 至少需要 {n_cb} 个元素")
    return mask[:n_cb].copy()


def _selection_indices(
    n_cb: int,
    k0: int,
    filler_mask: np.ndarray,
    e_len: int,
) -> np.ndarray:
    """生成速率匹配读取的 d 下标序列（长度 e_len，允许循环重复）。

    步骤：

    1. ``order`` 是从 ``k0`` 开始绕缓存一圈的下标（自然取模）；
    2. 按 ``filler_mask`` 过滤，得到所有可读的非 NULL 地址 ``valid``；
    3. ``np.resize(valid, e_len)`` 把 ``valid`` 循环重复到恰好 ``e_len``。

    第 3 步同时实现了协议里的“绕回”和“重复传输”：当 ``E_r`` 大于
    可读比特数时，前面的比特会被再次选中，接收端据此做软合并。
    """
    if e_len < 0:
        raise ValueError("E 不能为负")
    order = (k0 + np.arange(n_cb, dtype=np.int64)) % n_cb
    valid = order[~filler_mask[order]]
    if valid.size == 0:
        raise ValueError("循环缓存中所有位置都是 filler，无法速率匹配")
    if e_len == 0:
        return np.empty(0, dtype=np.int64)
    return np.resize(valid, e_len)


def rate_match(
    codeword: Iterable[int],
    e_len: int,
    rv: int,
    qm: int,
    bgn: int,
    z: int,
    filler_mask: np.ndarray | None = None,
    n_cb: int | None = None,
) -> np.ndarray:
    """发送端速率匹配：bit selection + bit interleaving。

    Parameters
    ----------
    codeword:
        长度 N 的 d 序列（已经去掉前 2*Zc 个系统比特）。filler 位置的值
        会被跳过，传 0 即可。
    e_len:
        该码块输出长度 E_r。
    rv, qm, bgn, z:
        冗余版本、调制阶数、基图、lifting size。
    filler_mask:
        长度 N 的 bool 掩码，True 表示 filler；默认无 filler。掩码坐标
        是**打孔后的 d 域**（长度 N），不是完整码字域。
    n_cb:
        循环缓存长度，默认等于 N（PUSCH 未配置 LBRM 时）。

    Returns
    -------
    numpy.ndarray
        长度 E_r 的交织后比特序列 f。
    """
    n = codeword_length(bgn, z)
    d = np.asarray(list(codeword), dtype=np.uint8)
    if d.size != n:
        raise ValueError(f"codeword 长度应为 {n}，收到 {d.size}")
    if n_cb is None:
        n_cb = n
    if not (0 < n_cb <= n):
        raise ValueError(f"N_cb 必须在 (0, {n}] 内")
    mask = _validate_filler_mask(filler_mask, n_cb)
    k0 = starting_position(bgn, z, rv, n_cb)
    idx = _selection_indices(n_cb, k0, mask, e_len)
    selected = d[idx]
    return bit_interleave(selected, qm)


def rate_recover(
    llr: Iterable[float],
    rv: int,
    qm: int,
    bgn: int,
    z: int,
    filler_mask: np.ndarray | None = None,
    n_cb: int | None = None,
) -> np.ndarray:
    """接收端解速率匹配：解交织 + 循环缓存 LLR 合并。

    输入 LLR 长度就是 E_r。若某位置被重复传输（E_r 大于非 filler 的
    缓存长度），重复 LLR 直接相加，这是独立观测下的最优软合并。

    实现用 ``np.bincount(idx, weights=e)`` 一次完成“按地址累加”：``idx``
    中重复出现的地址自动求和，没有出现的地址计数为 0。

    Returns
    -------
    numpy.ndarray
        长度 N 的 d 域 LLR。filler 位置为 ``FILLER_LLR``（已知 0）；
        未传输且非 filler 的位置为 0（完全未知）。调用方随后在头部补
        ``2*Zc`` 个 0，得到长度 N_full 的 LDPC 译码输入。

    Notes
    -----
    ``n_cb < N`` 时循环缓存只覆盖 d 的前 ``n_cb`` 个位置，但缓存之外的
    filler 在协议里仍然是 NULL（已知 0），所以代码保留完整 filler 掩码
    并在扩容到 N 后单独注入 ``FILLER_LLR``。
    """
    n = codeword_length(bgn, z)
    if n_cb is None:
        n_cb = n
    if not (0 < n_cb <= n):
        raise ValueError(f"N_cb 必须在 (0, {n}] 内")
    mask = _validate_filler_mask(filler_mask, n_cb)
    # 保留完整 filler 掩码：有限缓存 n_cb<N 时，缓存之外的 filler 仍是
    # 协议已知的 0，译码器可以据此注入大正 LLR。
    full_mask = None
    if filler_mask is not None:
        candidate = np.asarray(filler_mask, dtype=bool).reshape(-1)
        if candidate.size >= n:
            full_mask = candidate[:n]
    e = np.asarray(list(llr), dtype=np.float64).reshape(-1)
    e = bit_deinterleave(e, qm)
    k0 = starting_position(bgn, z, rv, n_cb)
    idx = _selection_indices(n_cb, k0, mask, e.size)

    # np.bincount 正确累加重复下标；未出现的下标计数为 0。
    out = np.bincount(idx, weights=e, minlength=n_cb).astype(np.float64)
    # filler 是已知的 0；必须在扩容到 N 之前按 n_cb 长度的 mask 赋值。
    out[mask] = FILLER_LLR
    if out.size < n:
        out = np.pad(out, (0, n - out.size))
        if full_mask is not None:
            out[n_cb:][full_mask[n_cb:]] = FILLER_LLR
    return out


def rate_match_code_blocks(
    codewords: Iterable[Iterable[int]],
    e_lengths: Iterable[int],
    rv: int,
    qm: int,
    bgn: int,
    z: int,
    filler_mask: np.ndarray | None = None,
) -> np.ndarray:
    """对 C 个码块依次速率匹配并按 TS 38.212 5.5 拼接成 G 长序列。"""
    cws = list(codewords)
    es = list(e_lengths)
    if len(cws) != len(es):
        raise ValueError("码块数与 E_r 数不一致")
    parts = [
        rate_match(cw, int(e), rv, qm, bgn, z, filler_mask)
        for cw, e in zip(cws, es)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint8)


def rate_recover_code_blocks(
    llr: Iterable[float],
    e_lengths: Iterable[int],
    rv: int,
    qm: int,
    bgn: int,
    z: int,
    filler_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """把 G 长 LLR 按 E_r 切开，逐码块解速率匹配，返回 d 域 LLR 列表。"""
    arr = np.asarray(list(llr), dtype=np.float64).reshape(-1)
    es = [int(e) for e in e_lengths]
    if sum(es) != arr.size:
        raise ValueError(f"sum(E_r)={sum(es)} 与 LLR 长度 {arr.size} 不一致")
    out: list[np.ndarray] = []
    start = 0
    for e in es:
        out.append(
            rate_recover(arr[start:start + e], rv, qm, bgn, z, filler_mask)
        )
        start += e
    return out
