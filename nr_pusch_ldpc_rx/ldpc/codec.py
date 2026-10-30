"""5G NR LDPC 编码与分层归一化最小和（Layered NMS）译码。

在包内的位置
------------
* :mod:`.tables`    -- TS 38.212 表 5.3.2-2/3 的原始表项；
* :mod:`.base_graph` -- 把表项提升为 H 矩阵并提供 ``H*c^T`` 校验；
* 本模块            -- 编码器与译码器，只依赖基图的“按块移位”操作。

码字结构
--------
完整码字 ``c = [s_0 ... s_{Kb-1}, p_0 ... p_{Mb-1}]``，每个元素是长度
``Zc`` 的块（BG1 的 ``Kb=22``、``Mb=46``；BG2 的 ``Kb=10``、``Mb=42``）。
因此 ``N_full = 68*Zc``（BG1）或 ``52*Zc``（BG2）。

.. note::

   这里的 ``Kb`` 是**编码器的系统列数**（22 或 10）。TS 38.212 5.2.2
   码块分割阶段的 ``Kb=6/8/9/10`` 只用于选择 ``Zc``，不改变编码器输入
   长度；BG2 小 TB 时多出来的系统列全部是 filler。

编码
----
前 4 个校验块利用基图前 4 行的双对角结构直接解出，其余每个校验块在
自己的行上有一个移位为 0 的单位阵，可以逐行递推。推导见 :func:`encode`。

译码
----
采用分层归一化最小和（Layered NMS）：

* ``L`` 保存变量节点的后验 LLR（信道 LLR + 所有校验节点消息）；
* 逐行（layer）先减去该行旧消息，做 check-node min-sum 更新，再加回；
* 校验节点外信息使用“除自身外”的最小值和次小值，乘归一化因子 ``alpha``；
* 每个完整迭代后用 ``H*c^T=0`` 检查硬判决；启用早停时满足即退出。

LLR 符号约定：**正数倾向比特 0，负数倾向比特 1**。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base_graph import base_graph_for_z, hard_syndrome_ok, shift
from ..constants import BG1_KB, BG2_KB, Z_TO_ILS, full_codeword_length


@dataclass(frozen=True)
class LDPCDecodeResult:
    """LDPC 译码结果。

    Attributes
    ----------
    info_bits:
        前 ``K = 22*Zc``（BG1）或 ``10*Zc``（BG2）个系统比特
        （包含 filler 位置，调用方需按计划去掉）。
    hard_bits:
        打孔前的完整硬判决码字，长度 ``N_full``。
    iterations:
        实际执行的迭代次数（1..max_iter）。
    converged:
        是否满足全部校验方程。启用早停时为“首次满足的迭代”，否则为
        最后一次迭代（通常是 ``max_iter``）的检查结果。
    """

    info_bits: np.ndarray
    hard_bits: np.ndarray
    iterations: int
    converged: bool


def _info_syndrome(
    info_blocks: np.ndarray,
    bg: np.ndarray,
    row: int,
    kb: int,
) -> np.ndarray:
    """计算基图第 ``row`` 行中“信息列部分”的异或和。

    ``info_blocks[j]`` 是第 j 个系统块（长度 Zc）。返回
    ``sum_j P_{row,j} * info_blocks[j]``，其中加法为 GF(2) 异或。
    这是编码方程里与校验块无关的那一部分，记为 ``A_row * s``。
    """
    acc = np.zeros(info_blocks.shape[1], dtype=np.uint8)
    for j in range(kb):
        s = int(bg[row, j])
        if s >= 0:  # -1 表示该位置是零矩阵，跳过
            acc ^= shift(info_blocks[j], s)
    return acc


def encode(info_bits: np.ndarray, bgn: int, z: int) -> np.ndarray:
    """执行 TS 38.212 5.3.2 的 LDPC 编码（不含打孔）。

    Parameters
    ----------
    info_bits:
        长度必须为 ``22*Zc``（BG1）或 ``10*Zc``（BG2）的一维 0/1
        序列。协议中的 filler（NULL）在
        送入编码器时应置 0；本函数也接受 -1 并自动按 0 处理。
    bgn:
        LDPC base graph（1 或 2）。
    z:
        lifting size Zc，必须属于表 5.3.2-1。

    Returns
    -------
    numpy.ndarray
        长度 ``N_full = (68 或 52)*Zc`` 的完整码字 uint8 数组。
        前 ``2*Zc`` 个系统比特在后续速率匹配中被 punctured。

    Notes
    -----
    **为什么需要特殊处理前 4 行？** 基图其余行都有一个移位为 0 的对角
    校验块 ``(i, Kb+i)``，可以直接解出 ``p_i``；但前 4 行的校验块彼此
    纠缠，没有这种“本行唯一未知”的结构。TS 38.212 的表 5.3.2-2/3 特意
    把这 4 行设计成双对角形式，才能线性时间编码。

    对 BG1（``P_A`` 在第 0 列校验块，``P_B`` 在第 1 行）::

        row0: A0*s + P_A*p0 + p1           = 0
        row1: A1*s + P_B*p0 + p1 + p2      = 0
        row2: A2*s +            p2 + p3     = 0
        row3: A3*s + P_A*p0          + p3  = 0

    对 BG2（``P_B`` 在第 2 行）::

        row0: A0*s + P_A*p0 + p1           = 0
        row1: A1*s +          p1 + p2      = 0
        row2: A2*s + P_B*p0 + p2 + p3      = 0
        row3: A3*s + P_A*p0          + p3  = 0

    其中 ``A_r*s`` 是第 r 行的信息列异或和（:func:`_info_syndrome`）。

    把 4 个方程在 GF(2) 上相加：``P_A*p0`` 出现两次、``p1..p3`` 各出现
    两次，全部相互抵消，只剩::

        P_B*p0 = A0*s + A1*s + A2*s + A3*s

    于是先解 ``p0``，再回代：

    * ``p1 = A0*s + P_A*p0``（row0）
    * ``p3 = A3*s + P_A*p0``（row3）
    * BG1: ``p2 = A1*s + P_B*p0 + p1``（row1）
    * BG2: ``p2 = A1*s + p1``（row1）

    ``P_B^{-1}`` 的实现：本模块把 ``P^s`` 定义为 ``np.roll(x, -s)``，所以
    求逆就是 ``np.roll(x, +s)``，即下面 ``p0 = np.roll(..., pb)``。

    其余行 ``i>=4`` 的对角块 ``(i, Kb+i)`` 移位为 0，本行除它以外的
    异或和恰好就是 ``p_i``（GF(2) 下移项不变号）。
    """
    if bgn not in (1, 2):
        raise ValueError("bgn 必须为 1 或 2")
    if z not in Z_TO_ILS:
        raise ValueError(f"Zc={z} 不是合法 lifting size")

    # 5G NR 的 LDPC 编码器固定使用 22 个（BG1）或 10 个（BG2）
    # 系统列。分割阶段的 Kb=6/8/9/10 只用于选 Zc，不改变这里的 K。
    kb = BG1_KB if bgn == 1 else BG2_KB
    bg = base_graph_for_z(bgn, z)
    expected_k = kb * z
    info = np.asarray(info_bits, dtype=np.int64).reshape(-1)
    if info.size != expected_k:
        raise ValueError(f"info_bits 长度应为 {expected_k}，收到 {info.size}")
    # filler 在协议中是 NULL；编码时按 0 参与运算。
    info = np.where(info < 0, 0, info).astype(np.uint8)
    if np.any(info > 1):
        raise ValueError("info_bits 只能包含 0/1（或 -1 表示 filler）")

    s = info.reshape(kb, z)               # (Kb, Zc)：Kb 个系统块
    mb = bg.shape[0]                      # 校验块数（46 或 42）
    p = [np.zeros(z, dtype=np.uint8) for _ in range(mb)]

    # --- 前 4 个校验块 -------------------------------------------------
    # 两个基图都把 P_A 放在 (0, Kb)，即第 0 行、第一个校验列。
    pa = int(bg[0, kb])
    # P_B 的位置随基图不同：BG1 在 row1，BG2 在 row2。
    pb_row = 1 if bgn == 1 else 2
    pb = int(bg[pb_row, kb])
    if pa < 0 or pb < 0:
        raise AssertionError("基图前 4 行结构不符合 TS 38.212 预期")

    a = [_info_syndrome(s, bg, r, kb) for r in range(4)]
    # P_B * p0 = a0+a1+a2+a3。P^pb 的逆是 np.roll(x, +pb)（见模块说明）。
    p0 = np.roll(a[0] ^ a[1] ^ a[2] ^ a[3], pb)
    # 回代：row0 给出 p1，row3 给出 p3；p2 的表达式两个基图不同。
    p1 = a[0] ^ shift(p0, pa)
    p3 = a[3] ^ shift(p0, pa)
    if bgn == 1:
        # BG1 row1: A1 + P_B*p0 + p1 + p2 = 0
        p2 = a[1] ^ shift(p0, pb) ^ p1
    else:
        # BG2 row1: A1 + p1 + p2 = 0
        p2 = a[1] ^ p1
    p[0], p[1], p[2], p[3] = p0, p1, p2, p3

    # --- 其余校验块：本行除对角块外的异或和 -------------------------
    for i in range(4, mb):
        acc = _info_syndrome(s, bg, i, kb)
        # 本行可能在 p0..p3 中还有若干非零块。
        for q in range(4):
            s_q = int(bg[i, kb + q])
            if s_q >= 0:
                acc ^= shift(p[q], s_q)
        # 对角块 (i, Kb+i) 的移位必须为 0，于是方程退化为 p_i = acc。
        # 这也是 TS 38.212 基图表能支持线性时间编码的关键性质。
        if int(bg[i, kb + i]) != 0:
            raise AssertionError(
                f"BG{bgn} 第 {i} 行对角校验块移位不是 0，编码公式不适用"
            )
        p[i] = acc

    codeword = np.concatenate([s.reshape(-1)] + p).astype(np.uint8, copy=False)
    if codeword.size != bg.shape[1] * z:
        raise AssertionError("编码输出长度与基图不一致")
    return codeword


def _check_node_update(
    u: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """归一化最小和校验节点更新（输入/输出均为校验节点坐标系）。

    Parameters
    ----------
    u:
        形状 ``(degree, Zc)`` 的变量到校验消息，已按各自的移位对齐。
    alpha:
        归一化因子，通常取 0.75~0.85。

    Returns
    -------
    numpy.ndarray
        与 ``u`` 同形状的校验到变量消息（未做反移位）。

    Notes
    -----
    最小和（min-sum）是 sum-product 的近似：对每个校验节点、每个提升
    位置 n，令 ``m1``、``m2`` 为 ``|u_e[n]|`` 的最小值和次小值，``s_e``
    为符号。第 e 条边收到的外信息幅度为

    * ``m2``：当 e 正好是幅度最小的那条边（去掉它后最小值变成次小值）；
    * ``m1``：其余边（去掉任意一条非最小边，最小值仍是 m1）。

    符号为“所有边符号的乘积再乘 ``s_e``”，等价于去掉第 e 条边自身的
    符号（因为 ``s_e * s_e = 1``）。乘以 ``alpha < 1`` 是归一化最小和，
    用于补偿 min-sum 相对 sum-product 的过估计，改善译码性能。
    """
    deg, z = u.shape
    absu = np.abs(u)
    idx_min = np.argmin(absu, axis=0)
    ar = np.arange(z)
    min1 = absu[idx_min, ar]

    # 次小值：把每列的最小元素临时置为 inf 后再取最小。
    # 若 deg==1，min2 会是 inf；实际基图每行度数 >= 2（除打孔列），
    # 且 decoder 只在完整行上调用，因此这里不会把 inf 传播到结果。
    tmp = absu.copy()
    tmp[idx_min, ar] = np.inf
    min2 = tmp.min(axis=0)

    signs = np.where(u < 0, -1.0, 1.0)
    sign_all = np.prod(signs, axis=0)

    # 最小边取次小值 m2，其余边取最小值 m1。
    mag = np.tile(min1, (deg, 1))
    mag[idx_min, ar] = min2
    return (alpha * mag) * (sign_all[None, :] * signs)


def decode(
    llr: np.ndarray,
    bgn: int,
    z: int,
    max_iter: int = 30,
    alpha: float = 0.8,
    early_stop: bool = True,
) -> LDPCDecodeResult:
    """LDPC 分层归一化最小和译码。

    Parameters
    ----------
    llr:
        长度 ``N_full = (68 或 52)*Zc`` 的 LLR。前 ``2*Zc`` 个系统比特
        被打孔，通常填 0（表示完全未知）；filler 位置应填一个很大的
        正 LLR（表示已知为 0）。
    bgn, z:
        base graph 与 lifting size。
    max_iter:
        最大迭代次数，必须 >= 1。
    alpha:
        归一化最小和因子，0 < alpha <= 1。
    early_stop:
        是否在每个完整迭代后用 H*c^T=0 提前终止。关闭时仍会跑满
        ``max_iter`` 次，并在最后一次检查收敛标志。

    Returns
    -------
    LDPCDecodeResult
        含系统比特、完整硬判决、实际迭代次数和收敛标志。

    Notes
    -----
    **分层（layered）调度**：与“先算完所有校验节点、再统一更新变量节点”
    的 flooding 调度不同，这里按基图行逐行更新。处理第 r 行时：

    1. 从后验 ``L`` 中减掉第 r 行上一轮的贡献 ``R[r]``，得到干净的
       变量到校验消息 ``v2c``（否则旧消息会被重复计算）；
    2. 把 ``v2c`` 旋转到校验节点坐标系，做 NMS 更新；
    3. 旋转回变量节点坐标系，写回 ``L = v2c + new`` 并保存 ``R[r]=new``。

    由于行是顺序处理的，本行更新后的消息会立即被后面的行看到，收敛
    通常比 flooding 快一倍左右，这也是 5G NR 接收机常用的调度方式。

    实现上 ``R[r]`` 保存的是**已反移位回变量节点坐标系**的消息，这样
    ``L[cols] - old`` 可以直接相减，代价是每行每轮要做两次 ``np.roll``。
    若把 ``R`` 存成校验坐标，可以省掉反移位，但 ``L`` 的更新会多一次
    坐标变换，整体复杂度相近。
    """
    if bgn not in (1, 2):
        raise ValueError("bgn 必须为 1 或 2")
    if z not in Z_TO_ILS:
        raise ValueError(f"Zc={z} 不是合法 lifting size")
    if max_iter < 1:
        raise ValueError("max_iter 必须 >= 1")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha 必须在 (0, 1] 内")

    n_full = full_codeword_length(bgn, z)
    llr = np.asarray(llr, dtype=np.float64).reshape(-1)
    if llr.size != n_full:
        raise ValueError(f"llr 长度应为 {n_full}，收到 {llr.size}")
    if not np.all(np.isfinite(llr)):
        raise ValueError("llr 不能包含 inf/nan；filler 请用有限的大正数")

    kb = BG1_KB if bgn == 1 else BG2_KB
    bg = base_graph_for_z(bgn, z)
    mb = bg.shape[0]

    # L 保存变量节点的后验 LLR；R[r] 保存第 r 行当前的校验到变量消息。
    # R[r] 的行对应本行的非零列（按列号升序），已经反移位回变量节点
    # 坐标系，便于与 L 直接相加减。
    L = llr.reshape(-1, z).copy()
    R = [
        np.zeros((int(np.count_nonzero(bg[r] >= 0)), z), dtype=np.float64)
        for r in range(mb)
    ]

    converged = False
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        for r in range(mb):
            cols = np.flatnonzero(bg[r] >= 0)   # 本行非零块对应的码字块下标
            shifts = bg[r, cols].astype(np.int64)
            old = R[r]

            # 变量到校验：先扣除本行旧的外信息。
            v2c = L[cols] - old
            # 对齐到校验节点坐标：P^s * x 对应 np.roll(x, -s)。
            u = np.empty_like(v2c)
            for e, s in enumerate(shifts):
                u[e] = np.roll(v2c[e], -int(s))

            msg = _check_node_update(u, alpha)

            # 反移位回变量节点坐标，并更新后验 LLR。
            new = np.empty_like(v2c)
            for e, s in enumerate(shifts):
                new[e] = np.roll(msg[e], int(s))
            L[cols] = v2c + new
            R[r] = new

        # 硬判决：LLR<0 -> 1。早停时每轮都检查；否则只在最后一轮检查，
        # 避免为“不早停”的调用白白增加校验子计算开销。
        if early_stop or it == max_iter:
            hard = (L.reshape(-1) < 0.0).astype(np.uint8)
            converged = hard_syndrome_ok(hard, bgn, z)
            if converged and early_stop:
                break

    # 即使未收敛也返回硬判决，上层用 CB/TB CRC 判断是否可用。
    return LDPCDecodeResult(
        info_bits=hard[: kb * z].copy(),
        hard_bits=hard,
        iterations=iterations,
        converged=converged,
    )
