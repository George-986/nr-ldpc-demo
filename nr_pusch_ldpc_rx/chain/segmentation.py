"""TS 38.212 5.2.2：TB CRC、码块分割、CB CRC 与 TB 拼接。

在链路中的位置
--------------
发送方向::

    TB(A) --TB CRC--> b(B) --分割/CB CRC/filler--> C 个长度 K 的 CB
          --LDPC 编码--> 码字

接收方向::

    LDPC 系统比特 --CB CRC/去 filler/去 CB CRC--> 顺序拼接 --TB CRC--> TB

本模块只处理 CRC 与分割/拼接，不做 LDPC 编译码，因此可以单独测试
“TB -> CB -> TB”是否无损。

关键协议点（最容易写错的地方）
------------------------------
1. **Kb 只用于选择 Zc**：选出 Zc 后，LDPC 编码器输入长度固定为
   ``K = 22*Zc``（BG1）或 ``10*Zc``（BG2），不是 ``Kb*Zc``。BG2 小 TB
   时 ``Kb`` 可能只有 6/8/9，多出来的系统列全部是 filler。
2. **K' 向上取整**：``K' = ceil(B'/C)``。当 B 不能被 C 整除时，最后
   一个 CB 在计算 CB CRC 前补 0 到 K'；接收端拼接后只保留前 B 个比特，
   因此补零不会进入 TB。
3. **filler 在块尾**：filler 位于完整码字的 ``[K-F, K)``。码字前
   ``2*Zc`` 个系统比特会被打孔，所以到速率匹配的 d 域中，filler 位于
   ``[K-F-2*Zc, K-2*Zc)``——这个平移很容易漏掉。
4. **C=1 时没有 CB CRC**：``L_cb=0``；``C>1`` 时每个 CB 加 CRC24B，
   注意 C 的公式分母是 ``Kcb-24`` 而不是 ``Kcb``。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from ..constants import (
    BG1_KCB,
    BG2_KCB,
    CB_CRC_LEN,
    Z_LIST,
    codeword_length,
    full_codeword_length,
    kb_for_base_graph,
    select_base_graph,
    tb_crc_kind,
)
from .crc import crc_attach, crc_check, crc_width


@dataclass(frozen=True)
class TransportBlockPlan:
    """一个 TB 的全部分割参数（不可变）。

    这些字段就是 TS 38.212 5.2.2 中的预定义符号，后续速率匹配、LDPC
    编译码和接收端拼接都只依赖本对象，避免在多处重复推导。

    字段按推导顺序排列：``A/R`` -> ``bgn/tb_crc/b`` -> ``kcb/c/cb_crc_len``
    -> ``b_prime/k_prime`` -> ``kb/z/k/f``。派生量（N、N_full、filler
    位置等）用只读 property 提供，保证任何地方看到的都是同一份结果。
    """

    a: int                 # TB 信息比特数 A
    rate: float            # 调度目标码率 R
    bgn: int               # LDPC base graph（1 或 2）
    tb_crc: str            # "16" / "24A"
    tb_crc_len: int        # L
    b: int                 # A + L
    kcb: int               # 8448 (BG1) 或 3840 (BG2)
    c: int                 # 码块数 C
    cb_crc_len: int        # L_cb（0 或 24）
    b_prime: int           # B' = B + C*L_cb
    k_prime: int           # K' = ceil(B'/C)
    kb: int                # K_b
    z: int                 # Z_c
    k: int                 # K = 22*Z_c (BG1) 或 10*Z_c (BG2)
    f: int                 # F = K - K'

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------
    @property
    def n(self) -> int:
        """打孔后的编码比特数 N（进入循环缓存的长度）。"""
        return codeword_length(self.bgn, self.z)

    @property
    def n_full(self) -> int:
        """打孔前的 LDPC 码字长度 N_full。"""
        return full_codeword_length(self.bgn, self.z)

    @property
    def data_bits_per_cb(self) -> int:
        """每个 CB 中承载的 TB 数据比特数（不含 CB CRC）。"""
        return self.k_prime - self.cb_crc_len

    @property
    def total_data_capacity(self) -> int:
        """C 个 CB 的数据容量（可能比 B 多 0..C-1 个补零比特）。"""
        return self.c * self.data_bits_per_cb

    @property
    def zero_padding(self) -> int:
        """最后一个 CB 在计算 CRC 前补 0 的个数。"""
        return self.total_data_capacity - self.b

    @property
    def filler_start_d(self) -> int:
        """filler 在打孔后序列 d 中的起始下标。

        完整码字 c 的 filler 位于 ``[K-F, K)``；由于前 2*Zc 个系统比特
        被打孔，d 的下标要整体减去 2*Zc，故为 ``K-F-2*Zc``。
        """
        return self.k - self.f - 2 * self.z

    @property
    def filler_end_d(self) -> int:
        """filler 在 d 中的结束下标（开区间），等于 ``K-2*Zc``。"""
        return self.k - 2 * self.z

    @property
    def filler_mask(self) -> np.ndarray:
        """长度 N 的布尔数组，True 表示该 d 位置是 NULL filler。

        完整码字 c 中的 filler 在 ``[K-F, K)``；d 是 c 去掉前 ``2*Zc``
        个打孔系统比特后的序列，所以整体左移 ``2*Zc``。速率匹配在 d 域
        读取时必须跳过这些位置（协议中的 NULL），接收端则把它们当作
        “已知为 0”，注入很大的正 LLR。
        """
        mask = np.zeros(self.n, dtype=bool)
        if self.f:
            mask[self.filler_start_d:self.filler_end_d] = True
        return mask


def plan_transport_block(
    tb_size: int,
    target_rate: float,
    bgn: int | None = None,
) -> TransportBlockPlan:
    """根据 A 和 R 推导 TS 38.212 5.2.2 的全部码块参数。

    Parameters
    ----------
    tb_size:
        TB 信息比特数 A（不含 CRC）。
    target_rate:
        调度码率 R，用于默认的 base graph 选择。
    bgn:
        可选的强制基图编号（1 或 2）。正常链路应留空，让协议规则选择；
        测试中可强制某一 BG 来覆盖分支。

    Returns
    -------
    TransportBlockPlan
        包含 C、K'、Kb、Zc、K、F 等字段的不可变计划对象。

    Notes
    -----
    ``K' = ceil(B'/C)``。当 B 不能被 C 整除时，最后一个码块在 CB CRC
    之前补零到 K'，接收端拼接后只保留前 B 个比特，因此补零不会进入 TB。

    另一个容易出错的协议细节是：``Kb`` 只用于选择 Zc；选出 Zc 后，
    LDPC 编码器输入长度固定为 ``K=22*Zc``（BG1）或 ``K=10*Zc``（BG2），
    不是 ``Kb*Zc``。BG2 小 TB 时多出来的系统列全部作为 filler。

    示例：``A=1000, R=0.3`` 时 ``L=16``、``B=1016``、``C=1``、
    ``K'=1016``、``Kb=10``、``Zc=104``、``K=1040``、``F=24``。
    """
    if not isinstance(tb_size, int) or isinstance(tb_size, bool):
        raise TypeError("tb_size 必须是 int")
    if tb_size <= 0:
        raise ValueError("tb_size 必须大于 0")
    if not (0.0 < target_rate < 1.0):
        raise ValueError("target_rate 必须在 (0, 1) 内")
    if bgn is None:
        bgn = select_base_graph(tb_size, target_rate)
    if bgn not in (1, 2):
        raise ValueError("bgn 必须为 1 或 2")

    kind = tb_crc_kind(tb_size)
    tb_crc_len = crc_width(kind)
    b = tb_size + tb_crc_len
    kcb = BG1_KCB if bgn == 1 else BG2_KCB

    if b <= kcb:
        c = 1
        cb_crc_len = 0
        b_prime = b
    else:
        cb_crc_len = CB_CRC_LEN
        # C = ceil(B / (Kcb - L_cb))，注意分母是 Kcb-24 而不是 Kcb。
        c = ceil(b / (kcb - cb_crc_len))
        b_prime = b + c * cb_crc_len

    k_prime = ceil(b_prime / c)  # K'
    kb = kb_for_base_graph(bgn, b)

    # 找表 5.3.2-1 中最小的 Z，使 Kb*Z >= K'。
    candidates = [z for z in Z_LIST if kb * z >= k_prime]
    if not candidates:
        raise ValueError(
            f"无法为 K'={k_prime}、Kb={kb} 找到合法 Zc（超过最大 384）"
        )
    z = candidates[0]
    # 关键细节：TS 38.212 5.2.2 在选出 Zc 后，固定令
    # K = 22*Zc (BG1) 或 10*Zc (BG2)，而不是 Kb*Zc。
    # BG2 的 Kb 只在“选 Zc”时使用；当 B 较小时 Kb 可能为 6/8/9，
    # 但 LDPC 编码器输入仍然是 10*Zc，多出的列全部是 filler。
    k = (22 if bgn == 1 else 10) * z
    f = k - k_prime

    if k < k_prime:
        raise AssertionError("内部错误：K 必须不小于 K'")
    if f < 0:
        raise AssertionError("内部错误：filler 数不能为负")
    if c > 1:
        # 每个 CB 长度 K' 不超过 Kcb；这是分割公式应保证的性质。
        if k_prime > kcb:
            raise AssertionError("内部错误：K' 超过 Kcb")

    plan = TransportBlockPlan(
        a=tb_size,
        rate=target_rate,
        bgn=bgn,
        tb_crc=kind,
        tb_crc_len=tb_crc_len,
        b=b,
        kcb=kcb,
        c=c,
        cb_crc_len=cb_crc_len,
        b_prime=b_prime,
        k_prime=k_prime,
        kb=kb,
        z=z,
        k=k,
        f=f,
    )
    # 数据容量应覆盖 B，且最多多出 C-1 个补零比特。
    if not (0 <= plan.zero_padding < c):
        raise AssertionError("内部错误：补零比特数不在 [0, C) 内")
    # filler 必须落在打孔后 d 的 [0, N) 内。
    if not (0 <= plan.filler_start_d <= plan.filler_end_d <= plan.n):
        raise AssertionError("内部错误：filler 位置越界")
    return plan


def make_code_blocks(
    tb_bits: np.ndarray,
    plan: TransportBlockPlan,
) -> np.ndarray:
    """把 TB 分割成 C 个长度为 K 的 LDPC 信息块（含 filler=0）。

    Parameters
    ----------
    tb_bits:
        长度必须等于 ``plan.a`` 的一维 0/1 序列。
    plan:
        ``plan_transport_block`` 的结果。

    Returns
    -------
    numpy.ndarray
        形状 ``(C, K)`` 的 uint8 数组。每个 CB 的前 ``K'`` 位是
        ``数据 + (可选)CRC24B``，最后 ``F`` 位是 filler（协议记为 NULL，
        编码时按 0 处理）。
    """
    tb = np.asarray(tb_bits, dtype=np.uint8)
    if tb.ndim != 1 or tb.size != plan.a:
        raise ValueError(f"TB 长度应为 {plan.a}，收到 shape={tb.shape}")
    if tb.size and np.any(tb > 1):
        raise ValueError("TB 只能包含 0/1")

    # 1) 附加 TB CRC，得到长度 B 的 b。
    b = crc_attach(tb, plan.tb_crc)
    if b.size != plan.b:
        raise AssertionError("TB CRC 长度与计划不一致")

    # 2) 需要时把 b 补零到 C*K'，再均分。补零只可能出现在最后一个 CB，
    #    而且补零发生在附加 CB CRC 之前，接收端会连同 CRC 一起校验，
    #    最后再按 B 截断，不会把补零带进 TB。
    capacity = plan.total_data_capacity
    if capacity < b.size:
        raise AssertionError("CB 数据容量不足以容纳 B")
    b_padded = np.concatenate(
        (b, np.zeros(capacity - b.size, dtype=np.uint8))
    )

    cbs = np.zeros((plan.c, plan.k), dtype=np.uint8)
    for r in range(plan.c):
        start = r * plan.data_bits_per_cb
        data = b_padded[start:start + plan.data_bits_per_cb]
        if plan.cb_crc_len:
            block = crc_attach(data, "24B")
        else:
            block = data
        if block.size != plan.k_prime:
            raise AssertionError("CB 长度与 K' 不一致")
        # 3) filler 在协议中是 NULL；送入 LDPC 编码器前置 0。
        cbs[r, :plan.k_prime] = block
    return cbs


@dataclass(frozen=True)
class CodeBlockCheck:
    """一个 CB 的 CRC 检查结果，便于上层记录/调试。"""

    index: int
    crc_present: bool
    crc_ok: bool


def desegment_code_blocks(
    decoded_cbs: np.ndarray,
    plan: TransportBlockPlan,
) -> tuple[np.ndarray, list[CodeBlockCheck]]:
    """接收端：CB CRC 校验、去 filler、去 CB CRC、顺序拼接为 TB+CRC。

    Parameters
    ----------
    decoded_cbs:
        LDPC 译码输出的信息比特，形状 ``(C, K)``（filler 位置也包含在内）。
    plan:
        与发送端一致的 ``TransportBlockPlan``。

    Returns
    -------
    b_hat:
        长度 B 的“TB + TB CRC”比特序列，仍保留 TB CRC，供上层做最终校验。
    checks:
        每个 CB 的 CRC 检查记录。C=1 时没有 CB CRC，``crc_present=False``
        且 ``crc_ok=True``。

    Notes
    -----
    CB CRC 失败时仍然返回拼接结果，由调用方根据 ``checks`` 和 TB CRC
    决定是否丢弃整个 TB；这比在第一个错误处抛异常更适合链路级仿真。
    """
    cbs = np.asarray(decoded_cbs, dtype=np.uint8)
    if cbs.shape != (plan.c, plan.k):
        raise ValueError(f"decoded_cbs 形状应为 {(plan.c, plan.k)}，收到 {cbs.shape}")

    checks: list[CodeBlockCheck] = []
    pieces: list[np.ndarray] = []
    for r in range(plan.c):
        block = cbs[r, :plan.k_prime]
        if plan.cb_crc_len:
            crc_ok = bool(crc_check(block, "24B"))
            info = block[:-plan.cb_crc_len]
        else:
            crc_ok = True
            info = block
        checks.append(
            CodeBlockCheck(
                index=r,
                crc_present=bool(plan.cb_crc_len),
                crc_ok=crc_ok,
            )
        )
        pieces.append(info)

    b_hat = np.concatenate(pieces)
    # 去掉分割时为了对齐而补的 0，恢复严格的 B 比特。补零只可能出现在
    # 最后一个 CB 的尾部，所以直接截断前 B 个比特即可。
    if b_hat.size < plan.b:
        raise AssertionError("拼接后的比特数小于 B")
    return b_hat[:plan.b], checks


def check_transport_block_crc(
    b_hat: np.ndarray,
    plan: TransportBlockPlan,
) -> tuple[np.ndarray, bool]:
    """对拼接结果做 TB CRC 校验并剥离 CRC。

    Parameters
    ----------
    b_hat:
        长度 B 的“TB + TB CRC”序列。
    plan:
        TB 计划，提供 CRC 类型。

    Returns
    -------
    tb_hat:
        长度 A 的 TB 信息比特（即使 CRC 失败也返回，便于统计误码）。
    ok:
        TB CRC 是否通过。
    """
    arr = np.asarray(b_hat, dtype=np.uint8)
    if arr.size != plan.b:
        raise ValueError(f"b_hat 长度应为 {plan.b}，收到 {arr.size}")
    ok = bool(crc_check(arr, plan.tb_crc))
    return arr[:plan.a].copy(), ok


def tb_crc_ok(tb_bits: np.ndarray, kind: str) -> bool:
    """直接校验一个“TB + CRC”块（辅助函数，不依赖 plan）。"""
    return bool(crc_check(tb_bits, kind))


