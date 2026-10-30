"""3GPP NR PUSCH LDPC 链路使用的协议常量与选择规则。

本文件把 TS 38.212 中散落的预定义符号集中在一处，并给出严格的输入检查。
数据信道（UL-SCH/PUSCH）使用 LDPC，编码流程见 TS 38.212 第 5.2/5.3/5.4 节。

主要协议符号
------------
``A``      : 一个传输块（TB）的比特数（不含 TB CRC）；
``L``      : TB CRC 长度，A > 3824 时为 24（CRC24A），否则为 16（CRC16）；
``B``      : A + L，即附加 TB CRC 后的比特数；
``K_cb``   : 单个码块（CB）的最大长度，BG1 为 8448，BG2 为 3840；
``C``      : 码块个数；
``L_cb``   : 每个 CB 的 CRC 长度，C > 1 时为 24（CRC24B），否则为 0；
``B'``     : B + C * L_cb，即加入所有 CB CRC 后的总比特数；
``K'``     : 每个 CB 在插入 filler 前的长度，``ceil(B' / C)``；
``K_b``    : 基图信息列数，BG1 固定 22；BG2 由 B 决定，为 6/8/9/10；
``Z_c``    : lifting size，取自表 5.3.2-1；
``K``      : 22*Z_c（BG1）或 10*Z_c（BG2），即 LDPC 编码器输入长度
             （含 filler bits）。注意不是 K_b*Z_c；K_b 只用于选 Z_c；
``F``      : K - K'，每个 CB 末尾的 NULL/filler 比特数；
``N``      : 打孔后的编码比特数，BG1 为 66*Z_c，BG2 为 50*Z_c；
``N_full`` : 打孔前的 LDPC 码字长度，BG1 为 68*Z_c，BG2 为 52*Z_c。

注意：协议规定 LDPC 码字的前 2*Z_c 个系统比特被打孔，不进入循环缓存。

在包内的位置
------------
本模块位于依赖图最底层，不 import 包内任何其它模块；``ldpc``、``chain``、
``link``、``app`` 各层都从这里取协议常量和选择规则，保证“同一份协议
参数”只有一处定义。

命名提醒
--------
``BG2_KB``（值为 10）是**编码器的系统列数**，即 ``K = 10*Zc``；
``kb_for_bg2(b)`` 返回的才是 TS 38.212 5.2.2 中**用于选择 Zc** 的
``Kb = 6/8/9/10``。两者语义不同：前者决定编码器输入长度，后者只参与
``Kb*Zc >= K'`` 的查找，不能混用。
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# TS 38.212 Table 5.3.2-1: sets of LDPC lifting sizes Z
# ---------------------------------------------------------------------------
# 外层元组下标就是 i_LS（0..7）。同一个 Z 只会出现在一个集合中，
# 所以收到 Zc 后可以唯一确定 i_LS，进而从基图表中取对应的移位量。
Z_SETS: Final[tuple[tuple[int, ...], ...]] = (
    (2, 4, 8, 16, 32, 64, 128, 256),
    (3, 6, 12, 24, 48, 96, 192, 384),
    (5, 10, 20, 40, 80, 160, 320),
    (7, 14, 28, 56, 112, 224),
    (9, 18, 36, 72, 144, 288),
    (11, 22, 44, 88, 176, 352),
    (13, 26, 52, 104, 208),
    (15, 30, 60, 120, 240),
)

# 升序排列的全部合法 Zc；查找“最小满足条件的 Z”时直接遍历该元组。
Z_LIST: Final[tuple[int, ...]] = tuple(sorted(z for group in Z_SETS for z in group))

# Zc -> i_LS 的反查表。
Z_TO_ILS: Final[dict[int, int]] = {
    z: i_ls for i_ls, group in enumerate(Z_SETS) for z in group
}

# BG1/BG2 的基图尺寸（行数 x 列数）与信息列数。
BG1_ROWS: Final[int] = 46
BG1_COLS: Final[int] = 68
BG1_KB: Final[int] = 22
BG1_KCB: Final[int] = 8448

BG2_ROWS: Final[int] = 42
BG2_COLS: Final[int] = 52
# 注意：这里的 KB 是编码器系统列数 10（K=10*Zc），不是 5.2.2 里用于
# 选 Zc 的 Kb；后者的取值见 kb_for_bg2()。
BG2_KB: Final[int] = 10
BG2_KCB: Final[int] = 3840

# 打孔比特数：前 2*Zc 个系统比特不传输（TS 38.212 5.3.2）。
PUNCTURED_BLOCKS: Final[int] = 2

# 调制阶数 Qm（TS 38.211/38.212 的调制方式命名）。
# pi/2-BPSK 在编码链路上与 BPSK 一样都是 1 bit/symbol。
MODULATION_ORDER: Final[dict[str, int]] = {
    "pi/2-bpsk": 1,
    "bpsk": 1,
    "qpsk": 2,
    "16qam": 4,
    "64qam": 6,
    "256qam": 8,
}

# TS 38.212 5.2.2：TB 大于 3824 bit 时使用 CRC24A，否则使用 CRC16。
# 注意是严格大于：A=3824 时仍然用 CRC16。
TB_CRC24A_THRESHOLD: Final[int] = 3824

# TS 38.212 5.2.2：一个 TB 被分成多个 CB 时，每个 CB 附加 CRC24B。
CB_CRC_LEN: Final[int] = 24


def modulation_order(modulation: str | int) -> int:
    """把调制方式（或直接给出的 Qm）规范化为整数调制阶数。

    Parameters
    ----------
    modulation:
        ``"BPSK"``、``"QPSK"``、``"16QAM"``、``"64QAM"``、``"256QAM"``、
        ``"pi/2-BPSK"``（大小写不敏感），或直接给 1/2/4/6/8。

    Returns
    -------
    int
        Qm，即每个调制符号携带的编码比特数。
    """
    if isinstance(modulation, bool):  # bool 是 int 的子类，单独拒绝更清晰
        raise TypeError("modulation 不能是 bool")
    if isinstance(modulation, int):
        if modulation not in (1, 2, 4, 6, 8):
            raise ValueError(f"不支持的调制阶数 Qm={modulation}")
        return modulation
    if not isinstance(modulation, str):
        raise TypeError("modulation 必须是字符串或整数 Qm")
    key = modulation.strip().lower().replace(" ", "")
    if key not in MODULATION_ORDER:
        raise ValueError(
            f"不支持的调制方式 {modulation!r}，可选 "
            f"{sorted(MODULATION_ORDER)} 或 Qm=1/2/4/6/8"
        )
    return MODULATION_ORDER[key]


def select_base_graph(tb_size: int, target_rate: float) -> int:
    """按 TS 38.212 6.2.2 选择 LDPC base graph。

    规则（A 为 TB 大小，R 为目标码率）::

        BG2, 若 A <= 292
        BG2, 若 A <= 3824 且 R <= 0.67
        BG2, 若 R <= 0.25
        BG1, 其它情况

    Parameters
    ----------
    tb_size:
        TB 信息比特数 A（不含 CRC），必须为正整数。
    target_rate:
        目标码率 R（0 < R < 1）。接收端通常取 MCS 指示的码率，而不是
        用 A/G 反算的等效码率，因为基图选择规则中的 R 是调度码率。

    Returns
    -------
    int
        1 或 2。
    """
    if not isinstance(tb_size, int) or isinstance(tb_size, bool):
        raise TypeError("tb_size 必须是 int")
    if tb_size <= 0:
        raise ValueError("tb_size 必须大于 0")
    if not (0.0 < target_rate < 1.0):
        raise ValueError("target_rate 必须在 (0, 1) 内")

    if tb_size <= 292:
        return 2
    if tb_size <= 3824 and target_rate <= 0.67:
        return 2
    if target_rate <= 0.25:
        return 2
    return 1


def tb_crc_kind(tb_size: int) -> str:
    """返回 TB CRC 类型：A > 3824 用 ``"24A"``，否则用 ``"16"``。"""
    if tb_size <= 0:
        raise ValueError("tb_size 必须大于 0")
    return "24A" if tb_size > TB_CRC24A_THRESHOLD else "16"


def kb_for_bg2(b: int) -> int:
    """计算 BG2 的 K_b（TS 38.212 5.2.2）。

    ``b`` 是附加 TB CRC 后的长度 B，不是 B'。阈值按协议取严格大于：

    ============  ======
    B 的范围       K_b
    ============  ======
    B > 640        10
    560 < B <= 640  9
    192 < B <= 560  8
    B <= 192        6
    ============  ======
    """
    if b <= 0:
        raise ValueError("B 必须大于 0")
    if b > 640:
        return 10
    if b > 560:
        return 9
    if b > 192:
        return 8
    return 6


def kb_for_base_graph(bgn: int, b: int) -> int:
    """返回指定基图的信息列数 K_b（BG1=22，BG2 按 ``kb_for_bg2``）。"""
    if bgn == 1:
        return BG1_KB
    if bgn == 2:
        return kb_for_bg2(b)
    raise ValueError(f"不支持的 base graph {bgn}")


def codeword_length(bgn: int, z: int) -> int:
    """返回打孔 2*Zc 后的编码比特数 N（循环缓存长度）。"""
    _validate_bg_z(bgn, z)
    return (66 if bgn == 1 else 50) * z


def full_codeword_length(bgn: int, z: int) -> int:
    """返回打孔前的 LDPC 码字长度 N_full。"""
    _validate_bg_z(bgn, z)
    return (68 if bgn == 1 else 52) * z


def _validate_bg_z(bgn: int, z: int) -> None:
    """内部检查：基图编号和 lifting size 必须合法。"""
    if bgn not in (1, 2):
        raise ValueError(f"bgn 必须为 1 或 2，收到 {bgn}")
    if z not in Z_TO_ILS:
        raise ValueError(f"Zc={z} 不在 TS 38.212 Table 5.3.2-1 中")
