"""PUSCH 用例配置：JSON 模板 / 字典 -> 强类型 :class:`PUSCHCase`。

本模块属于应用层，只做“读配置、校验、解析”，不执行任何物理层运算；
真正的发送/接收链在 :mod:`nr_pusch_ldpc_rx.app.pipeline` 中运行。

在链路中的位置
--------------
``JSON 文件`` --(load_config)--> ``list[PUSCHCase]`` --(run_case)--> ``CaseResult``

设计原则
--------
1. **拒绝未知键**：``target_rat`` 这类拼写错误会直接报错，而不是被静默
   忽略后按默认值运行，导致实验结果难以解释。
2. **派生量不写进模板**：BG、Kb、Zc、C、K、F、N 全部由 TS 38.212 5.2.2
   的公式从 A/R 推导，模板只保存调度侧真正会给出的参数。
3. **错误信息带 JSON 路径**：例如 ``cases[1].num_coded_bits``，便于定位
   多用例模板中的问题。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..chain.scrambling import PUSCHScramblingConfig
from ..constants import modulation_order, tb_crc_kind

# ---------------------------------------------------------------------------
# 模板允许的键。新增字段时需要同步更新这里、README 的字段表以及测试。
# ---------------------------------------------------------------------------
_CASE_KEYS = frozenset(
    {
        "name",
        "description",
        "transport_block_size",
        "target_rate",
        "modulation",
        "rv",
        "n_layers",
        "num_coded_bits",
        "base_graph",
        "transport_block_bits",
        "scrambling",
        "ldpc",
        "channel",
    }
)
_TB_KEYS = frozenset({"source", "seed", "bits"})
_SCRAMBLING_KEYS = frozenset(
    {"n_rnti", "n_id", "codeword_index", "msg_a", "n_rapid"}
)
_LDPC_KEYS = frozenset({"max_iter", "alpha", "early_stop"})
_CHANNEL_KEYS = frozenset({"type", "ebn0_db", "seed"})
_TOP_KEYS = frozenset({"cases", "description"})

_TB_SOURCES = ("zeros", "ones", "alternating", "random", "explicit")


# ---------------------------------------------------------------------------
# 校验辅助函数：所有错误都带上 JSON 路径，便于定位模板问题
# ---------------------------------------------------------------------------
def _check_keys(
    obj: Mapping[str, Any],
    allowed: frozenset[str],
    where: str,
) -> None:
    """拒绝未知键，并在报错中列出允许的键。"""
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ValueError(
            f"{where}: 未知参数 {unknown}；允许的参数为 {sorted(allowed)}"
        )


def _as_int(value: Any, where: str) -> int:
    """把 JSON 数字转换为 int，拒绝 bool/浮点/字符串。

    Python 的 ``bool`` 是 ``int`` 的子类，``True`` 能通过 ``isinstance(x, int)``，
    因此必须显式排除，否则 ``"rv": true`` 会被当作 ``rv=1``。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{where} 必须是整数，收到 {value!r}")
    return value


def _as_float(value: Any, where: str) -> float:
    """把 JSON 数字转换为 float；整数也接受。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{where} 必须是数字，收到 {value!r}")
    return float(value)


def _as_bool(value: Any, where: str) -> bool:
    """要求 JSON 布尔值（不接受 0/1，避免 ``"early_stop": 1`` 这类歧义）。"""
    if not isinstance(value, bool):
        raise TypeError(f"{where} 必须是 true/false，收到 {value!r}")
    return value


def _as_mapping(value: Any, where: str) -> Mapping[str, Any]:
    """要求 JSON 对象。"""
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} 必须是 JSON 对象，收到 {type(value).__name__}")
    return value


def _in_range(value: int, lo: int, hi: int, where: str) -> int:
    """闭区间检查。"""
    if not (lo <= value <= hi):
        raise ValueError(f"{where} 必须在 {lo}..{hi} 内，收到 {value}")
    return value


# ---------------------------------------------------------------------------
# 用例配置对象
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PUSCHCase:
    """一个 JSON 用例解析后的强类型配置。

    该对象只保存"输入参数"，协议派生量（BG、Kb、Zc、C、K、F、N）在
    ``run_case`` 中通过 ``plan_transport_block`` 计算，避免模板里出现
    可以由规范推导的冗余字段。

    字段与 JSON 键一一对应，命名差异见 README 的模板字段表；其中
    ``tb_source`` / ``tb_seed`` / ``tb_bits`` 来自 ``transport_block_bits``
    子对象，``max_ldpc_iter`` / ``alpha`` / ``early_stop`` 来自 ``ldpc``
    子对象，``channel`` / ``ebn0_db`` / ``channel_seed`` 来自 ``channel``
    子对象。
    """

    name: str
    transport_block_size: int
    target_rate: float
    description: str | None = None
    modulation: str | int = "QPSK"
    rv: int = 0
    n_layers: int = 1
    num_coded_bits: int | None = None
    base_graph: int | None = None
    scrambling: PUSCHScramblingConfig | None = None
    tb_source: str = "random"
    tb_seed: int = 0
    tb_bits: tuple[int, ...] | None = None
    max_ldpc_iter: int = 30
    alpha: float = 0.8
    early_stop: bool = True
    channel: str = "none"
    ebn0_db: float = 0.0
    channel_seed: int = 0

    @property
    def qm(self) -> int:
        """调制阶数 Qm（每个调制符号携带的编码比特数）。"""
        return modulation_order(self.modulation)


# ---------------------------------------------------------------------------
# 子对象解析
# ---------------------------------------------------------------------------
def _parse_transport_block_bits(
    obj: Any,
    where: str,
    tb_size: int,
) -> tuple[str, int, tuple[int, ...] | None]:
    """解析 ``transport_block_bits``，返回 ``(source, seed, bits)``。

    ``source=explicit`` 时 ``bits`` 长度必须等于 A；其它来源不允许给
    ``bits``，避免"既随机又显式"的歧义。
    """
    if obj is None:
        return "random", 0, None
    data = _as_mapping(obj, where)
    _check_keys(data, _TB_KEYS, where)

    source = data.get("source", "random")
    if source not in _TB_SOURCES:
        raise ValueError(
            f"{where}.source 必须是 {list(_TB_SOURCES)} 之一，收到 {source!r}"
        )

    seed = _as_int(data.get("seed", 0), f"{where}.seed")
    raw_bits = data.get("bits")
    bits: tuple[int, ...] | None = None
    if raw_bits is not None:
        if not isinstance(raw_bits, Sequence) or isinstance(raw_bits, str):
            raise TypeError(f"{where}.bits 必须是数组")
        values: list[int] = []
        for index, bit in enumerate(raw_bits):
            value = _as_int(bit, f"{where}.bits[{index}]")
            if value not in (0, 1):
                raise ValueError(f"{where}.bits[{index}] 只能是 0 或 1")
            values.append(value)
        bits = tuple(values)

    if source == "explicit":
        if bits is None:
            raise ValueError(f"{where}: source=explicit 时必须提供 bits")
        if len(bits) != tb_size:
            raise ValueError(
                f"{where}.bits 长度 {len(bits)} 与 transport_block_size "
                f"{tb_size} 不一致"
            )
    elif bits is not None:
        raise ValueError(f"{where}: bits 只能与 source=explicit 一起使用")

    return source, seed, bits


def _parse_scrambling(obj: Any, where: str) -> PUSCHScramblingConfig | None:
    """解析 ``scrambling`` 对象；``null`` 表示不加扰（仅测试用）。

    正常 PUSCH 接收必须提供该对象，因为 TS 38.211 6.3.1.1 的加扰是
    发射机必做步骤；``null`` 只是为单元测试/对比实验保留的旁路。
    """
    if obj is None:
        return None
    data = _as_mapping(obj, where)
    _check_keys(data, _SCRAMBLING_KEYS, where)
    if "n_rnti" not in data:
        raise ValueError(f"{where}: 必须提供 n_rnti")
    return PUSCHScramblingConfig(
        n_rnti=_as_int(data["n_rnti"], f"{where}.n_rnti"),
        n_id=_as_int(data.get("n_id", 0), f"{where}.n_id"),
        codeword_index=_as_int(
            data.get("codeword_index", 0), f"{where}.codeword_index"
        ),
        msg_a=_as_bool(data.get("msg_a", False), f"{where}.msg_a"),
        n_rapid=(
            None
            if data.get("n_rapid") is None
            else _as_int(data["n_rapid"], f"{where}.n_rapid")
        ),
    )


def _parse_ldpc(obj: Any, where: str) -> tuple[int, float, bool]:
    """解析 ``ldpc``，返回 ``(max_iter, alpha, early_stop)``。"""
    if obj is None:
        return 30, 0.8, True
    data = _as_mapping(obj, where)
    _check_keys(data, _LDPC_KEYS, where)
    return (
        _as_int(data.get("max_iter", 30), f"{where}.max_iter"),
        _as_float(data.get("alpha", 0.8), f"{where}.alpha"),
        _as_bool(data.get("early_stop", True), f"{where}.early_stop"),
    )


def _parse_channel(obj: Any, where: str) -> tuple[str, float, int]:
    """解析 ``channel``，返回 ``(type, ebn0_db, seed)``。

    ``type=none`` 时不接受 ``ebn0_db``/``seed``：这两个字段只对 AWGN
    有意义，提前拒绝可以避免"以为加了噪声、实际没加"的模板错误。
    """
    if obj is None:
        return "none", 0.0, 0
    data = _as_mapping(obj, where)
    _check_keys(data, _CHANNEL_KEYS, where)
    kind = data.get("type", "none")
    if kind not in ("none", "awgn"):
        raise ValueError(f"{where}.type 必须是 'none' 或 'awgn'，收到 {kind!r}")
    ebn0 = _as_float(data.get("ebn0_db", 0.0), f"{where}.ebn0_db")
    seed = _as_int(data.get("seed", 0), f"{where}.seed")
    if kind == "none" and ("ebn0_db" in data or "seed" in data):
        raise ValueError(f"{where}: type=none 时不应提供 ebn0_db/seed")
    return kind, ebn0, seed


# ---------------------------------------------------------------------------
# 顶层解析入口
# ---------------------------------------------------------------------------
def parse_case(obj: Mapping[str, Any], where: str = "case") -> PUSCHCase:
    """解析单个用例对象。

    Parameters
    ----------
    obj:
        JSON 解析后的字典。
    where:
        错误信息中使用的路径前缀，例如 ``cases[0]``。

    Returns
    -------
    PUSCHCase
        校验完成的强类型配置。

    Raises
    ------
    ValueError / TypeError
        未知键、缺字段、类型或范围错误时抛出，消息中带 JSON 路径。
    """
    data = _as_mapping(obj, where)
    _check_keys(data, _CASE_KEYS, where)

    if "transport_block_size" not in data:
        raise ValueError(f"{where}: 缺少必填参数 transport_block_size")
    if "target_rate" not in data:
        raise ValueError(f"{where}: 缺少必填参数 target_rate")

    name = data.get("name", "case")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}.name 必须是非空字符串")

    description = data.get("description")
    if description is not None and not isinstance(description, str):
        raise TypeError(f"{where}.description 必须是字符串")

    tb_size = _as_int(data["transport_block_size"], f"{where}.transport_block_size")
    if tb_size <= 0:
        raise ValueError(f"{where}.transport_block_size 必须大于 0")

    rate = _as_float(data["target_rate"], f"{where}.target_rate")
    if not (0.0 < rate < 1.0):
        raise ValueError(f"{where}.target_rate 必须在 (0, 1) 内")

    modulation = data.get("modulation", "QPSK")
    if not isinstance(modulation, (str, int)) or isinstance(modulation, bool):
        raise TypeError(f"{where}.modulation 必须是字符串或整数 Qm")
    modulation_order(modulation)  # 提前触发非法调制方式的错误

    rv = _in_range(_as_int(data.get("rv", 0), f"{where}.rv"), 0, 3, f"{where}.rv")
    n_layers = _as_int(data.get("n_layers", 1), f"{where}.n_layers")
    if n_layers < 1:
        raise ValueError(f"{where}.n_layers 必须 >= 1")

    base_graph = data.get("base_graph")
    if base_graph is not None:
        base_graph = _as_int(base_graph, f"{where}.base_graph")
        if base_graph not in (1, 2):
            raise ValueError(f"{where}.base_graph 必须是 1、2 或 null")

    num_coded_bits = data.get("num_coded_bits")
    if num_coded_bits is not None:
        num_coded_bits = _as_int(num_coded_bits, f"{where}.num_coded_bits")
        if num_coded_bits <= 0:
            raise ValueError(f"{where}.num_coded_bits 必须大于 0")

    source, seed, bits = _parse_transport_block_bits(
        data.get("transport_block_bits"), f"{where}.transport_block_bits", tb_size
    )
    scrambling = _parse_scrambling(data.get("scrambling"), f"{where}.scrambling")
    max_iter, alpha, early_stop = _parse_ldpc(data.get("ldpc"), f"{where}.ldpc")
    channel, ebn0, channel_seed = _parse_channel(
        data.get("channel"), f"{where}.channel"
    )

    case = PUSCHCase(
        name=name,
        transport_block_size=tb_size,
        description=description,
        target_rate=rate,
        modulation=modulation,
        rv=rv,
        n_layers=n_layers,
        num_coded_bits=num_coded_bits,
        base_graph=base_graph,
        scrambling=scrambling,
        tb_source=source,
        tb_seed=seed,
        tb_bits=bits,
        max_ldpc_iter=max_iter,
        alpha=alpha,
        early_stop=early_stop,
        channel=channel,
        ebn0_db=ebn0,
        channel_seed=channel_seed,
    )

    # 用协议推导验证 G 与调制阶数/层数相容；同时尽早暴露 A/R/BG 组合错误。
    if num_coded_bits is None:
        derive_num_coded_bits(case)
    elif num_coded_bits % (case.qm * case.n_layers):
        raise ValueError(
            f"{where}.num_coded_bits={num_coded_bits} 必须是 Qm*N_L="
            f"{case.qm * case.n_layers} 的整数倍"
        )
    return case


def parse_config(obj: Any) -> list[PUSCHCase]:
    """解析整个 JSON 配置，支持单用例或 ``{"cases": [...]}``。"""
    data = _as_mapping(obj, "config")
    if "cases" in data:
        _check_keys(data, _TOP_KEYS, "config")
        cases = data["cases"]
        if not isinstance(cases, list) or not cases:
            raise ValueError("config.cases 必须是非空数组")
        return [
            parse_case(item, f"cases[{index}]")
            for index, item in enumerate(cases)
        ]
    return [parse_case(data, "case")]


def load_config(path: str | Path) -> list[PUSCHCase]:
    """从文件读取并解析 JSON 配置。"""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        try:
            obj = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{config_path} 不是合法 JSON: {exc}") from exc
    return parse_config(obj)


# ---------------------------------------------------------------------------
# 由目标码率推导 G
# ---------------------------------------------------------------------------
def derive_num_coded_bits(case: PUSCHCase) -> int:
    """按目标码率推导 G，并向上取整到 ``Qm*N_L`` 的整数倍。

    模板省略 ``num_coded_bits`` 时使用本函数::

        B = A + L
        G = ceil( ceil(B / R) / (Qm*N_L) ) * (Qm*N_L)

    内层 ``ceil(B/R)`` 是把目标码率折算成编码比特数；外层向上取整到
    ``Qm*N_L`` 是因为 TS 38.212 5.4.2.1 要求 G 能被 ``Qm*N_L`` 整除，
    否则无法把 G 均分到 C 个码块（``E_r`` 以 ``Qm*N_L`` 为单位）。
    """
    tb_crc_len = 24 if tb_crc_kind(case.transport_block_size) == "24A" else 16
    b = case.transport_block_size + tb_crc_len
    unit = case.qm * case.n_layers
    return int(math.ceil(math.ceil(b / case.target_rate) / unit) * unit)


__all__ = [
    "PUSCHCase",
    "parse_case",
    "parse_config",
    "load_config",
    "derive_num_coded_bits",
]
