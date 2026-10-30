"""端到端运行器：TB 生成 -> 发送链 -> 信道 -> 接收链 -> 结果统计。

本模块把"一个用例"串起来：:mod:`~nr_pusch_ldpc_rx.link.transmitter` 生成
编码比特，:mod:`~nr_pusch_ldpc_rx.app.pipeline` 里的信道模型把它变成 LLR，
:mod:`~nr_pusch_ldpc_rx.link.receiver` 还原 TB，最后统计 CRC 与误比特数。
JSON 模板的解析与校验在 :mod:`~nr_pusch_ldpc_rx.app.config`，结果结构在
:mod:`~nr_pusch_ldpc_rx.app.report`。

用法
----
命令行::

    python3 -m nr_pusch_ldpc_rx templates/pusch_example.json
    python3 run_pusch.py templates/pusch_example.json          # 等价入口
    python3 -m nr_pusch_ldpc_rx templates/pusch_example.json --json-out report.json

作为库::

    from nr_pusch_ldpc_rx import run_config
    report = run_config("templates/pusch_example.json")
    print(report.passed, "/", len(report.cases))
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..chain.segmentation import TransportBlockPlan, plan_transport_block
from ..link.receiver import PUSCHReceiverConfig, decode_transport_block
from ..link.transmitter import encode_transport_block
from .config import (
    PUSCHCase,
    derive_num_coded_bits,
    load_config,
    parse_case,
    parse_config,
)
from .report import CaseResult, RunReport

# 无噪声时的软值幅度：只要大于 0 就能判决正确，取 4.0 是为了让 LLR
# 与 AWGN 场景的典型幅度接近，便于观察 LDPC 迭代次数。
_NOISELESS_LLR = 4.0


def _make_tb(case: PUSCHCase) -> np.ndarray:
    """按配置生成 TB 比特。"""
    size = case.transport_block_size
    if case.tb_source == "zeros":
        return np.zeros(size, dtype=np.uint8)
    if case.tb_source == "ones":
        return np.ones(size, dtype=np.uint8)
    if case.tb_source == "alternating":
        return (np.arange(size) % 2).astype(np.uint8)
    if case.tb_source == "random":
        rng = np.random.default_rng(case.tb_seed)
        return rng.integers(0, 2, size, dtype=np.uint8)
    if case.tb_source == "explicit":
        if case.tb_bits is None:  # parse_case 已保证，防御性检查
            raise AssertionError("source=explicit 但缺少 bits")
        return np.asarray(case.tb_bits, dtype=np.uint8)
    raise AssertionError(f"未知 TB 来源 {case.tb_source}")


def _coded_bits_to_llr(
    coded_bits: np.ndarray,
    case: PUSCHCase,
) -> np.ndarray:
    """把发送端编码比特变成接收端 LLR。

    无噪声时用 ``±_NOISELESS_LLR`` 的硬软值；``channel=awgn`` 时按
    **有效码率** ``R_eff = A/G`` 计算 Eb/N0 对应的噪声方差，再经 BPSK
    加噪后转成 LLR。

    推导（BPSK，Es=1，N0=2σ²）
    --------------------------
    ``Eb/N0 = Es / (R_eff * N0)``，因此::

        σ² = 1 / (2 * R_eff * 10^(EbN0_dB/10))

    对 ``y = x + n``（x=±1，n~N(0,σ²)），比特 LLR 为::

        L = ln P(b=0|y)/P(b=1|y) = 2y / σ²

    这里直接用 A/G 作为有效码率，而不是调度码率 R：二者在有 CRC、
    filler 和打孔时并不相等，信道模型应按实际传输的比特数折算。
    """
    binary = np.asarray(coded_bits, dtype=np.uint8)
    if binary.size and np.any(binary > 1):
        raise ValueError("coded_bits 只能包含 0/1")

    if case.channel == "none":
        return np.where(binary == 0, _NOISELESS_LLR, -_NOISELESS_LLR)

    # 0 -> +1，1 -> -1（BPSK）。
    symbols = 1.0 - 2.0 * binary.astype(np.float64)
    rate = case.transport_block_size / binary.size
    sigma = math.sqrt(1.0 / (2.0 * rate * 10.0 ** (case.ebn0_db / 10.0)))
    rng = np.random.default_rng(case.channel_seed)
    received = symbols + rng.normal(0.0, sigma, symbols.size)
    return 2.0 * received / (sigma * sigma)


def run_case(
    case: PUSCHCase,
    plan: TransportBlockPlan | None = None,
) -> CaseResult:
    """执行一个用例的完整发送/接收链。

    Parameters
    ----------
    case:
        ``parse_case`` 得到的配置。
    plan:
        可选的预计算 ``TransportBlockPlan``；测试中可复用。

    Returns
    -------
    CaseResult
        含协议派生量、CRC 结果、迭代次数、误比特数和耗时。

    Notes
    -----
    ``passed`` 的判据是 **TB CRC 通过且误比特数为 0**。只检查 CRC 不够：
    CRC 对全零码字没有检错能力（见 README FAQ），因此额外比较原始 TB。
    """
    if plan is None:
        plan = plan_transport_block(
            case.transport_block_size, case.target_rate, bgn=case.base_graph
        )
    g = (
        case.num_coded_bits
        if case.num_coded_bits is not None
        else derive_num_coded_bits(case)
    )

    tb = _make_tb(case)
    encoded = encode_transport_block(
        tb,
        g,
        rv=case.rv,
        qm=case.qm,
        n_layers=case.n_layers,
        plan=plan,
        scrambling=case.scrambling,
    )
    llr = _coded_bits_to_llr(encoded.coded_bits, case)

    receiver_config = PUSCHReceiverConfig(
        transport_block_size=case.transport_block_size,
        target_rate=case.target_rate,
        rv=case.rv,
        modulation=case.modulation,
        n_layers=case.n_layers,
        bgn=case.base_graph,
        max_ldpc_iter=case.max_ldpc_iter,
        alpha=case.alpha,
        early_stop=case.early_stop,
        scrambling=case.scrambling,
    )

    start = time.perf_counter()
    result = decode_transport_block(llr, receiver_config, plan=plan)
    decode_seconds = time.perf_counter() - start

    bit_errors = int(np.count_nonzero(result.tb_bits != tb))
    tb_crc_ok = bool(result.tb_crc_ok)
    passed = bool(tb_crc_ok and bit_errors == 0)

    return CaseResult(
        name=case.name,
        passed=passed,
        transport_block_size=case.transport_block_size,
        description=case.description,
        target_rate=case.target_rate,
        modulation=case.modulation,
        qm=case.qm,
        rv=case.rv,
        n_layers=case.n_layers,
        num_coded_bits=g,
        base_graph=plan.bgn,
        kb=plan.kb,
        z=plan.z,
        code_blocks=plan.c,
        k=plan.k,
        filler_bits=plan.f,
        n=plan.n,
        c_init=(case.scrambling.c_init if case.scrambling else None),
        tb_crc_ok=tb_crc_ok,
        cb_crc_ok=bool(result.cb_crc_ok),
        ldpc_converged=bool(result.all_ldpc_converged),
        iterations=tuple(cb.ldpc.iterations for cb in result.code_blocks),
        e_lengths=tuple(int(cb.e_len) for cb in result.code_blocks),
        bit_errors=bit_errors,
        decode_seconds=decode_seconds,
    )


def run_config(
    config: str | Path | Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> RunReport:
    """读取配置（路径或字典）并运行其中全部用例。

    ``config`` 可以是 JSON 文件路径、已解析的字典，或直接给用例字典列表；
    这是给交互式实验和单元测试提供的三种入口。
    """
    if isinstance(config, (str, Path)):
        cases = load_config(config)
        source = str(config)
    elif isinstance(config, Mapping):
        cases = parse_config(config)
        source = "<dict>"
    else:
        cases = [
            parse_case(item, f"cases[{index}]")
            for index, item in enumerate(config)
        ]
        source = "<list>"
    return RunReport(
        source=source,
        cases=tuple(run_case(case) for case in cases),
    )


# 兼容旧用法：``from nr_pusch_ldpc_rx.app.pipeline import PUSCHCase`` 等。
__all__ = [
    "CaseResult",
    "PUSCHCase",
    "RunReport",
    "derive_num_coded_bits",
    "load_config",
    "parse_case",
    "parse_config",
    "run_case",
    "run_config",
]
