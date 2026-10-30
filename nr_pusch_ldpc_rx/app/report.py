"""运行结果的数据结构：单个用例结果与整体报告。

这些 dataclass 只负责承载结果与序列化，不参与任何协议计算，因此 CLI、
库调用和测试可以共用同一套输出格式。``as_dict()`` 返回的字典保证只含
JSON 原生类型，可以直接 ``json.dump``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CaseResult:
    """单个用例的运行结果，可直接序列化为 JSON 报告。"""

    name: str
    passed: bool
    transport_block_size: int
    target_rate: float
    modulation: str | int
    qm: int
    rv: int
    n_layers: int
    num_coded_bits: int
    base_graph: int
    kb: int
    z: int
    code_blocks: int
    k: int
    filler_bits: int
    n: int
    c_init: int | None
    tb_crc_ok: bool
    cb_crc_ok: bool
    ldpc_converged: bool
    iterations: tuple[int, ...]
    e_lengths: tuple[int, ...]
    bit_errors: int
    decode_seconds: float
    description: str | None = None

    @property
    def bit_error_str(self) -> str:
        """TB 错误数/长度，便于表格显示（例如 ``0/1000``）。"""
        return f"{self.bit_errors}/{self.transport_block_size}"

    def as_dict(self) -> dict[str, Any]:
        """转成便于 ``json.dump`` 的字典。

        协议派生量统一放在 ``plan`` 子对象中，便于脚本按
        ``report["cases"][0]["plan"]["zc"]`` 读取。
        """
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "transport_block_size": self.transport_block_size,
            "target_rate": self.target_rate,
            "modulation": self.modulation,
            "qm": self.qm,
            "rv": self.rv,
            "n_layers": self.n_layers,
            "num_coded_bits": self.num_coded_bits,
            "plan": {
                "base_graph": self.base_graph,
                "kb": self.kb,
                "zc": self.z,
                "c": self.code_blocks,
                "k": self.k,
                "f": self.filler_bits,
                "n": self.n,
            },
            "scrambling": (
                None if self.c_init is None else {"c_init": self.c_init}
            ),
            "decode": {
                "tb_crc_ok": self.tb_crc_ok,
                "cb_crc_ok": self.cb_crc_ok,
                "ldpc_converged": self.ldpc_converged,
                "iterations": list(self.iterations),
                "e_lengths": list(self.e_lengths),
                "bit_errors": self.bit_errors,
                "seconds": round(self.decode_seconds, 6),
            },
        }


@dataclass(frozen=True)
class RunReport:
    """一次配置运行的整体结果（对应一个 JSON 文件或一组用例）。"""

    source: str
    cases: tuple[CaseResult, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> int:
        """通过数。"""
        return sum(1 for case in self.cases if case.passed)

    @property
    def failed(self) -> int:
        """失败数。"""
        return len(self.cases) - self.passed

    @property
    def all_passed(self) -> bool:
        """是否全部通过（空报告不算通过）。"""
        return self.failed == 0 and bool(self.cases)

    def as_dict(self) -> dict[str, Any]:
        """整体报告字典，含 summary 与逐用例详情。"""
        return {
            "source": self.source,
            "summary": {
                "total": len(self.cases),
                "passed": self.passed,
                "failed": self.failed,
            },
            "cases": [case.as_dict() for case in self.cases],
        }


__all__ = ["CaseResult", "RunReport"]
