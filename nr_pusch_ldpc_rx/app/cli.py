"""命令行接口：``python -m nr_pusch_ldpc_rx <template.json>``。

只负责参数解析和终端输出，不包含协议逻辑；实际运行在
:func:`nr_pusch_ldpc_rx.app.pipeline.run_config`。

退出码约定
----------
* ``0``：全部用例通过；
* ``1``：链路跑通但至少一个用例失败（CRC 错/误比特）；
* ``2``：配置或文件错误（模板不合法、文件不存在等）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import run_config
from .report import CaseResult, RunReport


def _format_case(index: int, total: int, case: CaseResult) -> list[str]:
    """把一个用例结果格式化成多行文本。

    第一行是状态与用例名；后三行依次是协议派生量、调制/加扰参数、
    译码结果，方便直接对比不同配置。
    """
    status = "PASS" if case.passed else "FAIL"
    title = f"[{index}/{total}] {case.name}  -> {status}"
    if case.description:
        title += f"  ({case.description})"
    return [
        title,
        (
            f"      A={case.transport_block_size} R={case.target_rate:g} "
            f"BG{case.base_graph} Kb={case.kb} Zc={case.z} C={case.code_blocks} "
            f"K={case.k} F={case.filler_bits} N={case.n} G={case.num_coded_bits}"
        ),
        (
            f"      Qm={case.qm} rv={case.rv} N_L={case.n_layers} "
            f"c_init={case.c_init if case.c_init is not None else 'off'}"
        ),
        (
            f"      TB CRC={'PASS' if case.tb_crc_ok else 'FAIL'} | "
            f"CB CRC={'PASS' if case.cb_crc_ok else 'FAIL'} | "
            f"LDPC={'converged' if case.ldpc_converged else 'not converged'} | "
            f"bits={case.bit_error_str} | "
            f"iters={','.join(str(i) for i in case.iterations)} | "
            f"{case.decode_seconds * 1e3:.2f} ms"
        ),
    ]


def print_report(report: RunReport) -> None:
    """在终端打印可读报告。"""
    total = len(report.cases)
    for index, case in enumerate(report.cases, start=1):
        for line in _format_case(index, total, case):
            print(line)
        print()
    print(
        f"总计 {total} 个用例：通过 {report.passed}，失败 {report.failed}"
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="python -m nr_pusch_ldpc_rx",
        description="读取 JSON 模板并运行 PUSCH 端到端（发送+接收）流程。",
    )
    parser.add_argument(
        "template",
        help='JSON 模板路径；可以是单用例或 {"cases": [...]} 形式',
    )
    parser.add_argument(
        "--json-out",
        metavar="FILE",
        help="把完整报告写成 JSON 文件",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印逐用例报告，只输出汇总（仍可用 --json-out 保存详情）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = run_config(args.template)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print_report(report)
    else:
        print(
            f"{report.passed}/{len(report.cases)} 个用例通过"
            f"（来源 {report.source}）"
        )

    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"报告已写入 {output}")

    return 0 if report.all_passed else 1


__all__ = ["build_parser", "main", "print_report"]
