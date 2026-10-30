"""应用层：配置解析、端到端运行、结果结构与命令行接口。

这一层不含协议算法，只负责"把各部分串起来"：

* :mod:`.config`   -- JSON 模板 -> ``PUSCHCase``（校验 + 默认值）；
* :mod:`.report`   -- ``CaseResult`` / ``RunReport``（结果与序列化）；
* :mod:`.pipeline` -- TB 生成、信道、``run_case`` / ``run_config``；
* :mod:`.cli`      -- argparse 与终端报告。
"""

from .config import (
    PUSCHCase,
    derive_num_coded_bits,
    load_config,
    parse_case,
    parse_config,
)
from .pipeline import run_case, run_config
from .report import CaseResult, RunReport

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
