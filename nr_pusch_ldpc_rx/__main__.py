"""``python -m nr_pusch_ldpc_rx``：按 JSON 模板运行 PUSCH 端到端流程。

真正的实现位于 :mod:`nr_pusch_ldpc_rx.app.cli`；这里只保留包级入口，
使 ``python -m nr_pusch_ldpc_rx`` 和 ``run_pusch.py`` 共用同一套逻辑。
"""

from __future__ import annotations

from .app.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
