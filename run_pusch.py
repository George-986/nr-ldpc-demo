"""便捷入口：``python3 run_pusch.py <template.json>``。

等价于 ``python3 -m nr_pusch_ldpc_rx <template.json>``；保留这个脚本是为了
在未把项目当作包安装、且习惯直接运行脚本时也能用。
"""

from __future__ import annotations

from nr_pusch_ldpc_rx.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
