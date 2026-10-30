"""运行 PUSCH 接收链演示：打印不同 BG/Kb/Zc 的参数并核对 TB CRC。

用法::

    python3 demo.py

演示只使用无噪声 BPSK 软比特，目的是验证算法链路；误码率/SNR 扫描可以
在 ``decode_transport_block`` 外自行替换 LLR 生成器。

每个用例都启用 PUSCH 加扰（TS 38.211 6.3.1.1），接收端先解扰再进入
解交织/解速率匹配/LDPC 译码。
"""

from __future__ import annotations

import math

import numpy as np

from nr_pusch_ldpc_rx import (
    PUSCHReceiverConfig,
    PUSCHScramblingConfig,
    decode_transport_block_from_bits,
    encode_transport_block,
    plan_transport_block,
)


CASES = [
    # A,     R,   Qm, rv
    (100, 0.30, 2, 0),      # BG2, Kb=6,  Zc=20
    (545, 0.30, 2, 0),      # BG2, Kb=9,  Zc=64
    (1000, 0.30, 6, 0),     # BG2, Kb=10, Zc=104
    (5000, 0.80, 2, 0),     # BG1, Kb=22, Zc=240
    (20000, 0.70, 4, 2),    # BG1, 3 个 CB, rv=2
    (11000, 0.20, 6, 1),    # BG2, 3 个 CB, Zc=384, rv=1
]


def main() -> int:
    rng = np.random.default_rng(2029)
    header = (
        f"{'A':>6} {'R':>5} {'BG':>3} {'Kb':>3} {'Zc':>4} {'C':>2} "
        f"{'K':>5} {'F':>4} {'G':>7} {'Qm':>3} {'rv':>2} "
        f"{'RNTI':>6} {'n_ID':>4} {'TB CRC':>7} {'LDPC iters':>12}"
    )
    print(header)
    print("-" * len(header))

    all_ok = True
    for a, rate, qm, rv in CASES:
        plan = plan_transport_block(a, rate)
        # 演示用的 G 选取策略：
        # * rv=0 时按目标码率把 B 折算成编码比特数；
        # * rv>0 时故意取满整个循环缓存 C*N，让“绕回重复读取/软合并”
        #   这条路径也被覆盖；
        # 最后向上取整到 Qm 的整数倍，满足 E_r 的分配单位。
        unit = qm
        raw_g = plan.c * plan.n if rv else math.ceil(plan.b / rate)
        g = math.ceil(raw_g / unit) * unit

        # 加扰参数：RNTI / dataScramblingIdentityPUSCH（未配置时填小区 ID）。
        scrambling = PUSCHScramblingConfig(
            n_rnti=0x4600 + (a % 200),
            n_id=(a * 7) % 1024,
        )
        tb = rng.integers(0, 2, a, dtype=np.uint8)
        encoded = encode_transport_block(
            tb, g, rv=rv, qm=qm, plan=plan, scrambling=scrambling
        )
        config = PUSCHReceiverConfig(
            transport_block_size=a,
            target_rate=rate,
            rv=rv,
            modulation=qm,
            max_ldpc_iter=30,
            scrambling=scrambling,
        )
        result = decode_transport_block_from_bits(
            encoded.coded_bits, config, plan=plan, amplitude=4.0
        )
        ok = (
            result.tb_crc_ok
            and result.cb_crc_ok
            and np.array_equal(result.tb_bits, tb)
        )
        all_ok &= ok
        iters = ",".join(str(cb.ldpc.iterations) for cb in result.code_blocks)
        print(
            f"{a:>6} {rate:>5.2f} {plan.bgn:>3} {plan.kb:>3} {plan.z:>4} "
            f"{plan.c:>2} {plan.k:>5} {plan.f:>4} {g:>7} {qm:>3} {rv:>2} "
            f"{scrambling.n_rnti:>6} {scrambling.n_id:>4} "
            f"{'PASS' if result.tb_crc_ok else 'FAIL':>7} {iters:>12}"
        )

    print()
    print("总体结果:", "全部通过" if all_ok else "存在失败")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
