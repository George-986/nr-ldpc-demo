# NR PUSCH LDPC 收发链（Python / NumPy）

按 3GPP TS 38.212 / TS 38.211 实现的 PUSCH（UL-SCH）LDPC **接收链**参考代码，
并附带一条严格互逆的**发送端参考链路**，用于生成逐比特可核对的测试向量。

```text
发送：TB CRC -> 码块分割 -> LDPC 编码 -> 速率匹配 -> 比特交织 -> 码块拼接 -> 加扰
接收：解扰 -> 按 E_r 切分 -> 解比特交织 -> 解速率匹配 -> LDPC 译码 -> CB CRC -> TB 拼接 -> TB CRC
```

只依赖 **NumPy**，不需要 MATLAB、TensorFlow 或通信工具箱。

参考规范：

- [ETSI TS 138 212 V18.3.0](https://www.etsi.org/deliver/etsi_ts/138200_138299/138212/18.03.00_60/ts_138212v180300p.pdf)
  （3GPP TS 38.212 V18.3.0）：CRC、分割、LDPC、速率匹配、比特交织；
- [ETSI TS 138 211 V18.3.0](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/18.03.00_60/ts_138211v180300p.pdf)
  （3GPP TS 38.211 V18.3.0）：Gold 序列与 PUSCH 加扰。

---

## 1. 功能范围

### 1.1 已实现

| 功能 | 规范条款 | 代码 |
|---|---|---|
| BG1/BG2 选择、Kb/Zc 查表 | 38.212 6.2.2、5.2.2、表 5.3.2-1 | [constants.py](nr_pusch_ldpc_rx/constants.py) |
| CRC16 / CRC24A / CRC24B | 38.212 5.1 | [crc.py](nr_pusch_ldpc_rx/chain/crc.py) |
| TB CRC、CB 分割、CB CRC、filler | 38.212 5.2.2 | [segmentation.py](nr_pusch_ldpc_rx/chain/segmentation.py) |
| 基图表、lifting、`H*c^T=0` 校验 | 38.212 5.3.2、表 5.3.2-2/3 | [ldpc/](nr_pusch_ldpc_rx/ldpc) |
| LDPC 编码与打孔 | 38.212 5.3.2 | [codec.py](nr_pusch_ldpc_rx/ldpc/codec.py) |
| 分层归一化最小和译码 | —（接收算法） | [codec.py](nr_pusch_ldpc_rx/ldpc/codec.py) |
| E_r、k0、循环缓存、比特交织 | 38.212 5.4.2 | [rate_matching.py](nr_pusch_ldpc_rx/chain/rate_matching.py) |
| Gold 序列、c_init、加扰/解扰 | 38.211 5.2.1、6.3.1.1 | [scrambling.py](nr_pusch_ldpc_rx/chain/scrambling.py) |
| 发送端参考链、接收端编排 | 38.212 5.2–5.5、38.211 6.3.1.1 | [transmitter.py](nr_pusch_ldpc_rx/link/transmitter.py)、[receiver.py](nr_pusch_ldpc_rx/link/receiver.py) |
| JSON 模板、CLI、报告 | — | [app/](nr_pusch_ldpc_rx/app) |

覆盖的配置维度：

- BG1 / BG2；单码块 / 多码块；`Zc = 2..384` 的全部 51 个 lifting size；
- `rv = 0/1/2/3`；`N_L = 1/2`；QPSK / 16QAM / 64QAM / 256QAM / BPSK / π/2-BPSK；
- PUSCH 常规加扰与 msgA（Type-2 随机接入）加扰；
- 无噪声硬软值与 AWGN 软 LLR 两种信道。

### 1.2 不在范围内

- DMRS、信道估计、均衡、星座解映射、资源解映射（应在调用接收链前完成）；
- UCI 复用到 PUSCH 产生的 `x`/`y` 占位比特（只处理纯 UL-SCH 数据）；
- CBGTI、LBRM（接收端默认 `N_cb = N`；底层函数保留 `n_cb` 参数）；
- 调制、层映射、资源映射（发送链输出的是编码比特，不是 OFDM 波形）；
- 精确 sum-product 译码（使用 NMS 近似）。

### 1.3 阅读建议

| 目标 | 建议从哪读 |
|---|---|
| 先把程序跑起来 | §2 快速开始 |
| 用 JSON 做实验 | §7 JSON 模板 |
| 理解每个协议符号 | §5 参数速查 |
| 弄清最容易写错的协议细节 | §6 关键协议点 |
| 修改/扩展代码 | §3 目录结构、§4 架构与数据流，再读各模块 docstring |

---

## 2. 快速开始

在项目根目录执行：

```bash
python3 -m pip install -r requirements.txt   # 只依赖 numpy
python3 demo.py
python3 run_pusch.py templates/pusch_example.json   # 从 JSON 模板跑整条链
python3 -m unittest discover -s tests -v             # 98 个测试
```

`demo.py` 会打印不同 BG / Kb / Zc / 码块数、且启用 PUSCH 加扰的端到端结果：

```text
     A     R  BG  Kb   Zc  C     K    F       G  Qm rv   RNTI n_ID  TB CRC   LDPC iters
---------------------------------------------------------------------------------------
   100  0.30   2   6   20  1   200   84     388   2  0  18020  700    PASS            1
   545  0.30   2   9   64  1   640   79    1870   2  0  18065  743    PASS            1
  1000  0.30   2  10  104  1  1040   24    3390   6  0  17920  856    PASS            1
  5000  0.80   1  22  240  1  5280  256    6280   2  0  17920  184    PASS            1
 20000  0.70   1  22  320  3  7040  341   63360   4  2  17920  736    PASS        1,1,1
 11000  0.20   2  10  384  3  3840  141   57600   6  1  17920  200    PASS        1,1,1
```

`python3 run_pusch.py ...` 与 `python3 -m nr_pusch_ldpc_rx ...` 完全等价，
前者只是为“不安装包、直接运行脚本”保留的入口。

---

## 3. 目录结构与模块职责

### 3.1 目录树

```text
nr_pusch_ldpc_rx/
├── demo.py                          # 可运行的算法演示（无 JSON）
├── run_pusch.py                     # 便捷 CLI 入口（等价于 python -m）
├── requirements.txt                 # numpy>=1.24
├── templates/
│   ├── pusch_template.json          # 单用例模板（列出全部可选字段）
│   └── pusch_example.json           # 多用例示例（BG1/BG2、多 CB、rv、msgA、AWGN）
├── nr_pusch_ldpc_rx/
│   ├── __init__.py                  # 公共 API 聚合（层间解耦的对外门面）
│   ├── __main__.py                  # python -m 入口，转调 app.cli
│   ├── constants.py                 # 协议常量、BG/Kb/Zc 选择规则
│   ├── ldpc/                        # LDPC 码层（TS 38.212 5.3.2）
│   │   ├── __init__.py              # 导出 encode/decode/基图工具
│   │   ├── tables.py                # 表 5.3.2-2/3 原始表项（纯数据）
│   │   ├── base_graph.py            # 提升、移位、H*c^T 校验子
│   │   └── codec.py                 # LDPC 编码器 + 分层 NMS 译码器
│   ├── chain/                       # 传输链处理级
│   │   ├── __init__.py              # 统一导出
│   │   ├── crc.py                   # CRC16/24A/24B（38.212 5.1）
│   │   ├── segmentation.py          # TB CRC、分割、CB CRC、拼接（5.2.2）
│   │   ├── rate_matching.py         # E_r、k0、循环缓存、交织（5.4.2）
│   │   └── scrambling.py            # Gold、c_init、加扰/解扰（38.211）
│   ├── link/                        # 端到端链路编排
│   │   ├── __init__.py              # 统一导出
│   │   ├── transmitter.py           # 发送端参考链
│   │   └── receiver.py              # 接收链主入口
│   └── app/                         # 应用层（无协议算法）
│       ├── __init__.py              # 统一导出
│       ├── config.py                # JSON -> PUSCHCase（校验/默认值）
│       ├── report.py                # CaseResult / RunReport
│       ├── pipeline.py              # TB 生成、信道、run_case/run_config
│       └── cli.py                   # argparse、终端报告、退出码
└── tests/
    ├── test_constants.py
    ├── test_base_graph.py
    ├── test_crc.py
    ├── test_segmentation.py
    ├── test_rate_matching.py
    ├── test_scrambling.py
    ├── test_ldpc.py
    ├── test_pipeline.py
    └── test_end_to_end.py
```

### 3.2 分层与依赖

依赖方向自上而下，**下层不 import 上层**：

```text
app/       配置解析、端到端运行、报告、CLI       （应用编排）
  |
link/      发送端参考链 / 接收端主入口             （链路编排）
  |
chain/     CRC、分割、速率匹配、加扰                （传输链处理级）
  |
ldpc/      基图表、编码、译码                       （信道编码）
  |
constants.py  全层共用的协议常量与选择规则          （最底层）
```

这样划分的好处：

- 每个模块只依赖规范中的一个条款，可以单独测试（例如 `chain/crc.py`
  不依赖分割和 LDPC）；
- 替换译码器、增加新的信道模型或新配置项时，不需要改协议原语；
- `app/` 只处理“配置/报告/命令行”，可以整体替换成其它前端（例如图形界面）。

### 3.3 模块职责速查

| 模块 | 职责 | 关键对象/函数 |
|---|---|---|
| [constants.py](nr_pusch_ldpc_rx/constants.py) | Z 表、BG 尺寸、Kb 阈值、Qm、CRC 阈值 | `select_base_graph`、`kb_for_bg2`、`codeword_length` |
| [ldpc/tables.py](nr_pusch_ldpc_rx/ldpc/tables.py) | 表 5.3.2-2/3 的原始三元组 | `BASE_GRAPH_1_EDGES`、`BASE_GRAPH_2_EDGES` |
| [ldpc/base_graph.py](nr_pusch_ldpc_rx/ldpc/base_graph.py) | 表项提升为基图、循环移位、校验子 | `base_graph_for_z`、`shift`、`hard_syndrome_ok` |
| [ldpc/codec.py](nr_pusch_ldpc_rx/ldpc/codec.py) | 编码（前 4 个校验块 + 逐行递推）、分层 NMS 译码 | `encode`、`decode`、`LDPCDecodeResult` |
| [chain/crc.py](nr_pusch_ldpc_rx/chain/crc.py) | 三个 CRC 的 LFSR 计算、附加、校验 | `crc_attach`、`crc_check`、`crc_strip` |
| [chain/segmentation.py](nr_pusch_ldpc_rx/chain/segmentation.py) | 由 A/R 推导全部分割参数；TB↔CB 拼接 | `TransportBlockPlan`、`plan_transport_block` |
| [chain/rate_matching.py](nr_pusch_ldpc_rx/chain/rate_matching.py) | E_r 分配、k0、循环缓存、比特交织/解交织 | `coded_bit_lengths`、`rate_match`、`rate_recover` |
| [chain/scrambling.py](nr_pusch_ldpc_rx/chain/scrambling.py) | Gold 序列、c_init、比特加扰、LLR 解扰 | `gold_sequence`、`PUSCHScramblingConfig` |
| [link/transmitter.py](nr_pusch_ldpc_rx/link/transmitter.py) | 把上述步骤串成发送链，保留每级中间结果 | `encode_transport_block` |
| [link/receiver.py](nr_pusch_ldpc_rx/link/receiver.py) | 把逆过程串成接收链 | `decode_transport_block`、`PUSCHReceiverConfig` |
| [app/config.py](nr_pusch_ldpc_rx/app/config.py) | JSON 校验、默认值、强类型配置 | `PUSCHCase`、`parse_config` |
| [app/pipeline.py](nr_pusch_ldpc_rx/app/pipeline.py) | TB 生成、AWGN、运行用例、汇总报告 | `run_case`、`run_config` |
| [app/report.py](nr_pusch_ldpc_rx/app/report.py) | 结果 dataclass 与 JSON 序列化 | `CaseResult.as_dict`、`RunReport` |
| [app/cli.py](nr_pusch_ldpc_rx/app/cli.py) | 命令行参数、终端输出、退出码 | `main`、`print_report` |

---

## 4. 架构与数据流

### 4.1 发送链

```text
TB (A bit)
  │ crc_attach(plan.tb_crc)                         TS 38.212 5.1
  ▼
b (B = A + L)
  │ make_code_blocks：分割 + CB CRC24B + filler      TS 38.212 5.2.2
  ▼
C 个 CB (每个 K = 22*Zc 或 10*Zc，含 filler=0)
  │ ldpc.encode                                      TS 38.212 5.3.2
  ▼
完整码字 (N_full = 68*Zc 或 52*Zc)
  │ 打孔前 2*Zc 个系统比特
  ▼
d (N = 66*Zc 或 50*Zc)
  │ rate_match：E_r、k0、跳过 NULL、比特交织        TS 38.212 5.4.2
  ▼
f_r (E_r)
  │ 码块拼接
  ▼
b (G)
  │ scramble_bits(Gold 序列)                        TS 38.211 6.3.1.1
  ▼
b~ (G) -> 调制（不在本项目范围）
```

### 4.2 接收链

```text
解调 LLR (G，正数 -> 0，负数 -> 1)
  │ 0) descramble_llr：按 Gold 序列翻转符号          TS 38.211 6.3.1.1
  ▼
b 的 LLR (G)
  │ 1) coded_bit_lengths：按 E_r 切分                TS 38.212 5.4.2.1
  ▼
C 段 LLR (E_r)
  │ 2) rate_recover：解交织 + 按 k0 填循环缓存       TS 38.212 5.4.2
  ▼
d 域 LLR (N)；filler 位置为 FILLER_LLR
  │ 3) 头部补 2*Zc 个 0 LLR（打孔位）
  ▼
LDPC 输入 LLR (N_full)
  │ 4) ldpc.decode：分层 NMS                         TS 38.212 5.3.2
  ▼
每个 CB 的系统比特 (K)
  │ 5) desegment_code_blocks：CB CRC、去 filler、拼接 TS 38.212 5.2.2
  ▼
b 的硬判决 (B)
  │ 6) check_transport_block_crc：TB CRC
  ▼
TB (A) + CRC 结果
```

### 4.3 关键设计约定

1. **比特顺序**：所有比特数组都是 `uint8` 一维数组，下标 0 是协议中
   最早/最高次的比特；CRC 输入输出均为 MSB-first。
2. **LLR 符号**：正数倾向比特 0，负数倾向比特 1。打孔位 LLR=0（完全
   未知），filler 位 LLR=`FILLER_LLR`（已知为 0）。
3. **NULL 的表示**：编码侧 filler 按 0 参与运算，用
   `TransportBlockPlan.filler_mask` 记录位置；速率匹配读取时跳过这些
   位置；接收侧再把它们填成大正 LLR。
4. **d 域 vs 完整码字域**：`rate_matching` 的所有下标（包括
   `filler_mask`）都在**打孔后的 d 域**；只有 LDPC 编解码使用完整
   码字域。二者相差 `2*Zc`，这是最容易错位的地方。
5. **计划对象共享**：`TransportBlockPlan` 是不可变的，发送端和接收端
   必须使用同一份（或由相同 A/R/BG 推导出的等价计划）。JSON 流程在
   内部只推导一次并同时传给两侧。

---

## 5. 协议参数速查

| 符号 | 含义 | 取值/公式 |
|---|---|---|
| `A` | TB 信息比特数（不含 CRC） | 调度给定 |
| `L` | TB CRC 长度 | `A > 3824` 时 24，否则 16 |
| `B` | 加 TB CRC 后的长度 | `A + L` |
| `K_cb` | 单 CB 最大长度 | BG1 8448，BG2 3840 |
| `C` | 码块数 | `B <= K_cb` 时 1；否则 `ceil(B/(K_cb-24))` |
| `L_cb` | 每 CB 的 CRC | `C=1` 时 0，否则 24（CRC24B） |
| `B'` | 加所有 CB CRC 后长度 | `B + C*L_cb` |
| `K'` | 每 CB 加 filler 前长度 | `ceil(B'/C)` |
| `K_b` | 选 Zc 用的信息列数 | BG1 22；BG2 按 `B` 取 6/8/9/10 |
| `Z_c` | lifting size | 表 5.3.2-1 中最小的 `Z` 使 `K_b*Z >= K'` |
| `K` | LDPC 编码器输入长度 | `22*Zc`（BG1）或 `10*Zc`（BG2） |
| `F` | filler 数 | `K - K'` |
| `N` | 打孔后码字长度 | `66*Zc`（BG1）或 `50*Zc`（BG2） |
| `N_full` | 打孔前码字长度 | `68*Zc`（BG1）或 `52*Zc`（BG2） |
| `N_cb` | 循环缓存长度 | 未配置 LBRM 时 `N_cb = N` |
| `G` | 整个 TB 的编码比特数 | 调度给定，须为 `Qm*N_L` 的整数倍 |
| `E_r` | 第 r 个 CB 的速率匹配输出长度 | 见 §6.5，`sum(E_r) = G` |
| `Qm` | 调制阶数 | 1/2/4/6/8 |
| `N_L` | 传输层数 | 通常 1 或 2 |
| `rv` | 冗余版本 | 0/1/2/3 |
| `k0` | 循环缓存读取起点 | 表 5.4.2.1-2，见 §6.5 |
| `c_init` | Gold 序列初值 | 见 §6.7 |

`TransportBlockPlan` 同时提供派生量：`n`、`n_full`、`data_bits_per_cb`、
`total_data_capacity`、`zero_padding`、`filler_start_d`、`filler_end_d`、
`filler_mask`。

---

## 6. 关键协议点（实现难点）

这一节按处理顺序解释规范中“看公式容易、写代码容易错”的地方。代码里
对应的位置都加了注释，可对照阅读。

### 6.1 Base graph 选择（TS 38.212 6.2.2）

```text
BG2，若 A <= 292
BG2，若 A <= 3824 且 R <= 0.67
BG2，若 R <= 0.25
BG1，其它情况
```

代码：`constants.select_base_graph(A, R)`。

注意：

- 这里比较的是**调度码率 R**（MCS 指示的目标码率），不是接收端用
  `A/G` 反算的等效码率。二者在打孔、CRC、filler 存在时并不相等。
- 阈值是闭区间：`A=292`、`A=3824`、`R=0.67`、`R=0.25` 都归入 BG2
  分支。
- 接收端必须知道发送端用的是哪个 BG。正常流程由 A/R 推导；测试代码
  可以用 `bgn` 强制指定，但收发两侧必须一致。

### 6.2 TB CRC（TS 38.212 5.1 / 6.2.1）

```text
A > 3824 : CRC24A，L = 24
A <= 3824: CRC16， L = 16
```

`crc.py` 用逐比特 LFSR，输入/输出均为 MSB-first：

```text
gCRC16  = D^16 + D^12 + D^5 + 1
gCRC24A = D^24 + D^23 + D^18 + D^17 + D^14 + D^11 + D^10
          + D^7 + D^6 + D^5 + D^4 + D^3 + D + 1
gCRC24B = D^24 + D^23 + D^6 + D^5 + D + 1
```

校验原理：CRC 比特是 `a(D) * D^L mod g(D)`，因此把“信息 + CRC”整体
再除以 `g(D)`，余数为 0 即通过。`crc_check` 不返回原始信息；
`crc_strip` 同时返回剥离后的信息和校验结果。

### 6.3 码块分割（TS 38.212 5.2.2）

```text
Kcb = 8448 (BG1) 或 3840 (BG2)
若 B <= Kcb: C=1, Lcb=0, B'=B
否则:        Lcb=24, C=ceil(B/(Kcb-24)), B'=B+C*Lcb

K' = ceil(B'/C)
Kb = 22 (BG1)
Kb = 10/9/8/6 (BG2，按 B>640 / B>560 / B>192 / 其它)
Zc = 表 5.3.2-1 中最小的 Z，使 Kb*Z >= K'
K  = 22*Zc (BG1) 或 10*Zc (BG2)
F  = K - K'
```

**示例 1（单码块）**：`A=1000, R=0.3`

```text
L=16, B=1016, C=1, K'=1016
Kb=10（B>640），Zc=104（10*104=1040 >= 1016）
K=1040, F=24, N=50*104=5200, N_full=52*104=5408
```

**示例 2（多码块）**：`A=20000, R=0.7`（BG1）

```text
L=24, B=20024, Kcb=8448
C=ceil(20024/8424)=3, B'=20024+72=20096, K'=ceil(20096/3)=6699
Kb=22, Zc=320（22*320=7040 >= 6699）
K=7040, F=341, N=66*320=21120, N_full=68*320=21760
数据容量=3*(6699-24)=20025，补零=20025-20024=1
```

这里有三个很容易实现错的点，代码中专门做了注释和测试：

1. **`Kb` 只用于选择 `Zc`**。选出 `Zc` 后，LDPC 编码器输入长度固定为
   `K=22*Zc/10*Zc`，不是 `Kb*Zc`。BG2 小 TB 时（Kb=6/8/9）多出来的
   系统列全部是 filler。
2. **`K'` 向上取整**；当 `B` 不能被 `C` 整除时，最后一个 CB 在计算
   CB CRC 前补 0 到 `K'`，接收端拼接后只保留前 `B` 个比特，因此补零
   不会进入 TB。补零数 `= C*K' - B - C*L_cb`，一定小于 `C`。
3. **filler 位置在块尾**：完整码字中 filler 位于 `[K-F, K)`。码字前
   `2*Zc` 个系统比特被打孔，所以到 d 域后整体左移 `2*Zc`，变成
   `[K-F-2*Zc, K-2*Zc)`。`TransportBlockPlan.filler_mask` 就是按这个
   偏移生成的；如果直接把完整码字的 filler 下标用于速率匹配，会整体
   偏移 `2*Zc` 个位置，导致接收端把 filler 当数据、把数据当 filler。

### 6.4 LDPC 编码与打孔（TS 38.212 5.3.2）

- BG1：46×68 基图，`Kb=22`，码字长度 `N_full=68*Zc`；
- BG2：42×52 基图，编码器固定 10 个系统列（分割阶段的 Kb 可为
  6/8/9/10），码字长度 `N_full=52*Zc`；
- 前 `2*Zc` 个系统比特被打孔，进入速率匹配的长度为
  `N=66*Zc`（BG1）或 `50*Zc`（BG2）；
- filler 在协议中是 `NULL`，编码时按 0 处理，速率匹配时跳过。

**为什么前 4 个校验块要单独解？** 其余每一行都有一个移位为 0 的对角
校验块 `(i, Kb+i)`，可以直接得到 `p_i`；但前 4 行的校验块相互耦合，
没有“本行唯一未知”的结构。TS 38.212 的基图把前 4 行设计成双对角
形式，才能线性时间编码。

以 BG1 为例（`A_r*s` 是第 r 行信息列异或和）：

```text
row0: A0*s + P_A*p0 + p1           = 0
row1: A1*s + P_B*p0 + p1 + p2      = 0
row2: A2*s +            p2 + p3     = 0
row3: A3*s + P_A*p0          + p3  = 0
```

把 4 个方程在 GF(2) 上相加：`P_A*p0` 出现两次，`p1..p3` 各出现两次，
全部抵消，只剩

```text
P_B*p0 = A0*s + A1*s + A2*s + A3*s
```

于是先解 `p0`（用一个逆循环移位），再回代：

```text
p1 = A0*s + P_A*p0                 （row0）
p3 = A3*s + P_A*p0                 （row3）
BG1: p2 = A1*s + P_B*p0 + p1      （row1）
BG2: p2 = A1*s + p1               （row1；BG2 的 P_B 在 row2）
```

BG2 的前 4 行是：

```text
row0: A0*s + P_A*p0 + p1           = 0
row1: A1*s +          p1 + p2      = 0
row2: A2*s + P_B*p0 + p2 + p3      = 0
row3: A3*s + P_A*p0          + p3  = 0
```

相加后同样得到 `P_B*p0 = sum(A_r*s)`。代码在 `ldpc/codec.py::encode`。

**循环移位方向**：本实现固定 `P^s` 为“列下标加 s”，对长度为 Zc 的
列向量 `x` 有 `P^s x = np.roll(x, -s)`。编码、译码和校验子必须使用
同一方向；方向搞反时收发可能同时错位而在无噪声自测中“自洽”，但与
协议码字不一致，所以测试用独立的 `H*c^T=0` 校验来约束。

### 6.5 速率匹配（TS 38.212 5.4.2.1）

每个码块的长度：

```text
Gp = G / (N_L * Qm)
E_r = N_L*Qm*floor(Gp/C)   前 C - (Gp mod C) 块
E_r = N_L*Qm*ceil (Gp/C)   其余块
```

余数放在**最后** `Gp mod C` 个码块上，这是协议原文顺序；多码块时
E_r 切分必须与发送端完全一致，否则后面所有 LLR 都会错位。

`k0` 查表 5.4.2.1-2：

| rv | BG1 | BG2 |
|----|-----|-----|
| 0 | 0 | 0 |
| 1 | floor(17*Ncb/(66*Zc))*Zc | floor(13*Ncb/(50*Zc))*Zc |
| 2 | floor(33*Ncb/(66*Zc))*Zc | floor(25*Ncb/(50*Zc))*Zc |
| 3 | floor(56*Ncb/(66*Zc))*Zc | floor(43*Ncb/(50*Zc))*Zc |

未配置 LBRM 时 `N_cb=N`，于是 `k0` 简化为 `0 / 17Zc / 33Zc / 56Zc`
（BG1）和 `0 / 13Zc / 25Zc / 43Zc`（BG2）。`k0` 总是 `Zc` 的整数倍，
读取起点才会落在提升块边界上。

循环缓存中的 filler 是 `NULL`，读取时跳过；若 `E_r` 超过非 filler 的
缓存长度，则绕回重复读取。接收端把落在同一缓存地址的 LLR **相加**，
这是独立观测下的最优软合并（`rate_recover` 用 `np.bincount` 实现）。

### 6.6 比特交织（TS 38.212 5.4.2.2）

对 `E = Qm*L`：

```text
f[i + Qm*j] = e[j + L*i],  0 <= i < Qm, 0 <= j < L
```

等价于 `e.reshape(Qm, L).ravel(order="F")`。接收端用
`bit_deinterleave` 精确求逆：

```python
e = f.reshape(Qm, L, order="F").ravel(order="C")
```

交织的作用是把相邻编码比特分散到不同调制符号的相同比特位置，避免
一个符号的深衰落集中打在一段连续编码比特上。它是纯置换，不改变信息量。

### 6.7 PUSCH 加扰 / 解扰（TS 38.211 6.3.1.1）

发送端在**码块拼接之后、调制之前**对整个码字加扰：

```text
b~(i) = (b(i) + c(i)) mod 2,   i = 0 .. G-1
```

`c(i)` 是 TS 38.211 5.2.1 的 Gold 序列（31 位 m 序列，`Nc = 1600`）：

```text
x1(n+31) = (x1(n+3) + x1(n)) mod 2
x2(n+31) = (x2(n+3) + x2(n+2) + x2(n+1) + x2(n)) mod 2
c(n)     = (x1(n+1600) + x2(n+1600)) mod 2
```

`x1(0)=1`，其余 30 位为 0；`x2(i) = bit_i(c_init)`（**最低位对应
x2(0)**，不是 MSB-first）。

初值 `c_init`：

```text
c_init = n_RNTI*2^16 + n_RAPID*2^10 + n_ID        msgA on PUSCH
c_init = n_RNTI*2^15 + q*2^14 + n_ID             其它情况
```

其中 `n_ID` 为 `dataScramblingIdentityPUSCH`（未配置时用小区 ID
`N_ID^cell`），`q` 为码字号（PUSCH 单码字恒为 0）。msgA 公式的最高位
可能到 bit31，而 `x2` 只有 31 位初值，因此 `gold_sequence` 会截断低
31 位；这是协议公式与 31 位发生器的边界，不是实现丢位。

接收端解扰等价于对 LLR 做符号翻转：

```text
LLR_b(i) = (1 - 2*c(i)) * LLR_b~(i)
```

因为加扰是异或，这一步没有信息损失，且**必须在按码块切分/解交织之前
完成**：发送端的 `c(i)` 与拼接后的整个 G 长码字对齐，提前解扰才能让
每个 LLR 落在正确的编码比特上。

`scrambling.py` 提供：

- `gold_sequence(length, c_init)`：Gold 序列（与 py3gpp 的 `nrPRBS`
  交叉核对过）；
- `PUSCHScramblingConfig`：`c_init` 计算、参数范围校验、msgA 支持；
- `scramble_bits` / `descramble_llr` / `descramble_pusch_llr`。

未配置 `scrambling`（`None`）时链路跳过加扰/解扰，仅用于单元测试或
对比实验，**不是协议行为**。

### 6.8 LDPC 译码

`ldpc.decode` 实现分层归一化最小和（Layered NMS）：

- `L` 保存变量节点后验 `L = 信道 LLR + 所有校验节点消息`；
- 逐行处理：先减去该行旧消息 `R[r]`，得到干净的变量到校验消息，做
  check-node min-sum 更新，再加回 `L`；
- 每条边使用“除自身外”的最小值和次小值：若该边是最小边则取次小值
  `m2`，否则取最小值 `m1`；
- 归一化因子 `alpha`（默认 0.8），补偿 min-sum 相对 sum-product 的
  过估计；
- 每个完整迭代后用 `H*c^T=0` 检查硬判决；启用 `early_stop` 时满足
  即退出，关闭时跑满 `max_iter` 并在最后一轮报告 `converged`；
- 打孔比特初始 LLR=0（无观测），filler 初始 LLR 为大正数（已知 0）。

分层调度的关键点是“先扣旧消息再更新”：如果直接从 `L` 计算校验节点
消息而不减去 `R[r]`，第 r 行上一轮的外信息会被重复计入，译码会发散
或振荡。代码在 `ldpc/codec.py::decode` 中按行保存 `R` 并在更新前扣除。

---

## 7. JSON 模板

`app.pipeline` 把 TB 生成、发送链、信道、接收链、CRC 校验封装成一个
调用：给一个 JSON 文件，直接跑完整个流程并输出报告。

### 7.1 运行方式

```bash
# 多用例示例
python3 run_pusch.py templates/pusch_example.json

# 等价写法
python3 -m nr_pusch_ldpc_rx templates/pusch_example.json

# 只输出汇总，并把完整报告写为 JSON
python3 run_pusch.py templates/pusch_example.json --quiet --json-out report.json
```

单个文件既可以是**一个用例对象**，也可以是 `{"cases": [...]}`。命令行
输出：

```text
[1/6] bg2_kb6_single_cb  -> PASS  (BG2 Kb=6，单码块)
      A=100 R=0.3 BG2 Kb=6 Zc=20 C=1 K=200 F=84 N=1000 G=388
      Qm=2 rv=0 N_L=1 c_init=590480060
      TB CRC=PASS | CB CRC=PASS | LDPC=converged | bits=0/100 | iters=1 | 6.76 ms
...
总计 6 个用例：通过 6，失败 0
```

退出码：

| 退出码 | 含义 |
|---|---|
| `0` | 全部用例通过 |
| `1` | 链路跑通，但至少一个用例 CRC 失败或误比特 > 0 |
| `2` | 配置/文件错误（错误信息会指出 JSON 路径，例如 `cases[1].num_coded_bits`） |

### 7.2 模板字段

| 字段 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `name` | 否 | `"case"` | 用例名，报告中显示 |
| `description` | 否 | `null` | 用例说明，允许出现在顶层或单个用例中 |
| `transport_block_size` | 是 | – | TB 信息比特数 `A`（不含 CRC） |
| `target_rate` | 是 | – | 调度码率 `R`，用于 BG 选择、推导 G |
| `modulation` | 否 | `"QPSK"` | `BPSK`/`QPSK`/`16QAM`/`64QAM`/`256QAM`/`pi/2-BPSK` 或 Qm |
| `rv` | 否 | `0` | 冗余版本 `0..3` |
| `n_layers` | 否 | `1` | 传输层数 `N_L` |
| `num_coded_bits` | 否 | 由 R 推导 | 编码比特总数 `G`，必须是 `Qm*N_L` 的整数倍 |
| `base_graph` | 否 | `null` | 强制 BG1/BG2；`null` 表示按 38.212 6.2.2 规则选择 |
| `transport_block_bits.source` | 否 | `"random"` | `zeros` / `ones` / `alternating` / `random` / `explicit` |
| `transport_block_bits.seed` | 否 | `0` | `random` 的随机种子 |
| `transport_block_bits.bits` | 否 | – | `explicit` 时的 0/1 数组，长度必须等于 `A` |
| `scrambling.n_rnti` | 视情况 | – | 提供 `scrambling` 对象时必填 |
| `scrambling.n_id` | 否 | `0` | `dataScramblingIdentityPUSCH`，未配置时填小区 ID |
| `scrambling.codeword_index` | 否 | `0` | 码字号 `q` |
| `scrambling.msg_a` / `n_rapid` | 否 | `false` / `null` | msgA 加扰，`msg_a=true` 时必须给 `n_rapid` |
| `ldpc.max_iter` | 否 | `30` | LDPC 最大迭代次数 |
| `ldpc.alpha` | 否 | `0.8` | 归一化最小和因子 |
| `ldpc.early_stop` | 否 | `true` | 是否用 `H*c^T=0` 早停 |
| `channel.type` | 否 | `"none"` | `none` 或 `awgn` |
| `channel.ebn0_db` | 否 | `0` | AWGN 的 Eb/N0（dB），仅 `type=awgn` |
| `channel.seed` | 否 | `0` | AWGN 噪声种子 |

```json
{
  "name": "bg1_three_cb_rv2",
  "transport_block_size": 20000,
  "target_rate": 0.7,
  "modulation": "16QAM",
  "rv": 2,
  "num_coded_bits": 63360,
  "transport_block_bits": {"source": "random", "seed": 3},
  "scrambling": {"n_rnti": 17920, "n_id": 736},
  "ldpc": {"max_iter": 30, "alpha": 0.8, "early_stop": true},
  "channel": {"type": "none"}
}
```

说明：

- 未列出的键会被**拒绝**，避免 `target_rat` 这类拼写错误被静默忽略；
- `num_coded_bits` 缺省时按
  `G = ceil(ceil(B/R)/(Qm*N_L))*(Qm*N_L)` 推导，`B = A + L`；
- `scrambling` 为 `null` 或不给表示不加扰，只能用于对比实验；
- `channel` 为 `awgn` 时，发送端比特经 BPSK + AWGN 后转 LLR 再进接收链，
  有效码率按 `A/G` 计算，CRC 失败会体现在报告的 `failed` 中，而不是
  抛异常；
- `transport_block_bits.source=explicit` 时，`bits` 长度必须等于 `A`；
  其它来源不允许同时给 `bits`。

### 7.3 作为库调用

```python
from nr_pusch_ldpc_rx import run_config

report = run_config("templates/pusch_example.json")
print(report.passed, "/", len(report.cases))
for case in report.cases:
    print(case.name, case.passed, case.bit_error_str, case.iterations)

# 也可以直接给字典或 {"cases": [...]}
report = run_config({
    "transport_block_size": 100,
    "target_rate": 0.3,
    "scrambling": {"n_rnti": 1, "n_id": 0},
})
print(report.as_dict())   # 可直接 json.dump
```

`CaseResult.as_dict()` 包含协议派生量（BG/Kb/Zc/C/K/F/N）、`c_init`、
TB/CB CRC、LDPC 收敛情况、每码块迭代次数、误比特数和耗时。

---

## 8. Python API 示例

### 8.1 生成测试向量并接收

```python
import numpy as np

from nr_pusch_ldpc_rx import (
    PUSCHReceiverConfig,
    PUSCHScramblingConfig,
    decode_transport_block_from_bits,
    encode_transport_block,
    plan_transport_block,
)

A, R, Qm, G = 1000, 0.30, 2, 3390
rng = np.random.default_rng(1)
tb = rng.integers(0, 2, A, dtype=np.uint8)

plan = plan_transport_block(A, R)
# n_ID 未配置 dataScramblingIdentityPUSCH 时填小区 ID；msgA 另见 msg_a/n_rapid。
scrambling = PUSCHScramblingConfig(n_rnti=0x4601, n_id=321)

encoded = encode_transport_block(
    tb, G, qm=Qm, plan=plan, scrambling=scrambling
)

cfg = PUSCHReceiverConfig(A, R, modulation="QPSK", scrambling=scrambling)
result = decode_transport_block_from_bits(encoded.coded_bits, cfg, plan=plan)

print(result.plan)        # BG2, Kb=10, Zc=104, C=1, K=1040, F=24 ...
print(encoded.scrambled)  # True：coded_bits 已加扰
print(result.tb_crc_ok)   # True
print(result.cb_crc_ok)   # True
```

### 8.2 从解调 LLR 开始

```python
from nr_pusch_ldpc_rx import decode_transport_block

# llr 长度必须为 G；正数 -> bit 0，负数 -> bit 1
# cfg.scrambling 不为 None 时，内部先按同一 Gold 序列解扰。
result = decode_transport_block(llr, cfg)
tb_hat = result.tb_bits
```

### 8.3 分层调用（协议原语）

```python
import numpy as np

from nr_pusch_ldpc_rx.chain import (
    coded_bit_lengths,
    crc_attach,
    make_code_blocks,
    plan_transport_block,
    rate_match,
)
from nr_pusch_ldpc_rx.ldpc import encode

A, G = 20000, 63360
tb = np.random.default_rng(3).integers(0, 2, A, dtype=np.uint8)

plan = plan_transport_block(A, 0.7)            # BG1, C=3
b = crc_attach(tb, plan.tb_crc)                # 长度 B
cbs = make_code_blocks(tb, plan)               # (C, K)，含 filler=0
cw = encode(cbs[0], plan.bgn, plan.z)          # 完整码字，长度 N_full
d = cw[2 * plan.z:]                            # 打孔，长度 N
e_lengths = coded_bit_lengths(G, plan.c, qm=4) # 每个 CB 的 E_r
f0 = rate_match(d, int(e_lengths[0]), rv=2, qm=4,
                bgn=plan.bgn, z=plan.z,
                filler_mask=plan.filler_mask)
```

### 8.4 msgA（Type-2 随机接入）加扰

```python
from nr_pusch_ldpc_rx import PUSCHScramblingConfig

scr = PUSCHScramblingConfig(
    n_rnti=0x2A1,      # RA-RNTI
    n_id=9,            # msgA-DataScramblingIndex（未配置时用小区 ID）
    msg_a=True,
    n_rapid=17,        # 0..63
)
print(scr.c_init)      # n_RNTI*2^16 + n_RAPID*2^10 + n_ID
```

### 8.5 多码块结果

```python
for cb in result.code_blocks:
    print(cb.index, cb.e_len, cb.ldpc.iterations,
          cb.ldpc.converged, cb.crc.crc_present, cb.crc.crc_ok)
```

`result` 还提供 `cb_crc_ok`（所有 CB CRC 是否通过）和
`all_ldpc_converged`（所有 CB 是否满足校验方程）。

---

## 9. 测试覆盖

运行：

```bash
python3 -m unittest discover -s tests -v
```

当前共 **98 个测试**，覆盖：

- CRC16/24A/24B 已知向量、单比特错误、附加/剥离；
- lifting size 表 51 个 Z 的完整性和唯一性；
- 基图尺寸、表项、移位方向、`H*c^T=0` 校验；
- **两个 BG、全部 51 个 Zc** 的编码和打孔后无噪声译码；
- Kb=6/8/9/10 边界（B=192/193、560/561、640/641）；
- 单码块/多码块、CB CRC 错误检测、末块补零、TB 拼接；
- rv=0/1/2/3，E_r 分配，循环缓存重复合并，比特交织/解交织；
- N_L=1/2 的层数配置；
- QPSK/16QAM/64QAM/256QAM/BPSK；
- AWGN 软 LLR 下 LDPC 译码与完整接收链；
- Gold 序列：多种 `c_init`、前缀性质、与逐位参考实现逐比特一致；
- 加扰参数：`c_init` 公式（含 q 和 msgA）、参数范围校验；
- 加扰/解扰自反性、LLR 符号翻转、RNTI/n_ID 错误时必须 CRC 失败；
- msgA（RA-RNTI + `n_RAPID`）端到端；
- 多组端到端配置：BG1/BG2、单/多码块、Zc=5..384；
- 早停关闭时仍正确报告收敛；
- JSON 模板：默认值、单/多用例、未知键拒绝、类型与范围校验、显式比特、
  G 的整数倍约束、AWGN 失败上报、报告 `as_dict` 可序列化；
- CLI：示例模板、`--json-out`、`--quiet`、退出码 0/1/2；
- 两个模板文件本身的端到端运行。

---

## 10. 常见问题与实现取舍

### 10.1 为什么 LLR 正数表示 0？

这是通信链路中常用的约定，与 BPSK 映射 `0 -> +1, 1 -> -1` 配套：
`L = ln P(b=0|y)/P(b=1|y)`，所以正 LLR 倾向 0。若你的解调器输出相反
符号，请在调用 `decode_transport_block` 前取反，否则 CRC 必然失败。

### 10.2 为什么 filler 用 `1e4` 而不是 `inf`？

filler 是协议中的 `NULL`，表示“已知为 0”。用有限大正值（`FILLER_LLR`）
可以让译码器把它当作非常可靠的 0，同时避免 `inf` 在 min-sum 的
`inf - inf` 或乘加中产生 `nan`。`decode` 也会显式拒绝非有限 LLR。

### 10.3 为什么全零码字能通过 CRC？

CRC 是线性校验，对全零码字没有检错能力：如果 LDPC 恰好收敛到全零
码字，CB/TB CRC 都会通过。这是 CRC 的性质，不是实现错误。实际系统
可以通过加扰、已知比特或上层协议进一步区分；本项目的 `run_case` 还会
额外比较原始 TB，因此报告不会把这种情况误判为 PASS。

### 10.4 为什么用 NMS 而不是 sum-product？

NMS 只使用最小/次小值和符号，复杂度低、适合硬件实现，性能略低于
精确 sum-product。`alpha` 用于补偿过估计，默认 0.8；可以在
`PUSCHReceiverConfig` 或 JSON 的 `ldpc.alpha` 中调整。

### 10.5 没有实现哪些协议特性？

- **CBGTI**：默认所有码块都被调度；
- **LBRM**：接收端默认 `N_cb=N`；`rate_matching` 底层函数保留 `n_cb`
  参数，可用于有限缓存实验；
- **UCI 复用**：只处理纯 UL-SCH，不建模 `x`/`y` 占位比特；
- **DMRS/信道估计/均衡/解映射**：接收链从 LLR 开始；
- **调制/层映射/资源映射**：发送链输出编码比特。

### 10.6 为什么 JSON 模板不能模拟“RNTI 配错”？

模板运行时发送端和接收端使用同一份配置，因此两侧参数天然一致。要
模拟“接收端 RNTI/n_ID 配错”“发送端多码块但接收端按单码块解”等不一致
场景，请直接调用 `encode_transport_block` / `decode_transport_block`
并分别传入不同参数，参考 `tests/test_end_to_end.py` 与
`tests/test_pipeline.py`。

### 10.7 G 为什么必须是 `Qm*N_L` 的整数倍？

TS 38.212 5.4.2.1 的 `E_r` 以 `Qm*N_L` 为单位分配，且要求
`sum(E_r)=G`。若 G 不是整数倍，就无法在不破坏调制符号/层边界的前提
下把 G 均分到 C 个码块。模板省略 `num_coded_bits` 时会自动向上取整。

### 10.8 从 1.0 升级：模块路径变了

1.1.0 按功能把原来的扁平模块移到了子包，**包级 API 不变**：

| 旧路径 | 新路径 |
|---|---|
| `nr_pusch_ldpc_rx.crc` | `nr_pusch_ldpc_rx.chain.crc` |
| `nr_pusch_ldpc_rx.segmentation` | `nr_pusch_ldpc_rx.chain.segmentation` |
| `nr_pusch_ldpc_rx.rate_matching` | `nr_pusch_ldpc_rx.chain.rate_matching` |
| `nr_pusch_ldpc_rx.scrambling` | `nr_pusch_ldpc_rx.chain.scrambling` |
| `nr_pusch_ldpc_rx.base_graph` | `nr_pusch_ldpc_rx.ldpc.base_graph` |
| `nr_pusch_ldpc_rx.ldpc`（模块） | `nr_pusch_ldpc_rx.ldpc`（包，函数同名） |
| `nr_pusch_ldpc_rx.receiver` | `nr_pusch_ldpc_rx.link.receiver` |
| `nr_pusch_ldpc_rx.transmitter` | `nr_pusch_ldpc_rx.link.transmitter` |
| `nr_pusch_ldpc_rx.pipeline` | `nr_pusch_ldpc_rx.app.pipeline` |

`from nr_pusch_ldpc_rx import encode_transport_block, decode_transport_block,
run_config` 等常用导入不受影响。

### 10.9 性能与适用场景

这是**参考实现**，优先保证可读性和逐比特可核对：

- CRC 和 Gold 序列用 Python 循环逐位实现，几十万比特规模足够快；
- LDPC 译码按基图行向量化（每个 Zc 块一次 NumPy 操作），适合算法
  验证和中小规模链路级仿真；
- 若需要大规模 BLER 曲线，建议把译码器换成 C/GPU 实现，或对
  `rate_matching`/`ldpc` 的块操作进一步向量化（例如批量处理多个 TB）。

---

## 11. 参考规范

- 3GPP TS 38.212 V18.3.0：Multiplexing and channel coding
  - 5.1 CRC calculation
  - 5.2.2 Transport block CRC attachment / code block segmentation
  - 5.3.2 LDPC coding
  - 5.4.2 Rate matching
- 3GPP TS 38.211 V18.3.0：Physical channels and modulation
  - 5.2.1 Pseudo-random sequence generation
  - 6.3.1.1 Scrambling

基图表来自 TS 38.212 Table 5.3.2-2/3；实现以对应版本的 3GPP 规范为准。
