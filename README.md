# AFL × VLM 异步联邦指令微调实验框架

研究假设：**在异步联邦 VLM 指令微调中，不同客户端承担不同任务 → 到达顺序不同；
聚合顺序（而非陈旧度）决定任务知识写入全局模型的次序**——age-based 聚合只能
re-weight、不能 re-order（详见 `related-work-audit.md`）。

Benchmark 锚定 **Pilot/FedMIT（AAAI-25, arXiv:2501.13985）** 的任务轴：9 客户端
3/3/3 任务异构（GQA / OCR-VQA / COCO Caption，每任务 3 客户端持互斥子集）；
时间轴来自本框架（五个正交延迟/顺序旋钮）。实验体系：V0–V3 前置验证 →
E1 master run → E2/E3 机制分析 → E5 baseline 矩阵，见「实验流程总览」。

本框架把"聚合顺序"做成服务器侧的显式可控变量：

```
客户端线程(×N，并发)                    服务器（主线程）
┌─────────────────────────┐            ┌────────────────────────────────────┐
│ 租一张空闲卡(每卡一份副本) │   上传     │ 到达队列 → arrival buffer           │
│ 同步全局 LoRA (v_t)      │ ────────→ │        ↓                            │
│ 本地 SFT（任务数据）      │  (delta)  │ AggregationPolicy.on_arrivals       │
│ 模拟延迟: speed_factor   │           │   → 返回 [更新列表]，列表顺序=施加顺序 │
│   + net_delay 分布       │           │        ↓                            │
│ 到达时间 = 训练耗时+延迟  │           │        ↓                            │
└─────────────────────────┘            │ 按顺序逐个施加 w ← w + lr·α·Δ_i      │
                                       │ （顺序施加 → 非交换效应真实存在）      │
                                       │ 每 G 批: 分任务 eval → evals.jsonl   │
                                       └────────────────────────────────────┘
```

## 目录

```
afl_vlm/
  config.py            配置与校验（支持 base: 递归继承）
  planner.py           虚拟时钟调度器（timing: virtual 的阶段1：离线生成调度表）
  server.py            AsyncServer（到达→策略→顺序化聚合；virtual 模式=计划回放器；
                       I3 delta 级探针插桩 probe_on_apply）
  client.py            ClientThread（本地 LoRA SFT + 延迟模拟 / 按计划同步指定版本；
                       FedProx 近端项 prox_mu）
  device_pool.py       设备解析/租借
  logging_utils.py     JSONL 日志
  aggregation/         策略: immediate_fifo / fedbuff / staleness_weighted /
                       fedasync / fedcompass / sync_avg / fixed_order
  models/              Qwen2.5-VL 适配器 + tiny_mock（CPU 冒烟）+ 共享模型管理器
  data/                任务注册表: local_json / hf_hub / synthetic；
                       client_view() 任务内互斥切分（Pilot 协议 data_partition: disjoint）
  evaluation/          分任务 eval + 遗忘分析 + 探针集（probe manifest 对齐）
scripts/
  run_train.py         主入口
  prepare_datasets.py  Pilot 协议三任务数据准备（GQA/OCR-VQA/COCO → LLaVA json + 探针清单）
  run_vseries.py       V0–V3 前置实验一键驱动（确定性/标定/到达结构性/矢量staleness/killer）
  run_e5.py            E5 baseline 矩阵一键驱动（9 策略 × 多种子 → 汇总表）
  run_killer.py        killer experiment 驱动（同 delta 集合、只变施加顺序）
  analyze_logs.py      日志分析/绘图
  analyze_structure.py V1 到达结构性分析（游程置换检验/块长/震荡幅度/遮蔽比）
  analyze_vector.py    V2 矢量 staleness 分析（CI 与 τ 脱钩/同 τ 不同命/离开期分组）
configs/
  v0_smoke / v0_calibration / v1_natural / v1_shuffled / v1_noise
  e1_natural           E1 master run（= v1_natural 拉长 12 轮）
  e5_*                 E5 baseline 矩阵 ×9
  killer/base_tiny / killer/base_pilot（+ 旧版 killer/base、order_*.yaml）
  smoke_cpu / smoke_virtual / qwen7b_local / qwen7b_hub / qwen7b_staleness / qwen7b_virtual
```

## 服务器部署

```bash
conda create -n aflvlm python=3.10 -y && conda activate aflvlm
pip install -r requirements.txt

# 冒烟测试（CPU、不下载任何权重，~1 分钟）
python -m scripts.run_train --config configs/smoke_cpu.yaml
```

真模型前先 `--dry-run` 检查实验计划：

```bash
python -m scripts.run_train --config configs/v1_natural.yaml --dry-run
```

## 实验流程总览（从零到 baseline 矩阵）

```bash
# ① 数据准备：Pilot 协议三任务（GQA / OCR-VQA / COCO Caption）→ data/pilot/
#    （HF 仓库 ID 服务器端在线验证，候选失败会打印搜索结果 + 支持 --dataset-id-* 指定）
python -m scripts.prepare_datasets --per-task 12000

# ② V 系列前置验证（V0 确定性+标定 → V1 到达结构性 → V2 矢量 staleness → V3 killer）
python -m scripts.run_vseries --stage all            # 无 GPU 先跑机制链: --skip-calibration
python -m scripts.run_vseries --stage v1 --seeds 42,43 --noise --plot

# ③ E1 master run（= v1_natural 拉长 12 轮，108 条更新）
python -m scripts.run_train --config configs/e1_natural.yaml

# ④ E5 baseline 矩阵：9 策略 × 3 种子 → runs/e5_summary.csv
python -m scripts.run_e5 --seeds 42,43,44
```

各阶段产出 `runs/<stage>_report.json`（go/no-go 判定）与 `runs/e5_summary.csv`
（final 能力 / 稳态震荡 amp / 慢任务权重占比 / 平均 staleness）。

## V 系列前置实验（go/no-go）

| 阶段 | 问题 | 判据 | 数据 |
|---|---|---|---|
| **V0a** smoke | 调度表确定性 | 同 config+seed 两次 `plan.json` 逐字节一致 | tiny_mock ×2 |
| **V0b** 标定 | 任务时长差真实存在 | 三任务单轮耗时 最快/最慢 ≥1.5，产出建议 task_profiles | 7B wallclock 2 轮 |
| **V1** 到达结构性 | 任务→时长 → 到按任务成块，能力震荡 | 游程置换 p<0.05、块长比 ≥2、震荡幅度比 ≥2、遮蔽比 M≪1 | natural vs shuffled（拉丁方）(± noise 臂) |
| **V2** 矢量 staleness | CI（纠偏指数）与 τ 脱钩、与离开期漂移方向挂钩 | Spearman(τ,\|CI\|)≈0 而 Spearman(‖d‖,\|CI\|)>0；同 τ 反向对；离开期分组 MW | delta_probes.jsonl |
| **V3** killer 直接层 | 冻结 delta、只变施加顺序 → 中途态发散 | 中途态离散 / 终点重合残差 ≥10（纯加法理论下终点应重合） | run_killer × 3 顺序 |

臂 B（`v1_shuffled`）用拉丁方把 600/700/1200s 打乱到 9 个客户端、边缘分布不变、
打断"任务→时长"因果链接——natural 与 shuffled 的全部差异只能来自到达结构。

## E5 baseline 矩阵（只换聚合，其余全同 e1_natural）

| 配置 | 策略 | 对应文献 |
|---|---|---|
| `e5_immediate_fifo` | 到达即施加、等权（零干预异步） | FedAsync 等权形态 |
| `e5_fedasync` / `e5_fedasync_l05` | staleness 降权 exp(−λτ)，λ=0.3/0.5 | FedAsync (1903.03934) |
| `e5_fedbuff_k4` | 缓冲 K=4 | FedBuff (2106.06639) |
| `e5_fedcompass` | 时间变化 staleness 阈值 θ(t) 准入 | FedCompass (2309.14675) |
| `e5_sync_fedavg` | 同步屏障锚（等齐 9 客户端） | FedAvg |
| `e5_sync_fedprox` | 同步 + 客户端近端项 μ=0.01 | FedProx |
| `e5_random_order` | 同批等权、批内随机序（fixed_order） | 顺序效应对照 |
| `e5_fullmix` | K=9 全混合 | ACE 式任务混合 |

核心论点：以上全部是**标量**干预（时间/缓冲/同步），都压不住**任务方向性**错位
——E5 的 amp_ocr / wshare_ocr 列就是证据表。

## 三步走

1. **smoke**（tiny_mock + synthetic，CPU）：验证线程/延迟/聚合顺序/日志/eval 全链路。
2. **单任务真模型**（`qwen7b_local.yaml` 里先只留一个任务）：验证 LoRA 训练与显存。
3. **完整实验**：任务异构客户端 + killer experiment。

## 数据准备

### Pilot 协议三任务（`scripts/prepare_datasets.py`）

```bash
python -m scripts.prepare_datasets                    # 默认 12000 条/任务 → data/pilot/
python -m scripts.prepare_datasets --per-task 6000 --only ocr   # 小规模/单任务
python -m scripts.prepare_datasets --dataset-id-ocr <repo>      # 手动指定 HF 仓库
```

- 任务源（按候选顺序自动验证可用性，失败时打印 HF 搜索结果）：
  vqa=`lmms-lab/GQA`、ocr=`HuggingFaceM4/OCR-VQA`（备选 `lmms-lab/OCR-VQA`）、
  caption=`HuggingFaceM4/COCO`（备选 coco-karpathy 等）；
- 流式拉取 + 列名自动探测（图片列/问答列，list 型问答逐条展开），落盘 LLaVA 格式
  `data/pilot/{vqa,ocr,caption}.json` + `data/pilot/images/`；
- 尾部 10% 作 held-out eval；训练部分由 `data_partition: disjoint` 随机均分给
  每任务 3 个客户端（互斥子集，种子来自 `experiment.seed`）——照抄 Pilot/FedMIT；
- 同时产出 `probe_manifest.json`：每任务 128 条探针索引，与运行时
  `max_eval: 256` 严格对齐（`probe_on_apply` 的 I3 插桩数据源）。
  **注意**：改配置里的 `max_eval` 时必须同步改 `--max-eval` 重跑准备脚本。

### local_json（LLaVA 标准格式）

```json
[
  {"id": "vqa_0001", "image": "COCO_train2014_00000123.jpg",
   "conversations": [
     {"from": "human", "value": "<image>\nWhat is the man doing?"},
     {"from": "gpt", "value": "The man is riding a horse."}
   ]}
]
```

- `image_root` 指向图片目录；无图任务可省略 `image` 字段（纯文本指令）。
- 同一任务数据集内要么全有图要么全无图（v1 不支持混批）。
- 任务名 → 数据集在 `tasks:` 里配，客户端通过 `task_mix: {vqa: 4, ocr: 3, caption: 3}` 认领任务。

### hf_hub

见 `configs/qwen7b_hub.yaml`：`filter_column + filter_keywords` 按列关键字切任务子集
（LLaVA-Instruct-150K 用 `id` 前缀）；`image_root` 指向 COCO 解压目录。

## 延迟模型（对齐主流 AFL 模拟的三种范式）

现有 AFL 代码设置客户端延迟的方式可归为三类，本框架实现了前两类 + 延迟作用点旋钮，
`clients.delay_mode` 切换：

| 范式 | 代表代码 | 本框架实现 |
|---|---|---|
| **墙钟模拟**：持久算力属性 + 延迟分布 + 到达时间戳 | FedASMU / FedCompass / FedFa | `delay_mode: wallclock`（默认）：`speed_factor` 补睡 + `net_delay` 上传睡眠（每轮重采样，fixed/uniform/lognormal/exponential） |
| **陈旧度注入**：延迟以服务器聚合**步数** τ 计量 | FedAsync 官方模拟 | `delay_mode: staleness`：每轮采样 `staleness_tau`，服务器把更新**持留**到版本推进 `version_trained_on + τ` 才进入 buffer（不睡眠；配 `staleness_weighted` 聚合即还原 α^τ 设定） |
| **真实并发/轨迹回放** | FedBuff 官方 / FedScale | 真实线程已有；轨迹回放 v1 不做 |

延迟**作用点**（正交旋钮）：下载侧 `download_lag`（客户端同步到旧版快照训练）、
计算侧 `speed_factor`、上传侧 `net_delay`、服务器侧 τ 持留。

**任务相关延迟** `clients.task_profiles`：按任务覆盖速度/网络/τ。现有模拟器的延迟
普遍与任务无关——这正是顺序效应从未被注意到的原因之一；真实部署中任务-时间强相关
（OCR 慢、业务时段同步）。开启后"任务聚集到达"成为延迟模型的**自然涌现**，
与 `fixed_order` 的人为重排（因果操纵）构成两条互补证据链：

```yaml
clients:
  task_profiles:
    vqa:     {speed_factor: 1.0, net_delay: {dist: lognormal, mu: 0.0, sigma: 0.4}}
    ocr:     {speed_factor: 0.5, net_delay: {dist: lognormal, mu: 0.8, sigma: 0.5}}
    caption: {speed_factor: 0.9, net_delay: {dist: lognormal, mu: 0.5, sigma: 0.5}}
```

staleness 模式完整示例见 `configs/qwen7b_staleness.yaml`。优先级：
全局默认 < `task_profiles` < `overrides`。

## 虚拟时钟模式（timing: virtual）——"计划–执行"两阶段

wallclock 模式的时间线由真实线程竞争产生：客户端数 > 卡数时，等卡的客户端
被迫晚启动，到达时间线被排队失真污染，且卡竞争的胜者由线程调度而非种子决定
（到达顺序不可复现）。`timing: virtual` 把**计时**与**执行**彻底解耦：

```
阶段1（planner.build_plan，秒级、零训练）          阶段2（真实训练，按表回放）
┌────────────────────────────────┐          ┌─────────────────────────────────┐
│ 虚拟时钟离散事件模拟：            │          │ 客户端线程:                      │
│  start_offset                  │          │  wait_state_of_version(计划版本)  │
│  + train_seconds/speed_factor  │  TrainPlan│   → 同步指定版本快照 → 真实训练    │
│  + net_delay (+ staleness τ)   │ ────────→ │  → delta → 提交（seq_no=计划号）  │
│  → 每轮同步哪个版本 / 到达顺序    │          │ 服务器: 就绪表收 delta            │
│  → 策略在元数据上跑出施加顺序/权重 │          │   → 按计划序施加，delta 未产出就等 │
└────────────────────────────────┘          │   （只拉伸墙钟，绝不重排）          │
                                            └─────────────────────────────────┘
```

- **时间线是 config+seed 的纯函数**：start_offset + train_seconds/speed_factor +
  net_delay（+ τ 持留）。卡数、卡排队、线程竞争对时间线零影响——客户端数超过
  卡数不再失真；同配置重跑得到同一条到达时间线。
- **客户端按计划同步"指定版本"**（不再是"最新版"）：管理器按计划保留会被同步的
  版本快照（`sync_consumers`），消费完即裁剪——内存有界。
- **服务器是计划回放器**：策略行为已在阶段1 完整确定（同一策略跑在计划元数据上），
  运行时只按计划序施加，delta 未产出就等。delta 早于/晚于计划虚拟时刻产出都无影响。
- **strict 可复现**：killer experiment 的"全员同步 v0"从依赖到达竞态变成计划保证
  （fixed_order 攒满才聚合 → 计划里版本恒为 0）。
- train_seconds 标定：先用 wallclock 跑一轮，取 `arrivals.jsonl` 里 train_seconds
  的每任务最大值，或直接取保守值（≥ 单卡真实单轮耗时）；取小了只是服务器多等。
- `time_scale` 在 virtual 模式无意义（无睡眠）；`speed_factor > 1` 允许（虚拟时间
  可压缩）。`--dry-run` 会打印完整调度表（到达顺序 + 施加批次）。

配置模板：`configs/qwen7b_virtual.yaml`（真模型）、`configs/smoke_virtual.yaml`
（CPU 冒烟，先跑这个验证两阶段链路）。

## 配置速查

| 字段 | 说明 |
|---|---|
| `model.device` | `auto` = 全部可见卡各放一份模型副本（一客户端一空闲卡，真并发）；`cuda:0` = 单卡串行；`cpu` = CPU |
| `experiment.timing` | `wallclock`（默认，真实线程+补睡模拟）/ `virtual`（两阶段"计划–执行"，时间线=配置的纯函数，可复现、无排队失真） |
| `experiment.time_scale` | 缩放 start_offset/net_delay 的真实睡眠（0.1 = 十倍速过延迟），不影响训练本身；virtual 模式忽略 |
| `clients.train_seconds` | 虚拟模式的单轮训练秒数（timeline 驱动值，≥ 单卡真实耗时；task_profiles/overrides 可覆盖；仅 virtual 必需） |
| `clients.delay_mode` | wallclock（睡眠模拟）/ staleness（τ 步数注入，FedAsync 式） |
| `clients.speed_factor` | (0,1]：0.5 = 慢一倍（补睡模拟）；>1 自动退化为 1 |
| `clients.net_delay` | fixed / uniform(low,high) / lognormal(mu,sigma) / exponential(rate)，每次上传采样（仅 wallclock 模式） |
| `clients.staleness_tau` | τ 分布：fixed / uniform_int(low,high)，单位 = 聚合步（仅 staleness 模式） |
| `clients.download_lag` | 下载侧延迟：训练所用版本落后最新版的步数（0=关；自动启用版本历史） |
| `clients.task_profiles` | 按任务覆盖 speed_factor/net_delay/staleness_tau/download_lag/start_offset |
| `clients.overrides` | 逐客户端覆盖 task/speed/start_offset/net_delay/num_rounds/delay_mode/train_seconds/prox_mu 等 |
| `clients.data_partition` | `shared` = 同任务客户端共用全量（旧行为）/ `disjoint` = 任务内随机均分互斥子集（Pilot 协议） |
| `clients.prox_mu` | FedProx 近端系数（>0 时本地损失加 μ/2·‖θ−θ_global‖²） |
| `server.aggregation.name` | immediate_fifo / fedbuff / staleness_weighted / fedasync / fedcompass / sync_avg / fixed_order |
| `server.eval_every_batches` | 每 N 个聚合批次做一次分任务 eval（0=关） |
| `server.gen_every_batches` | 生成评估独立频率（0=跟随 eval_every_batches）——loss 评估每批、生成评估降频 |
| `server.probe_on_apply` | I3 插桩：每施加一条 delta 前后测三任务探针 loss → `delta_probes.jsonl` |
| `server.probe_manifest` | 探针索引清单（prepare_datasets 产出）；缺省用 eval 集前 `probe_n` 条 |

## 聚合策略（研究钩子）

策略唯一职责：`on_arrivals(buffer, state) → [(更新, 权重)]`，**列表顺序 = 施加顺序**。

| 策略 | 顺序 | 权重 | 用途 |
|---|---|---|---|
| `immediate_fifo` | 到达序 | 1.0 | 经典 FedAsync 等权形态（E1/V1 主策略） |
| `fedbuff` | 到达序（攒 K 个） | 1.0 | FedBuff 基线（E5；K=9 即 fullmix） |
| `staleness_weighted` | 到达序（攒 K 个） | exp(−λ·staleness) | **re-weighting 天花板**基线 |
| `fedasync` | 到达序（不缓冲，立即逐条） | max(min_w, exp(−λ·staleness)) | FedAsync 标准形态（E5） |
| `fedcompass` | 阈值准入（staleness ≤ θ(t)），攒够 min_count 整批 | 1.0 | FedCompass 策略层近似（E5） |
| `sync_avg` | 攒齐全部客户端整批，到达序 | 1.0 | 同步 FedAvg 屏障锚（E5；+prox_mu 即 FedProx） |
| `fixed_order` | random/clustered/alternating/blocked/explicit | 恒 1.0 | **顺序效应执行器**（killer/V3/E5-random_order） |

新增策略：`aggregation/` 下新文件 → `aggregation/__init__.py` 注册 → `config.py` known_agg 加名字。

## Killer experiment（顺序 ≠ 年龄 的核心实验）

```bash
# tiny 快跑版（CPU，秒级）；正式版用 configs/killer/base_pilot.yaml（7B + Pilot 数据）
python -m scripts.run_killer --config configs/killer/base_tiny.yaml \
    --orders clustered,alternating,random --out runs/v3/killer
```

设计保证跨顺序严格可比：
- 所有客户端 `num_rounds: 1` → 全员基于**同一个初始全局模型**产出 delta 集合；
- 种子固定 → 数据采样/初始化完全一致；
- 权重恒为 1.0 → 批次内唯一自由度 = **施加顺序**；
- 每个 order 独立子进程 → 输出在 `runs/killer/<order>/`，汇总写 `killer_summary.csv`。

读结果：

```bash
python -m scripts.analyze_logs --dir runs/killer/clustered --plot
```

`aggregations.jsonl` 的 `order_index` 字段就是每次批次的施加顺序（论文核心数据）；
`evals.jsonl` 给出各任务能力随聚合阶段的曲线；`forgetting_summary` 输出峰值−终值遗忘。

## 日志（论文图表的原始数据）

| 文件 | 内容 | 关键字段 |
|---|---|---|
| `arrivals.jsonl` | 每次上传 | client/task/round/version_trained_on/t_arrival/net_delay/train_seconds + **tau/hold_until_version/download_lag** + t_submit_real |
| `aggregations.jsonl` | 每次施加 | **batch/order_index**/staleness/weight/global_version_after/t_apply |
| `evals.jsonl` | 分任务评估曲线 | tag/task/global_version/eval_loss/match_rate（+avg_answer_len 行为场） |
| `delta_probes.jsonl`（probe_on_apply） | 每条 delta 施加前后的三任务探针 loss（V2/E2 数据源） | client/task/staleness/weight/version_trained_on/version_before/version_after/loss_before/loss_after |
| `plan.json`（仅 virtual） | 完整调度表（时间线的权威记录） | rounds（每轮 sync_version/虚拟时刻）/ applies（施加顺序/权重/批次）/ arrival_order |

wallclock 模式下 tau=0；staleness 模式下 arrivals 记录提交时刻，
真正进入 buffer（被策略看见）是在版本推进到 `hold_until_version` 时。
virtual 模式下 t_arrival/t_apply/train_seconds 均为**虚拟时刻**（计划值，
可复现的时间线）；t_submit_real / t_apply_real 是真实时刻，仅作执行对照。

## 新增模型 / 任务

- 模型：`models/` 下新建 `XxxAdapter(BaseModelAdapter)`（实现 load/trainable_state/
  load_trainable_state/module_groups）→ `models/__init__.py` MODEL_REGISTRY 注册 →
  `config.py` known_models 加名字。模块分组按参数名子串匹配（vision/connector/llm 三组
  只是日志口径，connector 不做特殊化）。
- 任务：`tasks:` 里加一项即可，无需改代码。

## 设计取舍与已知限制（v1）

1. **单进程多线程 + 每卡一个模型副本**：所有客户端是同一进程内的真实线程；
   `device: auto` 时**每张可见卡放一份模型副本**，客户端训练前从设备池租一张
   空闲卡、把全局最新 LoRA 状态载入该卡副本再训练——多卡训练真并发（CUDA
   kernel 释放 GIL），聚合仍在服务器单点顺序进行。`cuda:0`/`cpu` 则退化为单卡
   串行（排队等同一副本），异步语义在两种模式下完全一致（到达时间戳 + 版本号
   + 聚合调度）。副本初始化为 CPU 母版 deepcopy，CPU 内存峰值 ≈ 2× 模型大小。
   **注意**：wallclock 模式下客户端数 > 卡数时，等卡客户端被迫晚启动 → 到达
   时间线被排队失真污染、且不可复现（卡竞争胜负由线程调度决定）——需要严格
   时间线时请用 `timing: virtual`（该模式下卡排队对时间线零影响）。
2. **delta 常驻 CPU**（float32）：10 客户端 × LoRA(r=16) ≈ 每个 ~100MB，注意内存。
   virtual 模式另有版本快照表：同时保留"仍会被计划同步"的版本（≤ 不同 sync
   版本数 × 每份 LoRA 状态），用完即裁剪。
3. `speed_factor > 1` 在 wallclock 模式无法压缩真实计算时间，自动退化为 1
   （打印警告）；virtual 模式允许 >1（虚拟时间可压缩）。
4. 评估租一张空闲卡跑（与其它卡上的训练并行）；全部卡忙时等待。正式实验建议
   `eval_every_batches` 不要太小。virtual 模式下评估不占虚拟时间线（真实侧活动）。
5. 任务内数据划分由 `clients.data_partition` 控制：`shared` = 同任务客户端共用
   完整任务数据；`disjoint` = 随机均分互斥子集（Pilot 协议，正式实验用这个）。
   Dirichlet 式非均匀切分后续可加。
6. staleness 模式下，模拟结束时仍在持留（τ 未到期）的更新会被放行并聚合
   （服务器打印提示）；分析时可用 `aggregations.jsonl` 里偏大的 staleness 值识别。
   virtual 模式的等价放行在计划生成时完成（policy.on_finish）。
7. `download_lag > 0` 时管理器自动保留版本历史（每版一份 LoRA 状态，CPU 内存），
   `required_history` 按 τ 上界自动 sizing；历史不足会报错并提示加大。
   virtual 模式改用计划驱动的快照保留（`sync_consumers`），无需 sizing。
