"""配置定义、YAML 加载、校验与客户端列表展开。"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


def stable_seed(*parts: Any) -> int:
    """跨进程稳定的哈希种子（内置 hash() 每次进程启动会变化，不能用于可复现实验）。"""
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha1(raw).digest()[:4], "little")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    include_visual: bool = False   # 是否给 vision tower 也挂 LoRA（默认只训 LLM 侧）


@dataclass
class ModelConfig:
    name: str = "qwen2.5-vl-7b"          # 模型注册表里的名字
    hf_id: Optional[str] = None           # 覆盖注册表默认的 HF id
    dtype: str = "bf16"                   # bf16 / fp16 / fp32
    device: str = "auto"                  # auto / cuda / cuda:0 / cpu
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    gradient_checkpointing: bool = False


@dataclass
class TaskSource:
    type: str = "synthetic"               # local_json / hf_hub / synthetic
    # local_json
    json_path: Optional[str] = None
    image_root: Optional[str] = None
    # hf_hub
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None    # HF dataset config 名（可选）
    split: str = "train"
    image_column: str = "image"
    question_column: str = "question"
    answer_column: str = "answer"
    filter_column: Optional[str] = None   # 按某列关键字过滤（如 LLaVA 的 id 前缀）
    filter_keywords: list[str] = field(default_factory=list)
    # synthetic
    num_samples: int = 64
    num_classes: int = 4


@dataclass
class TaskConfig:
    name: str
    source: TaskSource = field(default_factory=TaskSource)
    eval_ratio: float = 0.1               # 尾部样本作 held-out 评估集
    max_train: Optional[int] = None
    max_eval: Optional[int] = None


@dataclass
class DelayConfig:
    """延迟/陈旧度分布。两种模式下的语义（见 ClientsConfig.delay_mode）：

    - wallclock 模式：net_delay 单位 = 模拟秒（睡眠时长乘 time_scale）
      fixed / uniform(low,high) / lognormal(mu,sigma) / exponential(rate)
    - staleness 模式（FedAsync 官方做法）：延迟以服务器聚合**步数** τ 计量，
      staleness_tau 用 fixed(value) / uniform_int(low,high)；download_lag 同理
    """
    dist: str = "lognormal"               # fixed / uniform / uniform_int / lognormal / exponential
    value: float = 0.0                    # fixed 的值
    low: float = 0.0                      # uniform / uniform_int 下界
    high: float = 1.0                     # uniform / uniform_int 上界
    mu: float = 0.0                       # lognormal mu
    sigma: float = 0.5                    # lognormal sigma
    rate: float = 1.0                     # exponential 速率（均值 = 1/rate）

    def sample(self, rng) -> float:
        if self.dist == "fixed":
            return self.value
        if self.dist == "uniform":
            return rng.uniform(self.low, self.high)
        if self.dist == "uniform_int":
            return int(rng.randint(int(self.low), int(self.high)))
        if self.dist == "lognormal":
            return rng.lognormvariate(self.mu, self.sigma)
        if self.dist == "exponential":
            return rng.expovariate(self.rate)
        raise ValueError(f"未知延迟分布: {self.dist}")


@dataclass
class ClientOverride:
    id: str
    task: Optional[str] = None
    speed_factor: Optional[float] = None
    start_offset: Optional[float] = None
    net_delay: Optional[DelayConfig] = None
    num_rounds: Optional[int] = None
    delay_mode: Optional[str] = None
    staleness_tau: Optional[DelayConfig] = None
    download_lag: Optional[DelayConfig] = None
    train_seconds: Optional[float] = None
    prox_mu: Optional[float] = None


@dataclass
class ClientsConfig:
    """客户端列表：既支持显式逐个配置，也支持 task_mix 按比例生成。

    delay_mode（对应主流 AFL 模拟的两种延迟范式，可逐客户端覆盖）：
        wallclock : 延迟 = 模拟秒。speed_factor 补睡 + net_delay 上传睡眠
                    （FedASMU / FedCompass / FedFa 一系的墙钟模拟）
        staleness : 延迟 = 服务器聚合步数 τ。更新提交后由服务器**持留**到
                    版本推进 version_trained_on + τ 才进入 buffer（FedAsync 官方
                    模拟的做法）；staleness_tau 采样 τ，本模式下 net_delay 不生效
    download_lag：下载侧延迟（延迟作用点旋钮）——客户端同步到
    全局版本 − download_lag 的旧快照上训练（两种模式都可用，0=关闭）。

    train_seconds：单轮本地训练的（虚拟）秒数，仅 timing: virtual 时必需——
    虚拟到达时间线由 start_offset + train_seconds/speed_factor + net_delay
    纯配置决定，与 GPU 卡数、线程竞争无关。wallclock 模式忽略。
    """
    num_clients: int = 8
    task_mix: dict[str, int] = field(default_factory=dict)   # task -> 客户端数
    num_rounds: int = 3
    speed_factor: float = 1.0          # 建议取 (0,1]：1=基准算力，0.5=慢一倍
    start_offset: float = 0.0
    net_delay: DelayConfig = field(default_factory=DelayConfig)
    delay_mode: str = "wallclock"      # wallclock / staleness
    train_seconds: Optional[float] = None   # 虚拟模式的单轮训练秒数（按任务/客户端覆盖）
    staleness_tau: DelayConfig = field(   # staleness 模式的 τ 分布
        default_factory=lambda: DelayConfig(dist="uniform_int", low=1, high=5))
    download_lag: DelayConfig = field(    # 下载侧延迟（同步到多旧的版本）
        default_factory=lambda: DelayConfig(dist="fixed", value=0))
    # 任务相关延迟：task -> 覆盖字段（speed_factor/net_delay/staleness_tau/
    # download_lag/start_offset/delay_mode）。真实系统的任务-时间相关性
    # （OCR 慢、业务时段同步等）→ 到达按任务自然聚集；优先级高于全局默认、
    # 低于 overrides
    task_profiles: dict[str, dict] = field(default_factory=dict)
    client_lr: float = 1e-4
    batch_size: int = 2
    grad_clip: float = 1.0
    # FedProx 近端系数（0=关闭；>0 时本地损失加 μ/2·‖θ−θ_global‖²）
    prox_mu: float = 0.0
    # 任务内训练数据的客户端划分：
    #   shared  = 同任务全部客户端共用同一份训练集（旧行为）
    #   disjoint= 每个数据集随机均分成 n_clients 份、每客户端一份
    #             （Pilot/FedMIT 的划分协议；种子来自 experiment.seed，可复现）
    data_partition: str = "shared"
    overrides: list[ClientOverride] = field(default_factory=list)

    def expand(self) -> list[dict]:
        """展开成显式客户端字典列表。task_mix 与 num_clients 不一致时以 task_mix 总数为准。"""
        task_names = list(self.task_mix.keys())
        clients: list[dict] = []
        if self.task_mix:
            for t in task_names:
                for _ in range(self.task_mix[t]):
                    clients.append({"task": t})
        else:
            raise ValueError("clients.task_mix 不能为空（请按任务指定客户端数量）")
        base = {
            "num_rounds": self.num_rounds,
            "speed_factor": self.speed_factor,
            "start_offset": self.start_offset,
            "net_delay": self.net_delay,
            "delay_mode": self.delay_mode,
            "staleness_tau": self.staleness_tau,
            "download_lag": self.download_lag,
            "train_seconds": self.train_seconds,
            "client_lr": self.client_lr,
            "batch_size": self.batch_size,
            "grad_clip": self.grad_clip,
            "prox_mu": self.prox_mu,
        }
        out = []
        for i, c in enumerate(clients):
            profile = dict(self.task_profiles.get(c["task"], {}))
            for k in ("net_delay", "staleness_tau", "download_lag"):
                if k in profile:
                    profile[k] = _as_delay(profile[k])
            merged = {"id": f"c{i}", **base, **profile, **c}
            out.append(merged)
        for ov in self.overrides:
            for c in out:
                if c["id"] == ov.id:
                    for k in ("task", "speed_factor", "start_offset", "net_delay",
                              "num_rounds", "delay_mode", "staleness_tau", "download_lag",
                              "train_seconds", "prox_mu"):
                        v = getattr(ov, k)
                        if v is not None:
                            c[k] = v
                    break
            else:
                raise ValueError(f"overrides 里的客户端 id 不存在: {ov.id}")
        return out


@dataclass
class AggregationConfig:
    name: str = "fedbuff"
    K: int = 4                            # fedbuff/staleness_weighted 缓冲大小
    server_lr: float = 1.0                # 全局更新步长（乘在 delta 上）
    # staleness_weighted / fedasync
    staleness_lambda: float = 0.3
    min_weight: float = 0.1
    # fedcompass（时间变化 staleness 阈值，单位见 fedcompass.py 文档）
    theta0: float = 1.0
    theta_growth: float = 0.0025
    theta_max: float = 6.0
    min_count: int = 3
    # fixed_order（killer experiment）
    batch_size: int = 8                   # 攒够多少个更新触发一次聚合
    order: dict = field(default_factory=lambda: {"mode": "random", "seed": 0})


@dataclass
class ServerConfig:
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    eval_every_batches: int = 5           # 每 N 个聚合批次触发一次分任务评估（0=关闭）
    gen_every_batches: int = 0            # 生成评估的独立频率（0=跟随 eval_every_batches）
    eval_batch_size: int = 4
    gen_eval_samples: int = 8             # 每任务做生成匹配评估的样本数（0=只算 loss）
    max_new_tokens: int = 32
    # I3 插桩：每施加一个 delta 前后各测一次分任务 probe loss → delta_probes.jsonl
    probe_on_apply: bool = False
    probe_manifest: Optional[str] = None  # 探针集索引清单（prepare_datasets 产出）
    probe_n: int = 128                    # 无 manifest 时的回退：取 eval 集前 N 个样本


@dataclass
class ExperimentConfig:
    name: str = "afl_vlm_run"
    seed: int = 42
    output_dir: str = "runs/run"
    time_scale: float = 1.0               # 缩放 start_offset/net_delay 的真实睡眠时长
    # 计时模式（时间线与执行的解耦程度）：
    #   wallclock : 到达时间线 = 真实线程竞争 + 补睡模拟（clients ≤ 卡数时无失真）
    #   virtual   : 两阶段"计划–执行"——阶段1 按虚拟时钟（train_seconds/speed +
    #               net_delay + τ）离线算出完整调度表（到达顺序/同步版本/施加顺序，
    #               config+seed 的纯函数，严格可复现）；阶段2 真实训练按计划回放，
    #               卡排队、线程竞争对时间线零影响
    timing: str = "wallclock"
    model: ModelConfig = field(default_factory=ModelConfig)
    tasks: dict[str, TaskConfig] = field(default_factory=dict)
    clients: ClientsConfig = field(default_factory=ClientsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# ---------------------------------------------------------------------------
# YAML → dataclass
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """dict 深合并：override 优先；list/标量整体替换。"""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _build_task_source(d: dict) -> TaskSource:
    return TaskSource(**d)


def _build_task(name: str, d: dict) -> TaskConfig:
    d = dict(d)
    d["source"] = _build_task_source(d.get("source", {}))
    d["name"] = name
    return TaskConfig(**d)


def _build_delay(d: dict) -> DelayConfig:
    return DelayConfig(**d)


def _as_delay(v) -> DelayConfig:
    """task_profiles 里的延迟字段可能是 dict（YAML 直填）或已是 DelayConfig。"""
    if isinstance(v, DelayConfig):
        return v
    if isinstance(v, dict):
        return DelayConfig(**v)
    raise ValueError(f"延迟字段应为映射或 DelayConfig，得到: {v!r}")


def required_history(cfg: ExperimentConfig) -> int:
    """download_lag > 0 时管理器需要保留的版本历史条数（0 = 关闭，省内存）。"""
    mx = 0
    for c in cfg.clients.expand():
        d = c.get("download_lag")
        if d is None:
            continue
        if d.dist == "fixed":
            ub = d.value
        elif d.dist in ("uniform", "uniform_int"):
            ub = d.high
        else:
            ub = max(d.high, d.value)
        mx = max(mx, int(ub))
    return mx + 2 if mx > 0 else 0


def _load_raw(path: str) -> dict:
    """读 YAML 并递归解析 base: 继承链（dict 深合并，子文件优先）。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if raw.get("base"):
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), raw["base"])
        raw = _deep_merge(_load_raw(base_path), raw)
    raw.pop("base", None)
    return raw


def load_config(path: str) -> ExperimentConfig:
    import os

    raw = _load_raw(path)

    exp = raw.get("experiment") or {}
    model = raw.get("model") or {}
    lora = model.pop("lora", {}) if "lora" in model else {}
    tasks = {name: _build_task(name, td) for name, td in (raw.get("tasks") or {}).items()}

    clients_raw = raw.get("clients") or {}
    for key in ("net_delay", "staleness_tau", "download_lag"):
        if isinstance(clients_raw.get(key), dict):
            clients_raw[key] = _build_delay(clients_raw[key])
    profiles = {}
    for tname, prof in (clients_raw.get("task_profiles") or {}).items():
        prof = dict(prof)
        for key in ("net_delay", "staleness_tau", "download_lag"):
            if key in prof:
                prof[key] = _build_delay(prof[key])
        profiles[tname] = prof
    clients_raw["task_profiles"] = profiles
    overrides = []
    for ov in clients_raw.get("overrides", []):
        ov = dict(ov)
        for key in ("net_delay", "staleness_tau", "download_lag"):
            if isinstance(ov.get(key), dict):
                ov[key] = _build_delay(ov[key])
        overrides.append(ClientOverride(**ov))
    clients_raw["overrides"] = overrides
    clients = ClientsConfig(**clients_raw)

    server_raw = raw.get("server") or {}
    agg_raw = dict(server_raw.get("aggregation", {}))
    server_raw["aggregation"] = AggregationConfig(**agg_raw)
    server = ServerConfig(**server_raw)

    cfg = ExperimentConfig(
        name=exp.get("name", "afl_vlm_run"),
        seed=exp.get("seed", 42),
        output_dir=exp.get("output_dir", "runs/run"),
        time_scale=exp.get("time_scale", 1.0),
        timing=exp.get("timing", "wallclock"),
        model=ModelConfig(**model, lora=LoRAConfig(**lora)) if model else ModelConfig(lora=LoRAConfig(**lora)),
        tasks=tasks,
        clients=clients,
        server=server,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    if not cfg.tasks:
        raise ValueError("必须在 tasks: 里至少定义一个任务")
    mix_total = sum(cfg.clients.task_mix.values())
    missing = [t for t in cfg.clients.task_mix if t not in cfg.tasks]
    if missing:
        raise ValueError(f"task_mix 引用了未定义的任务: {missing}（已定义: {list(cfg.tasks)}）")
    if mix_total <= 0:
        raise ValueError("task_mix 的客户端总数必须 > 0")
    known_agg = {"immediate_fifo", "fedbuff", "staleness_weighted",
                 "fedasync", "fedcompass", "sync_avg", "fixed_order"}
    if cfg.server.aggregation.name not in known_agg:
        raise ValueError(f"未知聚合策略 {cfg.server.aggregation.name}，可选: {sorted(known_agg)}")
    known_models = {"qwen2.5-vl-7b", "qwen2.5-vl-3b", "tiny_mock"}
    if cfg.model.name not in known_models:
        raise ValueError(f"未知模型 {cfg.model.name}，可选: {sorted(known_models)}（可在 models/__init__.py 注册新模型）")
    known_dists = {"fixed", "uniform", "uniform_int", "lognormal", "exponential"}
    if cfg.clients.delay_mode not in {"wallclock", "staleness"}:
        raise ValueError(f"未知 delay_mode {cfg.clients.delay_mode}，可选: wallclock / staleness")
    if cfg.clients.data_partition not in {"shared", "disjoint"}:
        raise ValueError(f"未知 data_partition {cfg.clients.data_partition}，可选: shared / disjoint")
    for fname, d in (("net_delay", cfg.clients.net_delay),
                     ("staleness_tau", cfg.clients.staleness_tau),
                     ("download_lag", cfg.clients.download_lag)):
        if d.dist not in known_dists:
            raise ValueError(f"clients.{fname} 未知分布 {d.dist}，可选: {sorted(known_dists)}")
    if cfg.clients.delay_mode == "staleness" and cfg.clients.staleness_tau.dist in ("uniform", "lognormal", "exponential"):
        raise ValueError("staleness 模式下 staleness_tau 应取整步数分布: fixed / uniform_int")
    unknown_profiles = [t for t in cfg.clients.task_profiles if t not in cfg.clients.task_mix]
    if unknown_profiles:
        raise ValueError(f"task_profiles 引用了未在 task_mix 中出现的任务: {unknown_profiles}")
    if cfg.timing not in {"wallclock", "virtual"}:
        raise ValueError(f"未知 timing {cfg.timing}，可选: wallclock / virtual")
    if cfg.timing == "virtual":
        for c in cfg.clients.expand():
            ts = c.get("train_seconds")
            if not ts or ts <= 0:
                raise ValueError(
                    f"timing: virtual 需要为每个客户端配置 train_seconds（客户端 {c['id']} 缺失或非正）。"
                    "虚拟时间线由 train_seconds/speed_factor + net_delay 驱动，"
                    "建议取 ≥ 单卡真实单轮训练耗时（可用 wallclock 跑一轮从 arrivals.jsonl 的 "
                    "train_seconds 字段标定）；可在 clients.train_seconds 全局设置、"
                    "task_profiles 按任务覆盖、overrides 按客户端覆盖")
            if c["speed_factor"] <= 0:
                raise ValueError(f"客户端 {c['id']} 的 speed_factor 必须 > 0（virtual 模式下"
                                 "虚拟训练时长 = train_seconds/speed_factor）")


def dump_config(cfg: ExperimentConfig, path: str) -> None:
    """把配置原样存进输出目录，保证实验可追溯。"""
    data = dataclasses.asdict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
