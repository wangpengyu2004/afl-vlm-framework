"""AFL × VLM 指令微调实验框架.

研究假设：异步联邦 VLM 指令微调中，任务异构客户端产生不同的到达顺序，
聚合顺序（而非陈旧度）决定任务知识写入全局模型的次序。

模块结构:
    config       配置定义与校验
    server       异步服务器（到达队列 → 调度策略 → 顺序化聚合）
    client       客户端线程（本地 LoRA SFT + 模拟延迟上传）
    models       模型适配器（Qwen2.5-VL / tiny_mock）+ 共享模型管理器
    data         任务注册表与数据加载（local_json / hf_hub / synthetic）
    aggregation  聚合调度策略（研究主钩子：策略决定聚合哪些更新、以什么顺序）
    evaluation   分任务评估与遗忘分析
"""

__version__ = "0.1.0"
