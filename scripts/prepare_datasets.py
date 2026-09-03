"""准备 Pilot/FedMIT 协议的三任务数据集（GQA / OCR-VQA / COCO Caption）。

产出（--root 目录，默认 data/pilot/）：
    vqa.json      GQA 短答案 QA（LLaVA 格式）
    ocr.json      OCR-VQA 长答案逐字提取（LLaVA 格式）
    caption.json  COCO Caption 中长生成（LLaVA 格式）
    images/       全部图片（文件名 = 样本 id.jpg，三任务共用 image_root）
    probe_manifest.json  每任务探针集索引（eval 前部固定抽样，128 条）

划分协议（照抄 Pilot, AAAI-25）：每数据集随机抽取 N 条 → 尾部 10% 作 held-out
评估 → 训练部分由框架 data_partition: disjoint 随机均分给 3 个客户端。
probe_manifest 的索引与运行时 TaskData(max_eval=--max-eval) 的 eval 列表严格对齐。

数据源：优先从候选 HuggingFace 仓库流式拉取（schema 自动适配：图片列/问答列
自动探测，list 型问答逐条展开）；候选全部失败时调用 HF 搜索 API 列出可用仓库，
用 --dataset-id-<task> 手动指定重跑。**HF 仓库 ID 未在本仓库开发环境核验过**
（开发机无 HF 网络），脚本会先调 API 验证存在性再下载，失败即给出替代列表。

用法（服务器上，code/ 目录）:
    python -m scripts.prepare_datasets                       # 默认 12000 条/任务
    python -m scripts.prepare_datasets --per-task 6000       # 小规模试跑
    python -m scripts.prepare_datasets --dataset-id-ocrvqa <repo>   # 指定仓库
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from afl_vlm.config import TaskSource, stable_seed          # noqa: E402
from afl_vlm.data.loaders import load_local_json            # noqa: E402

# 候选 HF 仓库（按顺序尝试；都失败 → 打印搜索结果让用户 --dataset-id-* 指定）
CANDIDATES = {
    "vqa": ["lmms-lab/GQA"],
    "ocr": ["HuggingFaceM4/OCR-VQA", "lmms-lab/OCR-VQA"],
    "caption": ["HuggingFaceM4/COCO", "yerevann/coco-karpathy", "nlphuji/coco_captions_25k"],
}
SEARCH_KEYWORDS = {"vqa": "GQA", "ocr": "OCR-VQA", "caption": "coco captions"}

QUESTION_COLS = ["question", "qry", "problem", "query", "instruction", "questions"]
ANSWER_COLS = ["answer", "answers", "fullAnswer", "full_answer", "caption",
               "captions", "sentences", "text"]


def hf_api(path: str, timeout: int = 30):
    req = urllib.request.Request(f"https://huggingface.co/api/{path}",
                                 headers={"User-Agent": "afl-vlm-prepare"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def dataset_exists(repo: str) -> bool:
    try:
        hf_api(f"datasets/{repo}")
        return True
    except Exception:
        return False


def resolve_repo(task: str, override: str | None) -> str:
    if override:
        if not dataset_exists(override):
            raise SystemExit(f"[{task}] 指定的仓库不存在或不可达: {override}")
        return override
    for repo in CANDIDATES[task]:
        if dataset_exists(repo):
            print(f"[{task}] 使用 HF 仓库: {repo}")
            return repo
    # 全部候选失败：列出搜索结果供人工选择
    kw = SEARCH_KEYWORDS[task]
    print(f"[{task}] 候选仓库均不可用。HF 搜索 '{kw}' 的结果：", file=sys.stderr)
    try:
        hits = hf_api(f"datasets?search={urllib.request.quote(kw)}&limit=20")
        for h in hits[:20]:
            print(f"    {h.get('id')}  (downloads={h.get('downloads', '?')})",
                  file=sys.stderr)
    except Exception as e:
        print(f"    搜索 API 也不可达: {e!r}", file=sys.stderr)
    raise SystemExit(
        f"[{task}] 请从上面选一个含图片+问答的仓库，用 "
        f"--dataset-id-{task} <repo> 重跑；或检查服务器网络/镜像。")


def _to_text(v) -> str | None:
    """答案字段归一化：str / list[str] / list[dict(raw|text|caption)]。"""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    if isinstance(v, (list, tuple)):
        for item in v:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                for k in ("raw", "text", "caption", "answer"):
                    if item.get(k) and str(item[k]).strip():
                        return str(item[k]).strip()
    return None


def _detect_columns(rows: list[dict]) -> tuple[str, str | None, str | None]:
    """从样本行自动探测 图片列 / 问题列 / 答案列（Qwen 数据集 schema 各不相同）。"""
    img_col = q_col = a_col = None
    probe = next((r for r in rows if isinstance(r, dict)), {})
    for k, v in probe.items():
        if img_col is None and hasattr(v, "convert"):     # PIL.Image
            img_col = k
    for cand in QUESTION_COLS:
        if cand in probe:
            q_col = cand
            break
    for cand in ANSWER_COLS:
        if cand in probe:
            a_col = cand
            break
    return img_col, q_col, a_col


def _open_stream(repo: str):
    """流式打开数据集（不落全量盘）。split 名自动尝试 train。"""
    from datasets import load_dataset, get_dataset_config_names
    last_err = None
    for kwargs in ({"split": "train"},):
        try:
            return load_dataset(repo, streaming=True, **kwargs)
        except Exception as e:      # noqa: PERF203
            last_err = e
    # 有 config 名的数据集（如 lmms-lab/GQA）：逐个 config 试
    try:
        for cfg_name in get_dataset_config_names(repo):
            try:
                return load_dataset(repo, cfg_name, streaming=True, split="train")
            except Exception as e:  # noqa: PERF203
                last_err = e
    except Exception:
        pass
    raise SystemExit(f"打开 {repo} 失败: {last_err!r}")


def _save_image(img, path: str) -> bool:
    try:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def prepare_task(task: str, repo: str, n_target: int, root: str, img_root: str,
                 max_qa_per_row: int = 8) -> int:
    """流式扫描仓库，写 json + 图片；返回写出的样本数。"""
    from PIL import Image  # noqa: F401  (触发 PIL 以便 datasets 返回图片对象)

    out_path = os.path.join(root, f"{task}.json")
    ds = _open_stream(repo)
    items: list[dict] = []
    detect_rows: list[dict] = []
    img_col = q_col = a_col = None
    for row in ds:
        if len(items) >= n_target:
            break
        # 前 5 行内自动探测列名（首行可能是坏行）
        if img_col is None and len(detect_rows) < 5:
            detect_rows.append(row)
            img_col, q_col, a_col = _detect_columns(detect_rows)
            if img_col is None:
                continue
            print(f"[{task}] 列映射: image={img_col} question={q_col} answer={a_col}")
        img = row.get(img_col)
        if img is None:
            continue
        q_raw = row.get(q_col) if q_col else None
        a_raw = row.get(a_col) if a_col else None
        # list 型问答（OCR-VQA 常见：一图多问）逐对展开
        qs = q_raw if isinstance(q_raw, (list, tuple)) else [q_raw]
        ans = a_raw if isinstance(a_raw, (list, tuple)) else [a_raw]
        pairs = []
        for j in range(min(len(qs), len(ans), max_qa_per_row)):
            q, a = _to_text(qs[j]), _to_text(ans[j])
            if q and a and len(q) < 600 and len(a) < 900:
                pairs.append((q, a))
        if not pairs:
            continue
        # 同一行的一图多问共享同一张图片文件（文件名取首条的 id）
        fname = None
        for q, a in pairs:
            if len(items) >= n_target:
                break
            if fname is None:
                fname = f"{task}_{len(items):06d}.jpg"
                if not _save_image(img, os.path.join(img_root, fname)):
                    fname = None
                    break
            items.append({
                "id": fname[:-4],
                "image": fname,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{q}"},
                    {"from": "gpt", "value": a},
                ],
            })
        if len(items) and len(items) % 2000 < len(pairs):
            print(f"[{task}] 已收集 {len(items)}/{n_target}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    print(f"[{task}] 写出 {len(items)} 条 → {out_path}")
    return len(items)


def build_probe_manifest(task: str, root: str, eval_ratio: float, max_eval: int,
                         probe_size: int, seed: int) -> dict:
    """与运行时 TaskData 的 _split 逻辑严格一致地复算 eval 列表，抽固定探针索引。"""
    src = TaskSource(type="local_json", json_path=os.path.join(root, f"{task}.json"),
                     image_root=os.path.join(root, "images"))
    _, eval_part = load_local_json(src, None, max_eval, eval_ratio)
    rng = random.Random(stable_seed("probe", task, seed))
    idx = sorted(rng.sample(range(len(eval_part)), min(probe_size, len(eval_part))))
    return {"indices": idx, "n_eval": len(eval_part)}


def main() -> None:
    ap = argparse.ArgumentParser(description="准备 Pilot 协议三任务数据集")
    ap.add_argument("--root", default="data/pilot", help="产出根目录")
    ap.add_argument("--per-task", type=int, default=12000, help="每任务样本数")
    ap.add_argument("--eval-ratio", type=float, default=0.1)
    ap.add_argument("--max-eval", type=int, default=256,
                    help="运行时 TaskData 的 max_eval（manifest 与之对齐，改了要同步改配置）")
    ap.add_argument("--probe-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-id-vqa", default=None)
    ap.add_argument("--dataset-id-ocr", default=None)
    ap.add_argument("--dataset-id-caption", default=None)
    ap.add_argument("--only", default=None, help="只准备一个任务: vqa/ocr/caption")
    args = ap.parse_args()

    root = os.path.join(REPO_ROOT, args.root) if not os.path.isabs(args.root) else args.root
    img_root = os.path.join(root, "images")
    os.makedirs(img_root, exist_ok=True)

    tasks = [args.only] if args.only else ["vqa", "ocr", "caption"]
    ids = {"vqa": args.dataset_id_vqa, "ocr": args.dataset_id_ocr,
           "caption": args.dataset_id_caption}
    for task in tasks:
        repo = resolve_repo(task, ids.get(task))
        prepare_task(task, repo, args.per_task, root, img_root)

    # 探针清单（所有任务准备完后统一生成）
    manifest = {
        task: build_probe_manifest(task, root, args.eval_ratio, args.max_eval,
                                   args.probe_size, args.seed)
        for task in tasks
    }
    mpath = os.path.join(root, "probe_manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"\n探针清单 → {mpath}")
    for task, m in manifest.items():
        print(f"  {task}: {len(m['indices'])} 条探针 / eval 共 {m['n_eval']}")
    print("\n完成。实验配置使用相对路径 data/pilot/{vqa,ocr,caption}.json "
          "+ image_root data/pilot/images + probe_manifest data/pilot/probe_manifest.json")


if __name__ == "__main__":
    main()
