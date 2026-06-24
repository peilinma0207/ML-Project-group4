"""消融实验补跑脚本（跳过 +visual 条件）。

为已有 valid 结果补跑 baseline 和 +retrieval 条件：
- baseline: --skip-vlm --skip-repair（仅 ASR，无 API 调用）
- +retrieval: --skip-vlm（ASR + RAG + Repair，1 次 API 调用）
- full: 使用已有结果

运行方式：uv run python analysis/run_ablation.py --workers 5
"""

from __future__ import annotations

import sys
import json
import time
import os
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import (
    PROJECT_ROOT,
    DATA_DIR,
    CORRUPTED_DIR,
    RESULTS_DIR,
    compute_wer,
    compute_cer,
    compute_terminology_accuracy,
    load_glossary,
    load_ground_truth,
    load_corruption_stats,
    load_adversarial_report,
)

ENV_PATH = PROJECT_ROOT / ".env"
_print_lock = threading.Lock()


def _load_env():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_env()


def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)


def run_pipeline_condition(video_id: str, audio_path: str, output_dir: Path,
                           skip_vlm: bool = False, skip_repair: bool = False) -> list[dict]:
    """运行 pipeline 的特定条件。"""
    from agent.cli import main as pipeline_main

    video_path = DATA_DIR / "videos" / f"{video_id}.mp4"
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--video", str(video_path),
        "--audio", audio_path,
        "--output-dir", str(output_dir),
        "--whisper-model", "tiny",
        "--vlm-model", "Qwen/Qwen3.6-27B",
        "--text-model", "Qwen/Qwen3.6-27B",
        "--topic", "AI and Machine Learning",
        "--glossary", str(DATA_DIR / "glossary.json"),
    ]
    if skip_vlm:
        argv.append("--skip-vlm")
    if skip_repair:
        argv.append("--skip-repair")

    try:
        pipeline_main(argv)
    except SystemExit:
        pass

    for sub in output_dir.iterdir():
        script_json = sub / "script.json" if sub.is_dir() else None
        if script_json and script_json.exists():
            return json.loads(script_json.read_text(encoding="utf-8"))
    script_json = output_dir / "script.json"
    if script_json.exists():
        return json.loads(script_json.read_text(encoding="utf-8"))
    return []


def run_ablation_for_sample(video_id: str, audio_path: str, gt: str,
                            glossary: dict, idx: int, total: int) -> dict:
    """为单个 noise/adversarial 样本跑 baseline 和 +retrieval。"""
    tag = f"[{idx+1:02d}/{total}]"
    results = {}

    # baseline: skip-vlm + skip-repair
    out_base = RESULTS_DIR / "ablation_runs" / f"{video_id}_baseline"
    t0 = time.time()
    segs = run_pipeline_condition(video_id, audio_path, out_base,
                                  skip_vlm=True, skip_repair=True)
    text = " ".join(s.get("text", "") for s in segs)
    wer = compute_wer(gt, text)
    term = compute_terminology_accuracy(text, glossary)
    results["baseline"] = {"wer": wer, "term_accuracy": term["f1"]}
    t_base = time.time() - t0

    # +retrieval: skip-vlm (repair still runs with RAG)
    out_ret = RESULTS_DIR / "ablation_runs" / f"{video_id}_retrieval"
    t0 = time.time()
    segs = run_pipeline_condition(video_id, audio_path, out_ret,
                                  skip_vlm=True, skip_repair=False)
    text = " ".join(s.get("text", "") for s in segs)
    wer = compute_wer(gt, text)
    term = compute_terminology_accuracy(text, glossary)
    results["+retrieval"] = {"wer": wer, "term_accuracy": term["f1"]}
    t_ret = time.time() - t0

    _log(f"  {tag} {video_id} | baseline={results['baseline']['wer']:.3f} "
         f"+ret={results['+retrieval']['wer']:.3f} | {t_base:.0f}s+{t_ret:.0f}s")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="消融实验（跳过 +visual）")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print("=" * 60)
    print("消融实验 — baseline / +retrieval / full")
    print(f"  并发数: {args.workers}")
    print("=" * 60)

    # 加载已有结果
    eval_path = RESULTS_DIR / "evaluation_results.json"
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    records = eval_data["records"]
    raw_glossary = load_glossary()
    if isinstance(raw_glossary, list):
        glossary = {item["term"]: item.get("aliases", []) for item in raw_glossary}
    else:
        glossary = raw_glossary

    # 分类
    noise_adv = [r for r in records if r["corruption_type"] in ("noise", "adversarial")]
    homophone = [r for r in records if r["corruption_type"] == "homophone"]

    print(f"\n  noise/adversarial: {len(noise_adv)} 个（需跑 pipeline）")
    print(f"  homophone: {len(homophone)} 个（直接填充）")

    # homophone: 直接用已有数据
    for r in homophone:
        r["ablation_results"] = {
            "baseline": {"wer": r["baseline_wer"], "term_accuracy": 0.0},
            "+retrieval": {"wer": r["repaired_wer"], "term_accuracy": 0.0},
            "full": {"wer": r["repaired_wer"], "term_accuracy": 0.0},
        }
        # 计算 term_accuracy for retrieval/full
        if r.get("repaired_transcription"):
            term = compute_terminology_accuracy(r["repaired_transcription"], glossary)
            r["ablation_results"]["+retrieval"]["term_accuracy"] = term["f1"]
            r["ablation_results"]["full"]["term_accuracy"] = term["f1"]
        if r.get("baseline_transcription"):
            term = compute_terminology_accuracy(r["baseline_transcription"], glossary)
            r["ablation_results"]["baseline"]["term_accuracy"] = term["f1"]

    print(f"\n  homophone 填充完成")

    # noise/adversarial: 并发跑
    print(f"\n  开始跑 noise/adversarial 消融...")
    print("-" * 60)

    # 构建任务列表
    stats = load_corruption_stats()
    adv_report = load_adversarial_report()

    # 建立 audio_path 映射
    audio_map = {}
    for entry in stats["noise"]:
        vid = entry["video_id"]
        nt = entry["noise_type"]
        snr = entry["snr_db"]
        key = (vid, "noise", json.dumps({"noise_type": nt, "snr_db": snr}, sort_keys=True))
        audio_map[key] = str(PROJECT_ROOT / entry["output_path"])

    for entry in adv_report:
        vid = entry["video_id"]
        raw_path = PROJECT_ROOT / entry["output_path"]
        if not raw_path.exists():
            raw_path = CORRUPTED_DIR / "adversarial" / "audio" / raw_path.name
        key = (vid, "adversarial", json.dumps({"epsilon": entry["epsilon"]}, sort_keys=True))
        audio_map[key] = str(raw_path)

    tasks = []
    for r in noise_adv:
        vid = r["video_id"]
        ctype = r["corruption_type"]
        params_key = json.dumps(r["corruption_params"], sort_keys=True)
        key = (vid, ctype, params_key)
        audio_path = audio_map.get(key, "")
        if not audio_path or not Path(audio_path).exists():
            continue
        gt = load_ground_truth(vid)
        if not gt:
            continue
        tasks.append((r, vid, audio_path, gt))

    total = len(tasks)
    print(f"  任务数: {total}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, (r, vid, audio_path, gt) in enumerate(tasks):
            future = pool.submit(run_ablation_for_sample, vid, audio_path, gt, glossary, i, total)
            futures[future] = r
        for future in as_completed(futures):
            r = futures[future]
            try:
                abl = future.result()
                # full condition from existing results
                full_term = compute_terminology_accuracy(
                    r.get("repaired_transcription", ""), glossary)
                abl["full"] = {"wer": r["repaired_wer"], "term_accuracy": full_term["f1"]}
                r["ablation_results"] = abl
            except Exception as e:
                _log(f"  ERROR {r['video_id']}: {e}")

    print("-" * 60)

    # 保存
    eval_data["records"] = records
    eval_path.write_text(json.dumps(eval_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  [保存] {eval_path}")

    # 汇总
    has_abl = [r for r in records if r.get("ablation_results")]
    print(f"  消融数据: {len(has_abl)}/{len(records)} 条记录")

    print("\n" + "=" * 60)
    print("消融实验完成！接下来运行：")
    print("  uv run python analysis/03_statistical_analysis.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
