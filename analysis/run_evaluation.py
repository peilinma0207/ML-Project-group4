"""真实实验评估脚本（并发版）。

使用完整 agent 流水线对受损样本进行修复评估：
- 噪声/对抗样本：完整流水线（ASR → Frame Sampling → VLM → RAG → Repair）
- 同音词样本：直接调用修复模块（跳过 ASR，使用已有的受损文本）

运行方式：uv run python analysis/run_evaluation.py --resume --workers 5
"""

from __future__ import annotations

import sys
import json
import time
import os
import logging
import shutil
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# Fix PyTorch 2.6 weights_only issue for pyannote/omegaconf
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
    load_ground_truth,
    load_corruption_stats,
    load_adversarial_report,
)

# ============================================================
# 配置
# ============================================================

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

logger = logging.getLogger(__name__)


def _log(msg: str):
    with _print_lock:
        print(msg, flush=True)


# ============================================================
# 评估记录
# ============================================================

@dataclass
class EvalResult:
    video_id: str
    corruption_type: str
    corruption_params: dict
    ground_truth_words: int = 0
    baseline_transcription: str = ""
    baseline_wer: float = 0.0
    baseline_cer: float = 0.0
    repaired_transcription: str = ""
    repaired_wer: float = 0.0
    repaired_cer: float = 0.0
    wer_improvement: float = 0.0
    n_segments: int = 0
    n_review_required: int = 0
    pipeline_time_s: float = 0.0


# ============================================================
# Resume 检测
# ============================================================

def _find_existing_script_json(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    for sub in output_dir.iterdir():
        if sub.is_dir():
            p = sub / "script.json"
            if p.exists():
                return p
    p = output_dir / "script.json"
    return p if p.exists() else None


def _is_valid_pipeline_output(output_dir: Path) -> bool:
    script_path = _find_existing_script_json(output_dir)
    if not script_path:
        return False
    try:
        segments = json.loads(script_path.read_text(encoding="utf-8"))
        if not segments:
            return False
        if not any(s.get("confidence", 0) > 0 for s in segments):
            return False
        is_homophone = "homophone" in output_dir.name
        if not is_homophone:
            total_vis = sum(len(s.get("evidence", {}).get("visual", [])) for s in segments)
            if total_vis == 0:
                return False
        return True
    except Exception:
        return False


def _load_existing_segments(output_dir: Path) -> list[dict]:
    script_path = _find_existing_script_json(output_dir)
    if not script_path:
        return []
    return json.loads(script_path.read_text(encoding="utf-8"))


def _remove_failed_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)


# ============================================================
# Pipeline 运行
# ============================================================

def run_pipeline_on_audio(video_id: str, audio_path: str, output_dir: Path) -> list[dict]:
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


def run_pipeline_repair_only(video_id: str, corrupted_text: str, output_dir: Path) -> list[dict]:
    from agent.schema import JobConfig, ASRSegment, WordTimestamp
    from agent import rag_retrieve, evidence_merge, script_repair, export

    config = JobConfig(
        job_id=f"eval_{video_id}",
        video_uri=str(DATA_DIR / "videos" / f"{video_id}.mp4"),
        topic_hint="AI and Machine Learning",
        output_dir=str(output_dir),
        glossary_path=str(DATA_DIR / "glossary.json"),
        text_model="Qwen/Qwen3.6-27B",
        text_api_base=os.environ.get("LLM_API_BASE", ""),
        text_api_key=os.environ.get("LLM_API_KEY", ""),
        vlm_model="Qwen/Qwen3.6-27B",
        vlm_api_base=os.environ.get("LLM_API_BASE", ""),
        vlm_api_key=os.environ.get("LLM_API_KEY", ""),
    )

    words = corrupted_text.split()
    asr_words = [WordTimestamp(word=w, start=i * 0.5, end=(i + 1) * 0.5, confidence=0.6)
                 for i, w in enumerate(words)]
    segments = [ASRSegment(
        segment_id="seg_0000",
        start=0.0,
        end=len(words) * 0.5,
        text=corrupted_text,
        speaker="SPEAKER_00",
        words=asr_words,
        quality_flags=["low_confidence"],
    )]

    rag_hits = rag_retrieve.run(segments, [], config.glossary_path)
    merged = evidence_merge.run(segments, [], rag_hits)
    repaired = script_repair.run(merged, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    export.run(repaired, config)

    return [{"text": s.text, "confidence": s.confidence, "review_required": s.review_required}
            for s in repaired]


# ============================================================
# 统一任务描述
# ============================================================

@dataclass
class Task:
    task_id: str
    label: str
    corruption_type: str
    video_id: str
    corruption_params: dict
    output_dir: Path
    audio_path: str = ""
    corrupted_text: str = ""
    ground_truth: str = ""
    baseline_wer: float = 0.0
    baseline_cer: float = 0.0


def _collect_tasks() -> list[Task]:
    tasks = []
    stats = load_corruption_stats()

    for entry in stats["noise"]:
        vid = entry["video_id"]
        nt = entry["noise_type"]
        snr = entry["snr_db"]
        audio = str(PROJECT_ROOT / entry["output_path"])
        gt = load_ground_truth(vid)
        if not gt or not Path(audio).exists():
            continue
        tasks.append(Task(
            task_id=f"{vid}_{nt}_snr{snr}",
            label=f"{vid} {nt:6s} SNR={snr:2d}dB",
            corruption_type="noise",
            video_id=vid,
            corruption_params={"noise_type": nt, "snr_db": snr},
            output_dir=RESULTS_DIR / "pipeline_runs" / f"{vid}_{nt}_snr{snr}",
            audio_path=audio,
            ground_truth=gt,
        ))

    for entry in stats["homophone"]:
        vid = entry["video_id"]
        ratio = entry["ratio"]
        text_path = CORRUPTED_DIR / "homophone" / "text" / f"{vid}_homophone_r{int(ratio * 100)}.txt"
        gt = load_ground_truth(vid)
        if not gt or not text_path.exists():
            continue
        corrupted = text_path.read_text(encoding="utf-8").strip()
        if not corrupted:
            continue
        tasks.append(Task(
            task_id=f"{vid}_homophone_r{int(ratio * 100)}",
            label=f"{vid} homophone r={ratio:.0%}",
            corruption_type="homophone",
            video_id=vid,
            corruption_params={"ratio": ratio},
            output_dir=RESULTS_DIR / "pipeline_runs" / f"{vid}_homophone_r{int(ratio * 100)}",
            corrupted_text=corrupted,
            ground_truth=gt,
            baseline_wer=compute_wer(gt, corrupted),
            baseline_cer=compute_cer(gt, corrupted),
        ))

    adv_report = load_adversarial_report()
    for entry in adv_report:
        vid = entry["video_id"]
        raw_path = PROJECT_ROOT / entry["output_path"]
        if not raw_path.exists():
            raw_path = CORRUPTED_DIR / "adversarial" / "audio" / raw_path.name
        audio = str(raw_path)
        gt = load_ground_truth(vid)
        if not gt or not Path(audio).exists():
            continue
        tasks.append(Task(
            task_id=f"{vid}_adversarial",
            label=f"{vid} adversarial ε=0.01",
            corruption_type="adversarial",
            video_id=vid,
            corruption_params={"epsilon": entry["epsilon"]},
            output_dir=RESULTS_DIR / "pipeline_runs" / f"{vid}_adversarial",
            audio_path=audio,
            ground_truth=gt,
            baseline_wer=entry["word_error_rate"],
        ))

    return tasks


# ============================================================
# 单任务执行（worker 函数）
# ============================================================

def _run_one(task: Task, idx: int, total: int) -> EvalResult | None:
    tag = f"[{idx + 1:02d}/{total}]"
    gt = task.ground_truth

    try:
        _remove_failed_output(task.output_dir)

        t0 = time.time()
        if task.corruption_type == "homophone":
            segments = run_pipeline_repair_only(
                task.video_id, task.corrupted_text, task.output_dir)
        else:
            segments = run_pipeline_on_audio(
                task.video_id, task.audio_path, task.output_dir)
        elapsed = time.time() - t0

        repaired_text = " ".join(s.get("text", "") for s in segments)
        r_wer = compute_wer(gt, repaired_text)
        r_cer = compute_cer(gt, repaired_text)
        n_review = sum(1 for s in segments if s.get("review_required"))

        result = EvalResult(
            video_id=task.video_id,
            corruption_type=task.corruption_type,
            corruption_params=task.corruption_params,
            ground_truth_words=len(gt.split()),
            baseline_transcription=task.corrupted_text,
            baseline_wer=task.baseline_wer,
            baseline_cer=task.baseline_cer,
            repaired_transcription=repaired_text,
            repaired_wer=r_wer,
            repaired_cer=r_cer,
            wer_improvement=task.baseline_wer - r_wer if task.baseline_wer else 0.0,
            n_segments=len(segments),
            n_review_required=n_review,
            pipeline_time_s=elapsed,
        )

        if task.corruption_type == "homophone":
            status = "✓" if r_wer < task.baseline_wer else "✗"
            _log(f"  {tag} {task.label} | WER: {task.baseline_wer:.3f} → {r_wer:.3f} ({status}) | {elapsed:.1f}s")
        else:
            _log(f"  {tag} {task.label} | WER={r_wer:.3f} | segs={len(segments)} | {elapsed:.1f}s")

        return result

    except Exception as e:
        _log(f"  {tag} {task.label} | ERROR: {e}")
        return None


# ============================================================
# 汇总与保存
# ============================================================

def print_summary(results: list[EvalResult]) -> None:
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)

    for ctype in ["noise", "homophone", "adversarial"]:
        subset = [r for r in results if r.corruption_type == ctype]
        if not subset:
            continue

        r_wers = [r.repaired_wer for r in subset]
        has_baseline = any(r.baseline_wer > 0 for r in subset)

        print(f"\n  [{ctype}] (n={len(subset)})")
        if has_baseline:
            b_wers = [r.baseline_wer for r in subset if r.baseline_wer > 0]
            improvements = [r.wer_improvement for r in subset if r.baseline_wer > 0]
            improved_count = sum(1 for imp in improvements if imp > 0)
            print(f"    基线 WER:  {np.mean(b_wers):.4f} ± {np.std(b_wers):.4f}")
            print(f"    修复 WER:  {np.mean(r_wers):.4f} ± {np.std(r_wers):.4f}")
            print(f"    平均改善:  {np.mean(improvements):+.4f}")
            print(f"    改善比例:  {improved_count}/{len(improvements)} ({improved_count/len(improvements)*100:.1f}%)")
        else:
            print(f"    修复 WER:  {np.mean(r_wers):.4f} ± {np.std(r_wers):.4f}")

        avg_time = np.mean([r.pipeline_time_s for r in subset])
        print(f"    平均耗时:  {avg_time:.1f}s/样本")


def save_results(results: list[EvalResult]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "evaluation_results.json"

    data = {
        "metadata": {
            "pipeline": "agent (full pipeline)",
            "text_model": "qwen3.6-27b",
            "api_base": os.environ.get("LLM_API_BASE", ""),
            "whisper_model": "tiny",
            "total_samples": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "records": [asdict(r) for r in results],
    }

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  [保存] {output_path}")
    return output_path


# ============================================================
# main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="真实实验评估（并发版）")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已成功的 pipeline 运行，仅重跑失败的样本")
    parser.add_argument("--workers", type=int, default=5,
                        help="并发 worker 数（默认 5）")
    args = parser.parse_args()

    # 并发模式下压制 pipeline 内部的逐行日志，只保留 WARNING+
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print("=" * 60)
    print("真实实验评估 — 使用完整 Agent 流水线（并发版）")
    print(f"  修复模型: Qwen/Qwen3.5-9B @ {os.environ.get('LLM_API_BASE', 'N/A')}")
    print(f"  ASR: whisper-tiny (via pipeline)")
    print(f"  并发数: {args.workers}")
    if args.resume:
        print(f"  模式: resume（仅重跑失败样本）")
    print("=" * 60)

    if not os.environ.get("LLM_API_KEY"):
        print("[错误] 未设置 LLM_API_KEY，请在 .env 中配置")
        sys.exit(1)

    all_tasks = _collect_tasks()
    print(f"\n  总样本数: {len(all_tasks)}")

    # 分拣：cached / 需重跑
    cached_results: list[EvalResult] = []
    pending_tasks: list[Task] = []

    for task in all_tasks:
        if args.resume and _is_valid_pipeline_output(task.output_dir):
            segments = _load_existing_segments(task.output_dir)
            repaired_text = " ".join(s.get("text", "") for s in segments)
            gt = task.ground_truth
            r_wer = compute_wer(gt, repaired_text)
            r_cer = compute_cer(gt, repaired_text)
            n_review = sum(1 for s in segments if s.get("review_required"))
            cached_results.append(EvalResult(
                video_id=task.video_id,
                corruption_type=task.corruption_type,
                corruption_params=task.corruption_params,
                ground_truth_words=len(gt.split()),
                baseline_transcription=task.corrupted_text,
                baseline_wer=task.baseline_wer,
                baseline_cer=task.baseline_cer,
                repaired_transcription=repaired_text,
                repaired_wer=r_wer,
                repaired_cer=r_cer,
                wer_improvement=task.baseline_wer - r_wer if task.baseline_wer else 0.0,
                n_segments=len(segments),
                n_review_required=n_review,
                pipeline_time_s=0.0,
            ))
        else:
            pending_tasks.append(task)

    print(f"  已完成(cached): {len(cached_results)}")
    print(f"  待运行: {len(pending_tasks)}")

    if not pending_tasks:
        print("\n  所有样本已完成，无需重跑。")
        results = cached_results
    else:
        print(f"\n  开始并发执行（{args.workers} workers）...")
        print("-" * 60)

        new_results: list[EvalResult] = []
        total = len(pending_tasks)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_run_one, task, i, total): task
                for i, task in enumerate(pending_tasks)
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    new_results.append(result)

        print("-" * 60)
        print(f"  本轮完成: {len(new_results)}/{total}")

        results = cached_results + new_results

    # 按 (corruption_type, video_id) 排序，保持输出稳定
    type_order = {"noise": 0, "homophone": 1, "adversarial": 2}
    results.sort(key=lambda r: (type_order.get(r.corruption_type, 9), r.video_id,
                                str(r.corruption_params)))

    print_summary(results)
    save_results(results)

    print("\n" + "=" * 60)
    print("评估完成！接下来运行：")
    print("  uv run python analysis/03_statistical_analysis.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
