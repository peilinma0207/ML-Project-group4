"""共享工具函数：WER 计算、数据加载、路径配置。"""

from __future__ import annotations

import json
import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

# ============================================================
# 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORRUPTED_DIR = DATA_DIR / "corrupted"
METADATA_DIR = CORRUPTED_DIR / "metadata"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

CORRUPTION_STATS_PATH = METADATA_DIR / "corruption_stats.json"
ADVERSARIAL_REPORT_PATH = METADATA_DIR / "adversarial_report.json"
HOMOPHONE_REPORT_PATH = METADATA_DIR / "homophone_report.json"
VIDEOS_METADATA_PATH = DATA_DIR / "metadata" / "videos_metadata.csv"
GLOSSARY_PATH = DATA_DIR / "glossary.json"
GROUND_TRUTH_DIR = DATA_DIR / "transcripts_ground_truth"


# ============================================================
# WER 计算
# ============================================================

def compute_wer(reference: str, hypothesis: str) -> float:
    """计算词错误率 (Word Error Rate)，基于编辑距离。"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    r_len = len(ref_words)
    h_len = len(hyp_words)

    if r_len == 0:
        return 1.0 if h_len > 0 else 0.0

    d = np.zeros((r_len + 1, h_len + 1), dtype=np.int32)
    for i in range(r_len + 1):
        d[i, 0] = i
    for j in range(h_len + 1):
        d[0, j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(
                    d[i - 1, j] + 1,      # deletion
                    d[i, j - 1] + 1,      # insertion
                    d[i - 1, j - 1] + 1,  # substitution
                )

    return d[r_len, h_len] / r_len


def compute_cer(reference: str, hypothesis: str) -> float:
    """计算字符错误率 (Character Error Rate)。"""
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))

    r_len = len(ref_chars)
    h_len = len(hyp_chars)

    if r_len == 0:
        return 1.0 if h_len > 0 else 0.0

    d = np.zeros((r_len + 1, h_len + 1), dtype=np.int32)
    for i in range(r_len + 1):
        d[i, 0] = i
    for j in range(h_len + 1):
        d[0, j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(
                    d[i - 1, j] + 1,
                    d[i, j - 1] + 1,
                    d[i - 1, j - 1] + 1,
                )

    return d[r_len, h_len] / r_len


# ============================================================
# 术语匹配
# ============================================================

def load_glossary() -> dict[str, list[str]]:
    """加载术语表，返回 {术语: [别名列表]}。"""
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        return json.load(f)


def compute_terminology_accuracy(
    text: str, glossary: dict[str, list[str]]
) -> dict[str, float]:
    """计算术语准确率：在文本中检查术语覆盖情况。

    返回 {'precision': float, 'recall': float, 'f1': float}
    """
    text_lower = text.lower()
    all_terms = set()
    for term, aliases in glossary.items():
        all_terms.add(term.lower())
        for a in aliases:
            all_terms.add(a.lower())

    found = sum(1 for t in all_terms if t in text_lower)
    total = len(all_terms)

    recall = found / total if total > 0 else 0.0
    precision = found / max(found, 1)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


# ============================================================
# 数据加载
# ============================================================

def load_corruption_stats() -> dict[str, list[dict]]:
    """加载 corruption_stats.json。"""
    with open(CORRUPTION_STATS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_adversarial_report() -> list[dict]:
    """加载 adversarial_report.json。"""
    with open(ADVERSARIAL_REPORT_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_homophone_report() -> list[dict]:
    """加载 homophone_report.json。"""
    with open(HOMOPHONE_REPORT_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_videos_metadata() -> list[dict]:
    """加载 videos_metadata.csv。"""
    with open(VIDEOS_METADATA_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_ground_truth(video_id: str) -> str:
    """加载某个视频的 ground truth 转写文本。"""
    path = GROUND_TRUTH_DIR / f"{video_id}_ground_truth.txt"
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    lines = text.strip().split("\n")
    return " ".join(
        line.strip() for line in lines
        if not re.match(r"^\d{2}:\d{2}(:\d{2})?$", line.strip())
    ).strip()


def load_pipeline_output(output_dir: Path) -> list[dict]:
    """加载 pipeline 输出的 script.json。"""
    script_path = output_dir / "script.json"
    if not script_path.exists():
        return []
    with open(script_path, encoding="utf-8") as f:
        return json.load(f)


def load_evaluation_results() -> dict[str, Any]:
    """加载批量评估结果。"""
    path = RESULTS_DIR / "evaluation_results.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 图表辅助
# ============================================================

def setup_chinese_font():
    """配置 matplotlib 中文字体支持。"""
    import matplotlib.pyplot as plt
    import matplotlib

    font_candidates = [
        "SimHei", "Microsoft YaHei", "STHeiti", "WenQuanYi Micro Hei",
        "Noto Sans CJK SC", "Source Han Sans SC",
    ]
    for font in font_candidates:
        try:
            matplotlib.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [font] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue

    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(fig, name: str, dpi: int = 150):
    """保存图表到 figures/ 目录。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"  [保存] {path}")
    return path
