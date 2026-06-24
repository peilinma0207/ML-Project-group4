"""阶段一：现有数据的描述性分析。

基于已有的 corruption 元数据 JSON 文件，生成：
1. 数据集总览表
2. 三类受损数据对比可视化
3. 受损程度分布热力图
4. 同音词替换质量分析

运行方式：python analysis/01_corruption_descriptive.py
输出：analysis/figures/ 目录下的 PNG 图表
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    load_corruption_stats,
    load_adversarial_report,
    load_homophone_report,
    load_videos_metadata,
    setup_chinese_font,
    save_figure,
    FIGURES_DIR,
)


def analyze_dataset_overview(metadata: list[dict]) -> None:
    """数据集总览：视频信息统计表。"""
    print("\n" + "=" * 60)
    print("1. 数据集总览")
    print("=" * 60)

    print(f"\n视频总数: {len(metadata)}")
    print(f"{'视频ID':<12} {'说话人':<30} {'主题':<25} {'时长区间'}")
    print("-" * 90)
    for v in metadata:
        vid = v.get("video_id", "")
        speaker = v.get("speaker", "")[:28]
        theme = v.get("theme", "")[:23]
        start = v.get("start_time", "")
        end = v.get("end_time", "")
        print(f"{vid:<12} {speaker:<30} {theme:<25} {start}-{end}")


def plot_adversarial_analysis(adv_report: list[dict]) -> None:
    """FGSM 对抗样本分析：WER分布 + SNR vs WER 散点图。"""
    import matplotlib.pyplot as plt

    wers = [r["word_error_rate"] for r in adv_report]
    snrs = [r["snr_db"] for r in adv_report]
    video_ids = [r["video_id"].replace("video_", "V") for r in adv_report]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 箱线图：WER 分布
    ax = axes[0]
    bp = ax.boxplot(wers, vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("#FF6B6B")
    ax.set_ylabel("词错误率 (WER)")
    ax.set_title("FGSM 对抗样本 WER 分布")
    ax.set_xticklabels(["FGSM\n(ε=0.01)"])
    ax.axhline(np.mean(wers), color="red", linestyle="--", alpha=0.5,
               label=f"均值={np.mean(wers):.3f}")
    ax.legend(fontsize=8)

    # 柱状图：各视频 WER
    ax = axes[1]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(wers)))
    bars = ax.bar(video_ids, wers, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("视频")
    ax.set_ylabel("WER")
    ax.set_title("各视频 FGSM WER")
    ax.axhline(np.mean(wers), color="red", linestyle="--", alpha=0.7)
    ax.set_ylim(0, 1.0)
    for bar, w in zip(bars, wers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{w:.2f}", ha="center", va="bottom", fontsize=7)

    # 散点图：SNR vs WER
    ax = axes[2]
    ax.scatter(snrs, wers, c="#FF6B6B", s=80, edgecolors="black", linewidth=0.5, zorder=3)
    for i, vid in enumerate(video_ids):
        ax.annotate(vid, (snrs[i], wers[i]), fontsize=7, ha="left",
                    xytext=(3, 3), textcoords="offset points")
    # 回归线
    z = np.polyfit(snrs, wers, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(snrs) - 1, max(snrs) + 1, 50)
    ax.plot(x_line, p(x_line), "r--", alpha=0.5, label=f"线性拟合 (r={np.corrcoef(snrs, wers)[0,1]:.2f})")
    ax.set_xlabel("信噪比 SNR (dB)")
    ax.set_ylabel("WER")
    ax.set_title("FGSM: SNR vs WER")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, "fig4_fgsm_snr_vs_wer")
    plt.close(fig)


def plot_homophone_analysis(stats: dict) -> None:
    """同音词替换分析：替换率 vs 实际变化百分比。"""
    import matplotlib.pyplot as plt

    homo_data = stats["homophone"]
    if not homo_data:
        print("  [跳过] 无同音词替换数据")
        return

    # 按替换率分组
    ratios = sorted(set(h["ratio"] for h in homo_data))
    ratio_labels = [f"{int(r*100)}%" for r in ratios]

    word_changes_by_ratio = {r: [] for r in ratios}
    char_changes_by_ratio = {r: [] for r in ratios}

    for h in homo_data:
        word_changes_by_ratio[h["ratio"]].append(h["word_change_pct"])
        char_changes_by_ratio[h["ratio"]].append(h["char_change_pct"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 分组柱状图：目标替换率 vs 实际词变化百分比
    ax = axes[0]
    x = np.arange(len(ratios))
    width = 0.35

    word_means = [np.mean(word_changes_by_ratio[r]) for r in ratios]
    word_stds = [np.std(word_changes_by_ratio[r]) for r in ratios]
    char_means = [np.mean(char_changes_by_ratio[r]) for r in ratios]
    char_stds = [np.std(char_changes_by_ratio[r]) for r in ratios]

    bars1 = ax.bar(x - width/2, word_means, width, yerr=word_stds,
                   label="词级变化(%)", color="#4ECDC4", edgecolor="black", linewidth=0.5,
                   capsize=4)
    bars2 = ax.bar(x + width/2, char_means, width, yerr=char_stds,
                   label="字符级变化(%)", color="#FF6B6B", edgecolor="black", linewidth=0.5,
                   capsize=4)

    # 理想替换率参考线
    ideal_ratios = [r * 100 for r in ratios]
    ax.plot(x, ideal_ratios, "k--", marker="D", markersize=5, label="目标替换率", alpha=0.7)

    ax.set_xlabel("目标替换率")
    ax.set_ylabel("实际变化百分比 (%)")
    ax.set_title("同音词替换：目标 vs 实际变化程度")
    ax.set_xticks(x)
    ax.set_xticklabels(ratio_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 箱线图：词变化百分比按替换率分布
    ax = axes[1]
    bp_data = [word_changes_by_ratio[r] for r in ratios]
    bp = ax.boxplot(bp_data, tick_labels=ratio_labels, patch_artist=True)
    colors = ["#A8E6CF", "#FFD93D", "#FF6B6B"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("目标替换率")
    ax.set_ylabel("词级变化百分比 (%)")
    ax.set_title("同音词替换：各视频间变异度")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "fig3_homophone_substitution_rates")
    plt.close(fig)


def plot_homophone_quality(homo_report: list[dict]) -> None:
    """同音词替换质量分析：phonetic/edit similarity 分布。"""
    import matplotlib.pyplot as plt

    # 过滤有替换记录的条目
    all_subs = []
    for entry in homo_report:
        if entry.get("substitutions"):
            all_subs.extend(entry["substitutions"])

    if not all_subs:
        print("  [跳过] 无替换详情数据")
        return

    phonetic_sims = [s["phonetic_similarity"] for s in all_subs if "phonetic_similarity" in s]
    edit_sims = [s["edit_similarity"] for s in all_subs if "edit_similarity" in s]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 语音相似度分布
    ax = axes[0]
    ax.hist(phonetic_sims, bins=20, color="#4ECDC4", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.axvline(np.mean(phonetic_sims), color="red", linestyle="--",
               label=f"均值={np.mean(phonetic_sims):.2f}")
    ax.set_xlabel("语音相似度 (Phonetic Similarity)")
    ax.set_ylabel("频次")
    ax.set_title("同音词替换：语音相似度分布")
    ax.legend()

    # 编辑相似度分布
    ax = axes[1]
    ax.hist(edit_sims, bins=20, color="#FF6B6B", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.axvline(np.mean(edit_sims), color="red", linestyle="--",
               label=f"均值={np.mean(edit_sims):.2f}")
    ax.set_xlabel("编辑相似度 (Edit Similarity)")
    ax.set_ylabel("频次")
    ax.set_title("同音词替换：编辑距离相似度分布")
    ax.legend()

    plt.tight_layout()
    save_figure(fig, "fig3b_homophone_quality")
    plt.close(fig)


def plot_corruption_heatmap(stats: dict, adv_report: list[dict]) -> None:
    """受损程度热力图：video × condition 矩阵。"""
    import matplotlib.pyplot as plt

    video_ids = sorted(set(
        [h["video_id"] for h in stats["homophone"]] +
        [a["video_id"] for a in adv_report]
    ))

    # 构建条件列表
    conditions = [
        "噪声\nwhite\nSNR=3",
        "噪声\nwhite\nSNR=10",
        "噪声\nwhite\nSNR=20",
        "噪声\nreverb\nSNR=3",
        "噪声\nreverb\nSNR=10",
        "噪声\nreverb\nSNR=20",
        "同音词\n10%",
        "同音词\n25%",
        "同音词\n50%",
        "FGSM\nε=0.01",
    ]

    # 构建数据矩阵
    matrix = np.full((len(video_ids), len(conditions)), np.nan)

    # 噪声条件：用 SNR 的倒数作为受损程度代理（SNR 越低越严重）
    noise_severity = {3: 0.9, 10: 0.5, 20: 0.2}
    for n in stats["noise"]:
        vid_idx = video_ids.index(n["video_id"]) if n["video_id"] in video_ids else -1
        if vid_idx < 0:
            continue
        nt = n["noise_type"]
        snr = n["snr_db"]
        if nt == "white":
            col = {3: 0, 10: 1, 20: 2}.get(snr, -1)
        else:
            col = {3: 3, 10: 4, 20: 5}.get(snr, -1)
        if col >= 0:
            matrix[vid_idx, col] = noise_severity[snr]

    # 同音词条件：用 word_change_pct / 100
    for h in stats["homophone"]:
        vid_idx = video_ids.index(h["video_id"]) if h["video_id"] in video_ids else -1
        if vid_idx < 0:
            continue
        ratio = h["ratio"]
        col = {0.1: 6, 0.25: 7, 0.5: 8}.get(ratio, -1)
        if col >= 0:
            matrix[vid_idx, col] = h["word_change_pct"] / 100.0

    # FGSM：用 WER
    for a in adv_report:
        vid_idx = video_ids.index(a["video_id"]) if a["video_id"] in video_ids else -1
        if vid_idx >= 0:
            matrix[vid_idx, 9] = a["word_error_rate"]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=8, ha="center")
    ax.set_yticks(range(len(video_ids)))
    ax.set_yticklabels([v.replace("video_", "V") for v in video_ids])
    ax.set_xlabel("受损条件")
    ax.set_ylabel("视频")
    ax.set_title("受损程度分布热力图\n（噪声=严重度代理，同音词=实际词变化率，FGSM=WER）")

    # 标注数值
    for i in range(len(video_ids)):
        for j in range(len(conditions)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("受损程度 (0=轻微, 1=严重)")

    plt.tight_layout()
    save_figure(fig, "fig2_corruption_heatmap")
    plt.close(fig)


def plot_baseline_wer_boxplot(stats: dict, adv_report: list[dict]) -> None:
    """三类受损条件下 ASR 基线 WER 分布箱线图。

    注：噪声 WER 需要阶段二运行后才有真实数据，此处用 SNR 的估算 WER 作为占位。
    """
    import matplotlib.pyplot as plt

    # FGSM WER (已有实际数据)
    fgsm_wers = [r["word_error_rate"] for r in adv_report]

    # 同音词 WER (用 word_change_pct/100 作为近似 WER 下界)
    homo_wers = [h["word_change_pct"] / 100.0 for h in stats["homophone"]]

    # 噪声 WER (基于 SNR 的经验估算：SNR=3→WER≈0.6, SNR=10→WER≈0.3, SNR=20→WER≈0.1)
    # 这是经验估算，阶段二将用实际 WhisperX 结果替换
    snr_to_estimated_wer = {3: 0.55, 10: 0.25, 20: 0.08}
    noise_wers = []
    for n in stats["noise"]:
        base_wer = snr_to_estimated_wer.get(n["snr_db"], 0.3)
        noise_wers.append(base_wer + np.random.normal(0, 0.05))
    noise_wers = np.clip(noise_wers, 0, 1).tolist()

    fig, ax = plt.subplots(figsize=(8, 5))

    data = [noise_wers, homo_wers, fgsm_wers]
    labels = [f"噪声/混响\n(n={len(noise_wers)})",
              f"同音词替换\n(n={len(homo_wers)})",
              f"FGSM 对抗\n(n={len(fgsm_wers)})"]
    colors = ["#4ECDC4", "#FFD93D", "#FF6B6B"]

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 均值标注
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.scatter(i + 1, mean_val, marker="D", color="red", s=50, zorder=5)
        ax.annotate(f"μ={mean_val:.3f}", (i + 1, mean_val),
                    xytext=(10, 5), textcoords="offset points", fontsize=8, color="red")

    ax.set_ylabel("词错误率 (WER)")
    ax.set_title("三类受损条件下 ASR 基线 WER 分布\n（噪声为估算值*，阶段二运行后更新）")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_figure(fig, "fig1_baseline_wer_boxplot")
    plt.close(fig)


def plot_evidence_distribution() -> None:
    """Pipeline 输出中证据来源类型分布（基于现有示例）。"""
    import matplotlib.pyplot as plt

    # 加载已有的 pipeline 输出
    from utils import PROJECT_ROOT, load_pipeline_output
    output_dirs = list((PROJECT_ROOT / "output").glob("*/"))
    if not output_dirs:
        print("  [跳过] 无 pipeline 输出数据")
        return

    all_segments = []
    for d in output_dirs:
        segments = load_pipeline_output(d)
        all_segments.extend(segments)

    if not all_segments:
        print("  [跳过] pipeline 输出为空")
        return

    # 统计证据类型
    audio_count = 0
    visual_count = 0
    rag_count = 0
    review_required_count = 0
    confidences = []

    for seg in all_segments:
        ev = seg.get("evidence", {})
        if ev.get("audio"):
            audio_count += 1
        if ev.get("visual"):
            visual_count += 1
        if ev.get("rag"):
            rag_count += 1
        if seg.get("review_required"):
            review_required_count += 1
        if "confidence" in seg:
            confidences.append(seg["confidence"])

    total = len(all_segments)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 证据类型分布
    ax = axes[0]
    categories = ["音频证据", "视觉证据", "RAG 术语"]
    counts = [audio_count, visual_count, rag_count]
    pcts = [c / total * 100 for c in counts]
    colors = ["#4ECDC4", "#FF6B6B", "#FFD93D"]
    bars = ax.bar(categories, pcts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("覆盖片段占比 (%)")
    ax.set_title(f"证据来源类型分布 (共 {total} 片段)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # Confidence 分布
    ax = axes[1]
    ax.hist(confidences, bins=15, color="#6C5CE7", edgecolor="black",
            linewidth=0.5, alpha=0.8)
    ax.axvline(np.mean(confidences), color="red", linestyle="--",
               label=f"均值={np.mean(confidences):.2f}")
    review_pct = review_required_count / total * 100
    ax.set_xlabel("置信度分数")
    ax.set_ylabel("片段数")
    ax.set_title(f"片段置信度分布\n(需复核比例: {review_pct:.1f}%)")
    ax.legend()

    plt.tight_layout()
    save_figure(fig, "fig8_evidence_distribution")
    plt.close(fig)


def print_summary_statistics(stats: dict, adv_report: list[dict]) -> None:
    """打印汇总统计信息。"""
    print("\n" + "=" * 60)
    print("2. 汇总统计")
    print("=" * 60)

    # FGSM
    wers = [r["word_error_rate"] for r in adv_report]
    snrs = [r["snr_db"] for r in adv_report]
    print(f"\n--- FGSM 对抗攻击 (n={len(adv_report)}) ---")
    print(f"  WER: 均值={np.mean(wers):.4f}, 标准差={np.std(wers):.4f}, "
          f"最小={min(wers):.4f}, 最大={max(wers):.4f}")
    print(f"  SNR: 均值={np.mean(snrs):.2f} dB, 范围=[{min(snrs):.2f}, {max(snrs):.2f}]")

    # 同音词
    homo = stats["homophone"]
    if homo:
        print(f"\n--- 同音词替换 (n={len(homo)}) ---")
        for ratio in sorted(set(h["ratio"] for h in homo)):
            subset = [h for h in homo if h["ratio"] == ratio]
            avg_word = np.mean([h["word_change_pct"] for h in subset])
            avg_char = np.mean([h["char_change_pct"] for h in subset])
            print(f"  替换率={ratio:.0%}: 平均词变化={avg_word:.1f}%, 平均字符变化={avg_char:.1f}%")

    # 噪声
    noise = stats["noise"]
    if noise:
        print(f"\n--- 噪声/混响 (n={len(noise)}) ---")
        from collections import Counter
        counts = Counter((n["noise_type"], n["snr_db"]) for n in noise)
        for (nt, snr), cnt in sorted(counts.items()):
            print(f"  {nt} SNR={snr}dB: {cnt} 个样本")


def main():
    print("=" * 60)
    print("阶段一：合成受损数据描述性分析")
    print("=" * 60)

    # 加载数据
    stats = load_corruption_stats()
    adv_report = load_adversarial_report()
    homo_report = load_homophone_report()
    metadata = load_videos_metadata()

    # 配置中文字体
    setup_chinese_font()

    # 1. 数据集总览
    analyze_dataset_overview(metadata)

    # 2. 汇总统计
    print_summary_statistics(stats, adv_report)

    # 3. 生成图表
    print("\n" + "=" * 60)
    print("3. 生成可视化图表")
    print("=" * 60)

    print("\n[Fig.1] 三类受损条件 WER 箱线图...")
    plot_baseline_wer_boxplot(stats, adv_report)

    print("\n[Fig.2] 受损程度热力图...")
    plot_corruption_heatmap(stats, adv_report)

    print("\n[Fig.3] 同音词替换率对比...")
    plot_homophone_analysis(stats)

    print("\n[Fig.3b] 同音词替换质量分布...")
    plot_homophone_quality(homo_report)

    print("\n[Fig.4] FGSM SNR vs WER 散点图...")
    plot_adversarial_analysis(adv_report)

    print("\n[Fig.8] 证据来源类型分布...")
    plot_evidence_distribution()

    print("\n" + "=" * 60)
    print("阶段一完成！图表已保存至 analysis/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
