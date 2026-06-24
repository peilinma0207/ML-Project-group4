# 数据分析方法说明书

## 1. 分析总览

本分析围绕技术报告《面向多说话人视频转写修复的多证据 Agent 框架》中的三个研究问题（RQ1-RQ3），对 100 个合成受损样本进行系统性的定量评估。分析分为三个阶段，依次执行。

### 数据集构成

| 受损类型 | 样本数 | 参数 | 数据来源 |
|----------|--------|------|---------|
| 噪声/混响 | 60 | 白噪声/混响 × SNR 3/10/20 dB | `data/corrupted/noise/audio/` |
| 同音词替换 | 27 | 替换率 10%/25%/50%（video_01 无有效替换） | `data/corrupted/homophone/text/` |
| FGSM 对抗 | 10 | ε=0.01 | `data/corrupted/adversarial/audio/` |

参考转写：`data/transcripts_ground_truth/`，共 10 个视频的人工校对文本。

---

## 2. 阶段一：描述性分析

**脚本**：`01_corruption_descriptive.py`
**输入**：`data/corrupted/metadata/` 下的 JSON 元数据
**可直接运行**，不依赖 GPU 或外部 API。

### 2.1 数据集总览

从 `videos_metadata.csv` 提取 10 个视频的说话人、主题、时长区间，以表格形式打印。

### 2.2 FGSM 对抗样本分析

- **数据源**：`adversarial_report.json`（含每个视频的原始转写、对抗转写、WER、SNR）
- **箱线图**：展示 10 个视频的 WER 分布，标注均值
- **柱状图**：各视频 WER 对比，颜色编码严重程度
- **散点图 + 线性回归**：SNR vs WER，计算 Pearson 相关系数 r，检验信噪比与识别崩溃的关联

### 2.3 同音词替换分析

- **数据源**：`corruption_stats.json`（homophone 字段）+ `homophone_report.json`（替换详情）
- **分组柱状图**：目标替换率（10%/25%/50%）vs 实际词级/字符级变化百分比，附标准差误差线，叠加理想替换率参考线
- **箱线图**：展示相同替换率下各视频间的变异度
- **质量分布直方图**：phonetic_similarity 和 edit_similarity 的频次分布，评估替换的自然度

### 2.4 受损程度热力图

- **矩阵**：video_id（行）× corruption_condition（列），共 10×10
- **数值映射**：噪声用 SNR 倒数作为严重度代理（SNR=3→0.9, 10→0.5, 20→0.2）；同音词用实际词变化百分比；FGSM 用 WER
- **颜色**：YlOrRd 色图，0=轻微，1=严重，每格标注数值

### 2.5 证据来源分布

- **数据源**：`output/f72a56bc/script.json`（pipeline 输出示例）
- **柱状图**：音频/视觉/RAG 三类证据的片段覆盖率
- **直方图**：片段置信度分数分布，标注 review_required 比例

### 输出图表

| 文件名 | 内容 |
|--------|------|
| `fig1_baseline_wer_boxplot.png` | 三类受损条件 WER 箱线图 |
| `fig2_corruption_heatmap.png` | video × condition 热力图 |
| `fig3_homophone_substitution_rates.png` | 同音词替换率对比 |
| `fig3b_homophone_quality.png` | 同音词语音/编辑相似度分布 |
| `fig4_fgsm_snr_vs_wer.png` | FGSM SNR vs WER 散点图 |
| `fig8_evidence_distribution.png` | 证据来源类型分布 |

---

## 3. 阶段二：批量实验执行

**脚本**：`02_batch_evaluation.py`
**依赖**：WhisperX（GPU）、LLM API（用于 pipeline 修复）
**运行参数**：`--mode noise|pipeline|ablation|all`，`--model-size tiny`，`--device cuda`

### 3.1 噪声样本 WER 评估

- 对 60 个噪声音频运行 WhisperX（默认 tiny 模型），转写后与 ground truth 计算 WER 和 CER
- **WER 计算方法**：基于词级编辑距离（Levenshtein distance），WER = (S+D+I)/N，其中 S=替换、D=删除、I=插入、N=参考词数
- **CER 计算方法**：字符级编辑距离，去除空格后逐字符对齐
- **回退机制**：WhisperX 不可用时，使用 SNR 经验回归模型估算 WER：`WER ≈ 0.7 × exp(-0.08 × SNR)`，reverb 额外加 0.05 惩罚

### 3.2 全量 Pipeline 运行

对 97 个受损样本运行完整修复流水线，提取以下指标：

| 指标 | 计算方法 |
|------|---------|
| 修复后 WER | 拼接所有 segment 文本与 ground truth 的词级编辑距离 |
| 修复后 CER | 字符级编辑距离 |
| 术语精确率 | glossary.json 中术语在修复文本中的命中率 |
| 术语召回率 | 修复文本中覆盖的术语占全部术语的比例 |
| 术语 F1 | 2×P×R/(P+R) |
| 平均置信度 | 各 segment confidence 的算术平均 |
| 复核比例 | review_required=true 的 segment 占比 |
| 证据类型分布 | audio/visual/rag 各自覆盖的 segment 数 |

- **回退机制**：Pipeline 模块不可用时，使用统计模型模拟修复效果（基于受损类型调整改善幅度：噪声 10-25%、同音词 25-45%、对抗 15-30%）

### 3.3 消融实验

4 个实验条件，逐步叠加模块：

| 条件 | 描述 | 启用模块 |
|------|------|---------|
| baseline | ASR 原始输出 | WhisperX |
| +visual | 加入视觉证据 | WhisperX + OCR/VLM |
| +retrieval | 加入术语检索 | WhisperX + RAG glossary |
| full | 完整流水线 | WhisperX + OCR/VLM + RAG + Constrained Repair |

每个条件在每个样本上运行，记录 WER 和术语准确率。模拟模式下，各模块的改善效果按受损类型差异化建模（例如：视觉模块对噪声改善 10-20%，对同音词仅 3-8%；术语检索反之）。

### 输出

- `results/evaluation_results.json`：包含 97 条完整评估记录（含基线指标、修复指标、消融结果）

---

## 4. 阶段三：统计检验

**脚本**：`03_statistical_analysis.py`
**输入**：`results/evaluation_results.json`
**依赖**：scipy.stats

### 4.1 RQ1：不确定性驱动路由 vs 均匀处理

**假设**：在固定计算预算下，基于置信度的局部抽帧（不确定性路由）比全局均匀抽帧产生更低的 WER。

- **比较对**：baseline（均匀处理）vs +visual（不确定性路由）
- **检验方法**：Wilcoxon 符号秩检验（单侧，H₁: baseline WER > +visual WER）
  - 选择理由：配对样本，WER 分布不假定正态，非参数检验更稳健
- **效应量**：rank-biserial correlation r = 1 - 2W/(n(n+1))
- **显著性水平**：α = 0.05

**可视化**：
- 配对比较图：每个样本的 baseline → +visual WER 变化（绿=改善，红=退化）
- WER 改善幅度直方图
- 置信度阈值权衡曲线（双轴：WER + 帧抽取数 vs 阈值）

### 4.2 RQ2：术语检索对实体与稀有词的效果

**假设**：显式术语检索对同音词替换类型的 WER 改善显著大于对噪声类型。

- **比较对**：baseline vs +retrieval
- **检验方法**：
  - Wilcoxon 符号秩检验（WER 比较，单侧）
  - Wilcoxon 符号秩检验（术语准确率比较，单侧，H₁: +retrieval > baseline）
- **分层分析**：按受损类型（noise/homophone/adversarial）分别计算 ΔWER，检验术语检索在不同条件下的差异化效果

**可视化**：
- 受损类型 × 有无检索的分组柱状图（附误差线）
- 术语准确率对比柱状图

### 4.3 RQ3：证据约束修复 vs 自由改写

**假设**：约束修复虽未必在所有样本上产生最大 WER 改善，但在 hallucination 控制上显著优于自由改写。

- **比较对**：full（约束修复）vs 模拟的 free-form rewrite（无约束 LLM 改写）
- **检验方法**：
  - Wilcoxon 符号秩检验（WER 比较）
  - Fisher 精确检验（hallucination 发生率的 2×2 列联表，单侧）
    - 列联表构造：[约束修复 hallucination 数, 约束修复正常数; 自由改写 hallucination 数, 自由改写正常数]
    - 选择理由：小样本分类变量的精确概率检验

**可视化**：
- WER 箱线图对比
- Hallucination 发生率柱状图
- 多维指标雷达图（WER↓、术语准确率↑、结构保真↑、Hallucination 风险↓、可追溯性↑）

### 输出

| 文件名 | 内容 |
|--------|------|
| `fig5_rq1_uncertainty_routing.png` | RQ1 配对比较 + 改善分布 + 阈值权衡 |
| `fig6_rq2_terminology_retrieval.png` | RQ2 分组柱状图 + 术语准确率 |
| `fig7_rq3_constrained_vs_freeform.png` | RQ3 WER 箱线图 + Hallucination 率 + 雷达图 |
| `fig_ablation_overall.png` | 消融实验全局总览 |
| `results/statistical_results.json` | RQ1-RQ3 检验统计量、p 值、效应量 |

---

## 5. 共享工具函数

**文件**：`utils.py`

| 函数 | 用途 |
|------|------|
| `compute_wer(ref, hyp)` | 词级编辑距离 WER |
| `compute_cer(ref, hyp)` | 字符级编辑距离 CER |
| `compute_terminology_accuracy(text, glossary)` | 术语 P/R/F1 |
| `load_corruption_stats()` | 加载 corruption_stats.json |
| `load_adversarial_report()` | 加载 adversarial_report.json |
| `load_homophone_report()` | 加载 homophone_report.json |
| `load_ground_truth(video_id)` | 加载 ground truth 转写（去除时间戳行） |
| `load_pipeline_output(output_dir)` | 加载 pipeline 的 script.json |
| `setup_chinese_font()` | 配置 matplotlib 中文字体 |
| `save_figure(fig, name)` | 保存图表到 figures/ 目录 |

---

## 6. 运行指南

```bash
# 阶段一：立即可运行，基于现有元数据
uv run python analysis/01_corruption_descriptive.py

# 阶段二（真实 Pipeline 评估）：对受损样本运行完整 Agent 流水线
# 支持 --resume（跳过已成功样本）和 --workers N（并发数）
uv run python analysis/run_evaluation.py --resume --workers 5

# 阶段二补充（消融实验）：跑 baseline 和 +retrieval 条件（跳过 +visual）
uv run python analysis/run_ablation.py --workers 5

# 阶段三：依赖 evaluation_results.json（含 ablation_results 字段）
uv run python analysis/03_statistical_analysis.py
```

输出图表保存在 `analysis/figures/`，结果数据保存在 `analysis/results/`。

### 实际执行记录

**环境**：
- GPU: CUDA (PyTorch 2.8.0+cu128)
- ASR: WhisperX tiny 模型
- LLM/VLM: Qwen/Qwen3.6-27B via SiliconFlow API (`https://api.siliconflow.cn/v1`)
- 并发: 3-5 workers (ThreadPoolExecutor)

**阶段二执行情况**（2026-06-24 ~ 2026-06-25）：
- 总样本 90 个（noise 54 + homophone 27 + adversarial 9），video_01 无 ground truth 已排除
- 成功完成 83/90（93%），7 个因 API 超时（segment 过多导致 prompt 过长）未完成
- 未完成样本：video_07_white_snr3, video_08_white_snr10, video_09_white_snr20, video_10_reverb_snr3, video_02_homophone_r10, video_07_homophone_r25, video_07_homophone_r50

**阶段二结果摘要**：
| 类型 | n | 基线 WER | 修复 WER | 改善比例 |
|------|---|---------|---------|---------|
| noise | 50 | — | 0.361 ± 0.416 | — |
| homophone | 24 | 0.111 ± 0.067 | 0.016 ± 0.012 | 100% (24/24) |
| adversarial | 9 | 0.726 ± 0.096 | 0.718 ± 0.104 | 66.7% (6/9) |

**消融实验设计**（跳过 +visual 条件）：
| 条件 | 描述 | 启用模块 |
|------|------|---------|
| baseline | ASR 原始输出 | WhisperX (--skip-vlm --skip-repair) |
| +retrieval | ASR + 术语检索 + 修复 | WhisperX + RAG + Repair (--skip-vlm) |
| full | 完整流水线 | WhisperX + RAG + VLM + Repair |

**API 可靠性措施**：
- 超时设置 600s
- 自动重试 5 次（覆盖 429/500/502/503/504/TimeoutError/OSError）
- 指数退避：10s → 20s → 40s → 80s
- `_is_valid_pipeline_output()` 检测失败结果（confidence=0 或 visual=0），`--resume` 自动重跑

---

## 7. 局限性与注意事项

1. **模拟数据**：当 WhisperX 或 Pipeline 模块不可用时，阶段二使用统计模型生成模拟结果。模拟值基于文献经验和合理假设，但不应作为最终论文数据。获取真实结果后应重新运行阶段三。
2. **噪声 WER 估算**：回退模型 `WER ≈ 0.7 × exp(-0.08 × SNR)` 是经验近似，实际 WER 受视频内容、说话人语速等因素影响。
3. **Free-form Rewrite 模拟**：RQ3 中的自由改写条件通过统计模拟生成（WER 为 baseline 的 55-80%，hallucination 率约 22%），需在实际 LLM 实验中验证。
4. **样本量**：video_01 的同音词替换生成了 0 次替换（无有效候选词），因此同音词样本实际为 27 而非 30。
5. **PyTorch 兼容性**：WhisperX 的 pyannote VAD 模型在 PyTorch ≥ 2.6 下需设置 `torch.serialization.add_safe_globals` 或降级 PyTorch 版本。
