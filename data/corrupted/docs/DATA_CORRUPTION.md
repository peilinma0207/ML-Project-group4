# Data Corruption Methods

本文档描述项目中用于合成受损数据的三种方法。目标是模拟真实场景中 ASR 系统可能遇到的各类干扰，生成不同损坏程度的测试数据。

## 目录

1. [方法概览](#方法概览)
2. [方法一：音频加噪](#方法一音频加噪)
3. [方法二：同音词替换](#方法二同音词替换)
4. [方法三：对抗扰动 FGSM](#方法三对抗扰动-fgsm)
5. [输出结构](#输出结构)
6. [使用方法](#使用方法)
7. [统计结果](#统计结果)

---

## 方法概览

| 方法 | 输入 | 输出 | 损坏维度 | 难度 |
|------|------|------|----------|------|
| 音频加噪 | `.wav` 音频 | 加噪后的 `.wav` | 音频质量下降 | 低 |
| 同音词替换 | GT 文本 `.txt` | 替换后的 `.txt` | 文本层面错误 | 低 |
| 对抗扰动 | `.wav` 音频 | 扰动后的 `.wav` | 模型识别混乱 | 高 |

三种方法从不同角度制造"受损数据"，用于测试下游 ASR 修复 Agent 的能力。

---

## 方法一：音频加噪

### 原理

使用 ffmpeg 在原始音频上叠加噪声，模拟真实环境中的录音质量下降。

### 噪声类型

#### 1. 白噪声 (White Noise)

使用 ffmpeg 的 `anoisesrc` 滤波器生成粉红噪声（pink noise），叠加到原始音频上。

```bash
ffmpeg -i input.wav \
  -filter_complex "anoisesrc=d={duration}:c=pink:r=16000:a={amplitude}[noise];[in][noise]amix=inputs=2:duration=first[out]" \
  -map "[out]" output.wav
```

- `c=pink`：粉红噪声（比白噪声更接近真实环境噪声）
- `a={amplitude}`：噪声幅度，由 SNR 计算得出
- `amplitude = 10^(-snr/20)`，SNR 越低噪声越大

#### 2. 混响 (Reverb)

使用 ffmpeg 的 `aecho` 滤波器模拟室内混响环境（如会议室、大厅）。

```bash
ffmpeg -i input.wav \
  -af "aecho={dry_gain}:{wet_gain}:40|60|80:0.3|0.25|0.2" \
  output.wav
```

- 多延迟叠加：40ms、60ms、80ms，模拟墙壁反射
- 衰减系数：0.3、0.25、0.2，模拟不同反射面吸收

### SNR 等级

| SNR (dB) | 噪声强度 | 效果描述 |
|----------|----------|----------|
| 20 | 轻度 | 背景有轻微噪声，语音基本清晰 |
| 10 | 中度 | 明显噪声干扰，部分词汇难以辨认 |
| 3 | 重度 | 噪声几乎淹没语音，严重影响识别 |

### 生成文件数

- 10 个视频 × 2 种噪声类型 × 3 个 SNR 等级 = **60 个文件**

---

## 方法二：同音词替换

### 原理

自动找到文本中可替换的词，用发音相同或相近的词替换，模拟 ASR 系统常见的同音词识别错误。

### 候选词来源

#### 1. CMU Pronouncing Dictionary

使用 `pronouncing` 库查询 CMU 发音词典，构建反向索引（发音 → 词列表），快速找到同音词。

```python
import pronouncing

# 获取单词发音
phones = pronouncing.phones_for_word("machine")
# → ['M AH0 SH IY1 N']

# 通过反向索引找同音词
# "know" 和 "no" 发音相同: N OW1
# "their" 和 "there" 发音相同: DH EH1 R
```

#### 2. 内置同音词表

对于 CMU 词典未覆盖的常见同音词，维护一个内置映射表（约 100 对）。

```python
BUILTIN_HOMOPHONES = {
    "their": ["there", "they're"],
    "your": ["you're"],
    "to": ["too", "two"],
    "write": ["right", "rite"],
    "know": ["no"],
    # ... 约 100 对
}
```

### 替换策略

1. **分词**：将 GT 文本按空格分词
2. **筛选可替换词**：对每个词查 CMU 发音词典，有同音词候选的才可替换
3. **按比例选择**：根据设定比例随机选择要替换的词
4. **替换**：从候选列表中随机选一个替换
5. **保留大小写**：替换词匹配原词的大小写模式

### 替换比例

| 比例 | 含义 | 平均词变化率 | 平均字符变化率 |
|------|------|-------------|---------------|
| 10% | 轻度损坏 | ~3.6% | ~1.2% |
| 25% | 中度损坏 | ~9.2% | ~3.5% |
| 50% | 重度损坏 | ~18.7% | ~7.0% |

### 生成文件数

- 10 个视频 × 3 个比例 = **30 个文件**

### 示例

原文：
```
So Arthur Samuel was the father of machine learning
```

25% 替换后：
```
So Arthur Samuel was the farther of machine learn
```

- `father` → `farther`（发音相似）
- `learning` → `learn`（CMU 词典中的误识别变体）

---

## 方法三：对抗扰动 FGSM

### 原理

**FGSM (Fast Gradient Sign Method)** 是一种对抗攻击方法。通过对音频波形添加微小的、基于梯度的扰动，使 ASR 模型的输出发生错误。

### 技术细节

#### 模型

- 使用 **OpenAI Whisper tiny** 模型（39M 参数）
- 在 Apple M4 MPS 上运行（或 CPU 回退）

#### 攻击流程

```
原始音频 waveform x
      │
      ▼
计算 Mel 频谱图 mel(x)
      │
      ▼
Whisper Encoder → encoder_out
      │
      ▼
计算损失 L = ||encoder_out||²
      │
      ▼
反向传播求梯度 ∂L/∂x
      │
      ▼
FGSM 扰动: δ = ε · sign(∂L/∂x)
      │
      ▼
对抗音频: x_adv = clamp(x + δ, -1, 1)
```

#### 核心代码

```python
# 计算 Mel 频谱图（可微分）
mel = whisper.log_mel_spectrogram(audio_grad)

# 前向传播
encoder_out = model.encoder(mel.unsqueeze(0))

# 最大化 encoder 输出（非定向攻击）
loss = encoder_out.pow(2).mean()

# 反向传播
loss.backward()

# FGSM 扰动
perturbation = epsilon * audio_grad.grad.sign()
audio_adv = torch.clamp(audio_padded + perturbation, -1.0, 1.0)
```

### 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| epsilon | 0.01 | 扰动幅度 |
| max_perturbation | 0.01 | 最大扰动绝对值 |
| 平均 SNR | 33.1 dB | 信噪比（越高越不可感知） |

### 特点

- **不可感知**：扰动幅度极小（SNR > 30 dB），人耳几乎听不出差异
- **有效性**：微小扰动能显著改变 Whisper 的转录结果
- **梯度驱动**：基于模型梯度计算，针对性强

### 生成文件数

- 10 个视频 × 1 个 epsilon = **10 个文件**

---

## 输出结构

```
data/corrupted/
├── audio_noise/                    # 加噪音频
│   ├── video_01_white_snr20.wav    # 白噪声 SNR=20dB
│   ├── video_01_white_snr10.wav    # 白噪声 SNR=10dB
│   ├── video_01_white_snr3.wav     # 白噪声 SNR=3dB
│   ├── video_01_reverb_snr20.wav   # 混响 SNR=20dB
│   ├── video_01_reverb_snr10.wav   # 混响 SNR=10dB
│   ├── video_01_reverb_snr3.wav    # 混响 SNR=3dB
│   └── ...                         # 共 60 个文件
├── text_homophone/                 # 同音词替换文本
│   ├── video_01_homophone_r10.txt  # 10% 替换
│   ├── video_01_homophone_r25.txt  # 25% 替换
│   ├── video_01_homophone_r50.txt  # 50% 替换
│   └── ...                         # 共 30 个文件
├── audio_adversarial/              # 对抗扰动音频
│   ├── video_01_adv_eps0.0100.wav
│   └── ...                         # 共 10 个文件
├── corruption_stats.json           # 统计数据
└── homophone_report.json           # 同音词替换详情
```

---

## 使用方法

### 运行全部

```bash
python3 -m src.agent.corrupt.runner
```

### 只运行部分方法

```bash
# 跳过对抗攻击（最快）
python3 -m src.agent.corrupt.runner --skip-adversarial

# 只运行同音词替换
python3 -m src.agent.corrupt.runner --skip-noise --skip-adversarial
```

### 自定义参数

```bash
# 自定义 SNR 等级
python3 -m src.agent.corrupt.runner --noise-snrs 15 5

# 自定义替换比例
python3 -m src.agent.corrupt.runner --homophone-ratios 0.1 0.2 0.3 0.4

# 自定义对抗攻击强度
python3 -m src.agent.corrupt.runner --adv-epsilon 0.05

# 使用 GPU
python3 -m src.agent.corrupt.runner --device cuda
```

### 查看统计

```bash
python3 -m src.agent.corrupt.stats
```

---

## 统计结果

### 同音词替换

| 比例 | 平均词变化率 | 平均字符变化率 |
|------|-------------|---------------|
| 10% | 3.6% | 1.2% |
| 25% | 9.2% | 3.5% |
| 50% | 18.7% | 7.0% |

### 音频加噪

| 噪声类型 | SNR (dB) | 文件数 |
|----------|----------|--------|
| 白噪声 | 20 | 10 |
| 白噪声 | 10 | 10 |
| 白噪声 | 3 | 10 |
| 混响 | 20 | 10 |
| 混响 | 10 | 10 |
| 混响 | 3 | 10 |

### 对抗扰动

| 参数 | 值 |
|------|-----|
| epsilon | 0.01 |
| 平均最大扰动 | 0.01 |
| 平均 SNR | 33.1 dB |
| 文件数 | 10 |

---

## 依赖

- `ffmpeg`：音频处理
- `pronouncing`：CMU 发音词典
- `rapidfuzz`：模糊字符串匹配
- `openai-whisper`：对抗攻击目标模型
- `torch`：梯度计算
- `scipy`：音频 I/O
