# 大模型显存估算 — 计算方式说明

## 总显存 = 模型权重 + KV Cache + 推理开销

---

## 1. 模型权重显存

```
权重显存(GB) = FP16模型大小(GB) × 量化压缩比
```

| 量化方式 | bytes/参数 | 压缩比 |
|----------|-----------|--------|
| FP16     | 2         | 1      |
| BF16     | 2         | 1      |
| FP8      | 1         | 0.5    |
| INT8     | 1         | 0.5    |
| FP4      | 0.5       | 0.25   |
| INT4     | 0.5       | 0.25   |

模型预设的 size 值均为 FP16 精度下的估算大小（参数 × 2 bytes）。选择量化后自动乘以对应压缩比。

> FP4 为 4-bit 浮点量化，与 INT4 同级别（0.5 byte/param），但保留了浮点格式的动态范围优势，在部分推理场景中精度损失更小。

---

## 2. KV Cache 显存

每个 session 独立维护一份 KV Cache，大小取决于模型架构和上下文长度。

### 公式

```
KV Cache(GB) = 2 × numLayers × (hiddenDim × kvHeads/attnHeads) × contextLen × bytesPerElem / 1e9
```

### 各参数说明

| 参数 | HF 配置名 | 含义 | 典型值（GLM-5.1） |
|------|----------|------|-------------------|
| `2` | — | K 和 V 各一份 | — |
| `numLayers` | `num_hidden_layers` | Transformer 层数 | 62 |
| `hiddenDim` | `hidden_size` | 隐藏层维度 | 7168 |
| `kvHeads / attnHeads` | `num_key_value_heads` / `num_attention_heads` | GQA KV 头比例 | 8/64 = 0.125 |
| `hiddenDim × kvRatio` | — | KV 投影的实际维度 | 7168 × 0.125 = 896 |
| `contextLen` | — | 输入 + 输出 token 总数 | 可选项：16K~1M |
| `bytesPerElem` | — | FP16=2, FP8=1, INT4/FP4=0.5 | 2 |

## 7. KV Cache 格式说明

| 格式 | bytes/elem | 说明 | 适用硬件 |
|------|-----------|------|---------|
| FP16 | 2 | 传统默认，无损 | 所有 GPU |
| FP8 | 1 | 当前主流（OCP E5M2/E4M3），vLLM/SGLang 默认 | H100/H200/B200/H20 等 |
| INT4/FP4 | 0.5 | 量化 4-bit，相对 FP8 再减半显存 | B200 (NVFP4) / 支持 MXFP4 硬件 |

当前主流推理框架（vLLM、SGLang）默认使用 FP8 KV Cache，FP16 为传统基线。
INT4/FP4（如 NVIDIA NVFP4、OCP MXFP4）是 2025-2026 年新兴标准，在 Blackwell 及更高端硬件上支持，可在精度损失 <1% 的前提下将 KV Cache 显存再降低 50%。

> 非 GQA 模型（MHA）中 `kvHeads = attnHeads`，`kvRatio = 1`，KV Cache 将放大 8 倍。

### 示例：GLM-5.1 FP16, 4K context

```
KV Cache = 2 × 62 × 896 × 4096 × 2 / 1e9 ≈ 0.91 GB/session
```

---

## 3. 推理开销

包括激活值（activation memory）、临时 buffer、CUDA kernel 工作空间等。

```
开销(GB) = 权重显存(GB) × 开销比例
```

| 场景 | 建议比例 |
|------|---------|
| 高度优化（vLLM PagedAttention） | 8% |
| vLLM 默认 | 12% |
| SGLang | 15% |
| 保守估计 | 20% |

---

## 4. 单 Session 总显存

```
单Session(GB) = 权重显存 + KV Cache + 推理开销
```

权重显存仅在加载时一次性占用，并发 session 共享权重，仅需额外分配 KV Cache + 开销。

---

## 5. 集群与并发计算

### 总显存

```
集群总显存(GB) = 单卡显存 × GPU 数量
可用显存(GB)   = 集群总显存 - 系统预留
```

### 最少 GPU 数量

```
最少GPU数 = ceil((权重显存 + 预留) / 单卡显存)
```

模型权重必须能完整装入集群总显存（扣除预留），同时每张卡不超限。

### 最大并发 Session

```
最大并发 = floor((可用显存 - 权重显存) / (KV Cache + 开销))
```

权重只加载一份，所有 session 共享。剩余显存按每个 session 所需 KV Cache + 开销计算并发容量。

---

## 6. 模型预设数据

| 模型 | FP16 | hidden dim | layers | Q heads | KV heads | 最大上下文 |
|------|------|-----------|--------|---------|----------|-----------|
| DeepSeek V4-Pro | 3200 GB | 7168 | 61 | 64 | 8 | 1M |
| DeepSeek V4-Flash | 568 GB | 4096 | 43 | 32 | 8 | 1M |
| DeepSeek V3.2 | 1342 GB | 7168 | 67 | 64 | 8 | 128K |
| DeepSeek V3.1 | 1342 GB | 7168 | 67 | 64 | 8 | 128K |
| DeepSeek V3 | 1342 GB | 7168 | 67 | 64 | 8 | 128K |
| DeepSeek R1 | 1320 GB | 7168 | 67 | 64 | 8 | 128K |
| GLM-5 | 1490 GB | 7168 | 62 | 64 | 8 | 200K |
| GLM-5.1 | 1488 GB | 7168 | 62 | 64 | 8 | 200K |
| Qwen3.6 35B-A3B | 70 GB | 3072 | 40 | 48 | 8 | 1M |
| Qwen3 235B | 470 GB | 5120 | 60 | 64 | 8 | 128K |
| Qwen3.5 235B | 470 GB | 5120 | 60 | 64 | 8 | 128K |
| Minimax M2.7 | 460 GB | 3072 | 62 | 48 | 8 | 200K |

DeepSeek V4 系列基于 2026 年 4 月发布的技术报告，采用满血万亿参数 MoE + 华为昇腾 CANN 算力。
Qwen3.6 35B-A3B 是阿里 2026 年发布的 Agentic Coding 专用模型。

## 8. GPU 预设列表

| GPU | 显存 | 类型 |
|-----|------|------|
| H200 141G | 141 GB | NVIDIA |
| H20 141G | 141 GB | NVIDIA (中国特供) |
| B200 192G | 192 GB | NVIDIA |
| RTX 5090 | 32 GB | NVIDIA |
| RTX 4090 | 24 GB | NVIDIA |
| A710E | 96 GB | 阿里 PPU (HBM2e) |
| 真武810E | 96 GB | 阿里 PPU (HBM2e) |

A710E 与 真武810E 为阿里巴巴自研 PPU，采用 HBM2e 高带宽内存，单卡 96 GB。
