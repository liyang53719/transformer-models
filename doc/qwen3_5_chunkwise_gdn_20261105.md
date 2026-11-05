# Qwen3.5 chunk-wise GDN

日期：2026-11-05。实现为 `qwen3_5.layer.gatedDeltaNetChunkwise`。

## 实现与接入

`linearAttention` 默认使用 `GdnImplementation = 'auto'`：

- 输入不少于 16 tokens 时，使用默认 64-token chunk-wise GDN。
- 较短输入和单 token decode 使用原递推算子 `gatedDeltaNet`。
- `GdnImplementation = 'recurrent'` 可强制使用递推路径。
- `GdnImplementation = 'chunkwise'` 对所有多 token 输入使用分块路径。
- `GdnChunkSize` 可调整块长，默认 64。

```matlab
parameters.Hyperparameters.GdnImplementation = 'auto';
parameters.Hyperparameters.GdnChunkSize = 64;
```

接口保持 Q/K 为 `keyDim x heads x tokens`、V 为 `valueDim x heads x tokens`，
状态为 `keyDim x valueDim x heads`。Q/K 由调用者归一化，query scale 只应用一次。
所有权重、激活和状态继续使用浮点通路。输出和末态保持原接口，支持非零初态、
短尾块和跨调用续算。卷积、gated RMSNorm、输出投影及 cache 结构沿用现有接口。

## 与递推式的关系

令每个 chunk 的 Q/K/V 按 token 排成矩阵行，输入状态为 S，
`G_i = sum(g_1 ... g_i)`，`D_ij = exp(G_i - G_j)`（仅保留下三角）。
记 `E = diag(exp(G))`、`B = diag(beta)`。

块内 delta 满足：

```text
L     = strict_lower((B K K') .* D)
Delta = (I + L) \ (B (V - E K S))
Y     = (E Q S + ((Q K') .* D) Delta) / sqrt(keyDim)
Snext = exp(G_last) S + K' diag(exp(G_last - G)) Delta
```

实现通过 `pagemldivide` 求解单位下三角系统，不显式构造逆矩阵。
`pagemtimes` 批量处理各个 head；仅在 chunk 边界推进状态。
最后不足 64 tokens 的块直接按实际长度计算。
上三角在指数运算前设为负无穷，避免无效上三角项发生指数溢出。

该写法与 [Gated Delta Networks 论文](https://arxiv.org/abs/2412.06464)
及 [Transformers 5.3.0 的参考分块算法](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
数学等价；FP32 的运算顺序发生变化，因此进行容差比较。

## 实测性能

测试环境为 MATLAB R2024a Update 4、CPU、8 个计算线程。输入来自本地真实
Qwen3.5-0.8B 第一层投影、因果卷积和归一化后的 Q/K/V、decay 与 beta。
基准使用相同的冻结输入，调用 `timeit(..., 2)`，包含两个输出及初始化成本，
排除模型加载和前置投影。下表为 GDN 核心算子耗时。

| Tokens | 递推 / ms | Chunk64 / ms | 耗时比 |
|---:|---:|---:|---:|
| 1 | 0.232 | 0.341 | 0.68x |
| 16 | 1.809 | 0.560 | 3.23x |
| 32 | 3.314 | 0.841 | 3.94x |
| 64 | 6.404 | 2.008 | 3.19x |
| 128 | 12.679 | 4.367 | 2.90x |
| 241 | 24.624 | 8.235 | 2.99x |
| 1024 | 106.384 | 29.034 | 3.66x |

Profiler 显示递推路径的状态缩放、delta 预测、rank-one 更新与 query 读取，
在 1024 tokens 时各执行 1024 次。分块路径将块内计算组织成矩阵运算。
单 token 的分块开销更大，因此 auto 路径保留递推 decode。

1024-token 核心算子与递推式相比，最大输出绝对误差为 `2.682209e-7`，
最大末态绝对误差为 `1.287460e-5`。这些耗时比针对 GDN 核心算子。

## 数值验证

- 16 项算子测试通过，覆盖 1/17/64/65/129 tokens、不同 key/value 维度、
  非零初态、跨调用状态传递、强衰减、零 beta 和 linearAttention 接入。
- 默认 auto 路径的五组完整模型回归通过：6290 项张量比较、347/347 top-1 一致。
- 全模型对 Transformers/PyTorch 的最大 logits 绝对误差为 `4.863739e-5`。
- 独立 30-token 摘要生成的输入 IDs、生成 IDs 和文本完全一致。
- 新增及修改的 MATLAB 算子通过 Code Analyzer。

模型回归覆盖 1、17、39、241-token prefill 和缓存 decode。
1024-token 数据用于上述第一层 GDN 核心基准与数值对照。

复现全模型检查：

```matlab
TestQwen3_5Numerics( ...
    fullfile('work', 'models', 'Qwen3.5-0.8B-FULL'), ...
    fullfile('work', 'tmp', 'qwen3_5_transformers', 'references'));
TestQwen3_5Summarize
```

工作目录中的 `gdn_baseline.log`、`gdn_compare_bench.log`、`gdn_chunk_tests.log`、
`gdn_chunk_model.log` 和对应 MAT 数据保留本轮测量；模型输入继续默认只加载本地权重。
