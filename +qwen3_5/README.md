# Qwen3.5-0.8B FP32 text decoder

This package follows the repository's `+qwen2` organization: `load`, `model`,
`generate`, `+tokenizer`, and explicit operators in `+layer`.

The implementation runs all 24 text-decoder layers: 18 Gated DeltaNet layers
and 6 gated full-attention layers. It supports one text sequence, full or
chunked prefill, and cached autoregressive decoding. Vision and MTP are outside
this package's scope.

## Local weights and precision

`qwen3_5.load` reads the original local safetensors directly in MATLAB.
BF16 values expand exactly to `single`; F32 values remain `single`.
There is no quantized matrix multiplication, activation quantization, or
Python inference in the MATLAB forward pass. This is FP32 evaluation of the
original checkpoint, rather than simulation of BF16 activation rounding.

The loader creates `Parameters.Hyperparameters` and `Parameters.Weights.h0`
through `h23`, matching the existing model packages. Embeddings are stored
as hidden-by-vocabulary, linear weights as output-by-input. The output head
shares the token embedding weights.

Checkpoint RMSNorm weights use the Qwen `1 + weight` convention. The loader
adds this offset to decoder, Q/K, and final normalization weights once.
Gated DeltaNet's `linear_attn.norm.weight` is already multiplicative and is
loaded without an offset. `A_log` becomes `-exp(A_log)` once.

## MATLAB use

Run from the repository root. Place the local checkpoint files in the ignored
`work/models/Qwen3.5-0.8B-FULL/` directory:

```matlab
modelFolder = fullfile('work', 'models', 'Qwen3.5-0.8B-FULL');
parameters = qwen3_5.load(modelFolder);
[logits, state] = qwen3_5.model(int32([760 6511 314 9338 369]), parameters);
[~, next] = max(logits(:, end));
[nextLogits, state] = qwen3_5.model(int32(next-1), parameters, state);
```

Token IDs are zero based. The first dimension of logits is the 248320-token
vocabulary. Passing an empty state starts a new sequence. State contains both
the Gated DeltaNet convolution/SSM state and the full-attention KV cache.
Only base MATLAB is required for loading weights and evaluating token IDs.

For text, the optional tokenizer uses Python's `tokenizers` package to read
the local tokenizer JSON, following the existing Python-tokenizer pattern:

```matlab
mdl.Parameters = parameters;
mdl.Tokenizer = qwen3_5.tokenizer.QwenTokenizer(modelFolder);
[text, ids] = generateSummary_Qwen3_5(mdl, 'The capital of France is', ...
    'TopK', 1, 'MaxNewTokens', 8);
```

The generation interface uses greedy decoding. Provide a formatted chat prompt
when chat-template behavior is needed. Raw text is not automatically wrapped
in a chat template.

## Explicit operators

| Operator | Behavior |
|---|---|
| `rmsNormalization` | FP32 squares, wide reduction matching llama.cpp CPU, FP32 scaling |
| `l2Normalization` | GDN query/key normalization with epsilon inside the squared norm |
| `causalDepthwiseConv` | Four taps, oldest first, with three samples of retained history |
| `gatedDeltaNet` | Decay, delta correction, rank-one state update, query scaling |
| `linearAttention` | QKV/Z/A/B projections, convolution, GDN, gated RMSNorm, output projection |
| `rotaryEmbedding` | Text MRoPE positions; rotates 64 of 256 head channels, theta 1e7 |
| `fullAttention` | Per-head interleaved Q/gate split, Q/K norm, causal GQA, sigmoid gate |
| `gatedMLP` | SiLU gate multiplied by the up projection, then down projection |
| `block` | Attention and MLP with their two residual connections |

GDN state is stored as key-dimension-by-value-dimension-by-head. For each
token, the implementation first decays the state, predicts the value from
the current key, applies the beta-weighted correction, and reads the updated
state with the normalized query. The query factor `1/sqrt(128)` is applied
once. A chunk boundary preserves both convolution history and recurrent state.

## Python preparation and numerical comparison

The existing `tools/prepare_tinyllama.py` exports intermediate tensors,
logits, and greedy tokens. `tools/prepare_qwen3_5.py` provides the same workflow
with llama.cpp as the reference:

```bash
python tools/prepare_qwen3_5.py \
  --model work/models/Qwen3.5-0.8B-FULL \
  --llama-cpp work/deps/llama.cpp \
  --llama-revision 183d2a04c2a666187598abb771918cfbb40dcebb
```

The script archives the selected committed llama.cpp sources into the ignored
`work/tmp/qwen3_5/` directory, builds a CPU library and the small reference
driver, and uses the unmodified converter with `--outtype f32 --no-nextn`.
It verifies that every exported GGUF tensor is F32. Existing source-model
files and the source llama.cpp checkout are not changed. The input model and
llama.cpp checkout directories are ignored by Git; all relative command-line
paths resolve from this repository root.

Reference settings are F32 weights, F32 KV cache, F32 recurrent state, CPU,
and flash attention disabled. Python and llama.cpp prompt token IDs must agree.
MATLAB independently loads the source safetensors and executes its own operators.

Five cases cover a single token, the existing photosynthesis-summary input,
a Chinese chat prompt, the same Chinese prompt with seven-token prefill chunks,
and a longer prefill. Every case runs all 24 layers and cached decode steps.
Comparison includes convolution history, GDN state, operator outputs, all layer
outputs, and every vocabulary logit at every evaluated position. The fixed
elementwise bound is `1e-3 + 1e-4 * abs(reference)`, with finite values and
identical top-1 IDs required. Generated IDs and decoded text are also recorded.

Outputs include `provenance.json`, `summary.json`, per-case
`matlab_comparison.json`, generated text, binary tensors, and execution logs.
Generated tensors and model weights stay in the ignored work directory.

To prepare references without starting MATLAB, add `--prepare-only`. To replay
existing references in MATLAB:

```matlab
reports = TestQwen3_5Numerics(modelFolder, 'work/tmp/qwen3_5/references');
```

The entry point also runs five operator tests for chunked convolution,
nonzero-state GDN continuation, the RoPE partial prefix, attention cache
continuation, and causal masking.

The operator definitions were checked against the local model's config and
the pinned llama.cpp implementation, with the upstream
[Qwen configuration](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)
and [llama.cpp model source](https://github.com/ggml-org/llama.cpp/blob/183d2a04c2a666187598abb771918cfbb40dcebb/src/models/qwen35.cpp)
as references. Simulink and HDL flows are unchanged.
