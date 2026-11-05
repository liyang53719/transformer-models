function [Y, cache, trace] = block(X, cache, w, hp, layerIndex, position)
%BLOCK - Evaluate one Qwen3.5 decoder layer
%   Y = BLOCK(X,CACHE,W,HP,layerIndex,POSITION) evaluates the one-based
%   layerIndex using its configured attention type and dense MLP.
%
%   [Y,CACHE] = BLOCK(...) also returns the updated layer cache.
%
%   [Y,CACHE,TRACE] = BLOCK(...) also returns named operator outputs.
%
%   See also qwen3_5.model, qwen3_5.layer.gatedMLP

normalized = qwen3_5.layer.rmsNormalization( ...
    X, w.input_layernorm, hp.RmsNormEpsilon);
if hp.LayerTypes(layerIndex) == "linear_attention"
    [attention, cache, trace] = qwen3_5.layer.linearAttention(normalized, cache, w, hp);
else
    [attention, cache, trace] = qwen3_5.layer.fullAttention( ...
        normalized, cache, w, hp, position);
end
residual = X + attention;
ffnInput = qwen3_5.layer.rmsNormalization( ...
    residual, w.post_attention_layernorm, hp.RmsNormEpsilon);
ffn = qwen3_5.layer.gatedMLP(ffnInput, w);
Y = residual + ffn;
trace.attn_norm = normalized;
trace.attn_residual = residual;
trace.attn_post_norm = ffnInput;
trace.ffn_out = ffn;
trace.l_out = Y;
end
