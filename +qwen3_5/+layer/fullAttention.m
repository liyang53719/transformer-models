function [Y, cache, trace] = fullAttention(X, cache, w, hp, position)
%fullAttention - Apply gated causal GQA with partial rotary embedding
%   Y = fullAttention(X,CACHE,W,HP,POSITION) applies Q/K RMSNorm,
%   partial RoPE, grouped query attention, and sigmoid output gating.
%   POSITION is the zero-based position of the first input token.
%
%   [Y,CACHE] = fullAttention(...) also returns FP32 keys and values.
%
%   [Y,CACHE,TRACE] = fullAttention(...) also returns operator outputs.
%
%   See also qwen3_5.layer.rotaryEmbedding, qwen3_5.layer.linearAttention

tokens = size(X, 2);
d = hp.HeadDim;
h = hp.NumHeads;
kh = hp.NumKVHeads;
qg = reshape(w.self_attn_q_proj * X, 2*d, h, tokens);
q = qg(1:d, :, :);
gate = reshape(qg(d+1:end, :, :), d*h, tokens);
k = reshape(w.self_attn_k_proj * X, d, kh, tokens);
v = reshape(w.self_attn_v_proj * X, d, kh, tokens);
q = qwen3_5.layer.rmsNormalization(q, w.self_attn_q_norm, hp.RmsNormEpsilon);
k = qwen3_5.layer.rmsNormalization(k, w.self_attn_k_norm, hp.RmsNormEpsilon);
positions = position + (0:tokens-1);
q = qwen3_5.layer.rotaryEmbedding(q, positions, hp.RotaryDim, hp.RopeTheta);
k = qwen3_5.layer.rotaryEmbedding(k, positions, hp.RotaryDim, hp.RopeTheta);
if isempty(cache)
    cache = struct('Keys', k, 'Values', v);
else
    cache.Keys = cat(3, cache.Keys, k);
    cache.Values = cat(3, cache.Values, v);
end
assert(size(cache.Keys, 3) == position + tokens, ...
    'qwen3_5:CachePosition', 'KV cache length does not match position.');
attended = zeros(d, h, tokens, 'single');
scale = single(1) ./ sqrt(single(d));
for head = 1:h
    kvHead = floor((head-1)/(h/kh)) + 1;
    keys = reshape(cache.Keys(:, kvHead, :), d, []);
    values = reshape(cache.Values(:, kvHead, :), d, []);
    queries = reshape(q(:, head, :), d, tokens);
    scores = (keys.' * queries) .* scale;
    mask = (0:size(keys, 2)-1).' > positions;
    scores(mask) = -inf;
    probability = exp(scores - max(scores, [], 1));
    probability = probability ./ single(sum(double(probability), 1));
    attended(:, head, :) = reshape(values * probability, d, 1, tokens);
end
gate = single(1) ./ (single(1) + exp(-gate));
gated = reshape(attended, d*h, tokens) .* gate;
Y = w.self_attn_o_proj * gated;
if nargout > 2
    trace = struct('Qcur', q, 'Kcur', k, 'Vcur', v, ...
        'attn_pregate', reshape(attended, d*h, tokens), ...
        'gate_sigmoid', gate, 'attn_gated', gated, 'attn_output', Y);
end
end
