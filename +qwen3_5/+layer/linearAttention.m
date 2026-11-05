function [Y, cache, trace] = linearAttention(X, cache, w, hp)
%linearAttention - Apply Qwen3.5 Gated DeltaNet attention
%   Y = linearAttention(X,CACHE,W,HP) evaluates the causal convolution,
%   normalized delta rule, gated RMSNorm, and output projection.
%
%   [Y,CACHE] = linearAttention(...) also returns convolution history and
%   the SSM state.
%
%   [Y,CACHE,TRACE] = linearAttention(...) also returns operator outputs.
%
%   See also qwen3_5.layer.gatedDeltaNet, qwen3_5.layer.fullAttention

if isempty(cache)
    cache = struct('History', [], 'SSM', []);
end
tokens = size(X, 2);
kd = hp.LinearKeyDim;
vd = hp.LinearValueDim;
kh = hp.LinearKeyHeads;
vh = hp.LinearValueHeads;
qkv = w.linear_attn_in_proj_qkv * X;
z = reshape(w.linear_attn_in_proj_z * X, vd, vh, tokens);
beta = single(1) ./ (single(1) + exp(-(w.linear_attn_in_proj_b * X)));
a = w.linear_attn_in_proj_a * X + w.linear_attn_dt_bias;
softplus = max(a, single(0)) + log1p(exp(-abs(a)));
logDecay = softplus .* w.linear_attn_A;
[convolved, cache.History] = qwen3_5.layer.causalDepthwiseConv( ...
    qkv, w.linear_attn_conv1d, cache.History);
convolved = qwen3_5.layer.silu(convolved);
q = reshape(convolved(1:kd*kh, :), kd, kh, tokens);
k = reshape(convolved(kd*kh+1:2*kd*kh, :), kd, kh, tokens);
v = reshape(convolved(2*kd*kh+1:end, :), vd, vh, tokens);
q = qwen3_5.layer.l2Normalization(q, hp.RmsNormEpsilon);
k = qwen3_5.layer.l2Normalization(k, hp.RmsNormEpsilon);
[core, cache.SSM] = qwen3_5.layer.gatedDeltaNet( ...
    q, k, v, logDecay, beta, cache.SSM);
normalized = qwen3_5.layer.rmsNormalization( ...
    core, w.linear_attn_norm, hp.RmsNormEpsilon);
gated = normalized .* qwen3_5.layer.silu(z);
Y = w.linear_attn_out_proj * reshape(gated, vd*vh, tokens);
if nargout > 2
    trace = struct('q_conv_predelta', q, 'k_conv_predelta', k, ...
        'v_conv_predelta', v, 'gate', logDecay, 'beta_sigmoid', beta, ...
        'final_output', reshape(gated, vd*vh, tokens), ...
        'linear_attn_out', Y, 'new_state', cache.SSM, ...
        'attn_output', core, 'conv_state_last', cache.History.');
end
end
