function [Y, state] = gatedDeltaNet(Q, K, V, logDecay, beta, state)
%gatedDeltaNet - Evaluate the recurrent gated delta rule in FP32
%   Y = gatedDeltaNet(Q,K,V,logDecay,BETA,STATE) uses normalized queries
%   and keys of size key-dim-by-head-by-time. V has value-dim-by-head-by-time
%   dimensions.
%   STATE is key-dim-by-value-dim-by-head, or empty for an initial state.
%   logDecay and BETA are head-by-time. Query scaling is applied once.
%
%   [Y,STATE] = gatedDeltaNet(...) also returns the final recurrent state.
%
%   See also qwen3_5.layer.linearAttention, qwen3_5.layer.l2Normalization

[keyDim, heads, tokens] = size(Q);
valueDim = size(V, 1);
if isempty(state)
    state = zeros(keyDim, valueDim, heads, 'single');
end
Y = zeros(valueDim, heads, tokens, 'single');
queryScale = single(1) ./ sqrt(single(keyDim));
for t = 1:tokens
    q = reshape(Q(:, :, t), keyDim, 1, heads);
    k = reshape(K(:, :, t), keyDim, 1, heads);
    v = reshape(V(:, :, t), valueDim, 1, heads);
    state = state .* reshape(exp(logDecay(:, t)), 1, 1, heads);
    prediction = pagemtimes(state, 'transpose', k, 'none');
    delta = (v - prediction) .* reshape(beta(:, t), 1, 1, heads);
    state = state + pagemtimes(k, 'none', delta, 'transpose');
    y = pagemtimes(state, 'transpose', q, 'none') .* queryScale;
    Y(:, :, t) = reshape(y, valueDim, heads);
end
end
