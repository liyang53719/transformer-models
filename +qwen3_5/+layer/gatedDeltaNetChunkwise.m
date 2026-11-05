function [Y, state] = gatedDeltaNetChunkwise(Q, K, V, logDecay, beta, state, chunkSize)
%gatedDeltaNetChunkwise - Evaluate gated delta attention in FP32 chunks
%   Y = gatedDeltaNetChunkwise(Q,K,V,logDecay,BETA,STATE) evaluates normalized
%   queries and keys of size key-dim-by-head-by-time, with 64-token chunks.
%   V is value-dim-by-head-by-time; logDecay and BETA are head-by-time.
%   STATE is key-dim-by-value-dim-by-head, or empty for an initial state.
%
%   Y = gatedDeltaNetChunkwise(...,chunkSize) specifies the chunk size.
%
%   [Y,STATE] = gatedDeltaNetChunkwise(...) also returns the final state.
%   Within a chunk, a unit lower-triangular solve replaces token recursion.
%   State advances between chunks. A short final chunk needs no padding.
%
%   See also qwen3_5.layer.gatedDeltaNet, qwen3_5.layer.linearAttention

if nargin < 7
    chunkSize = 64;
end
validateattributes(chunkSize, {'numeric'}, {'scalar', 'integer', 'positive'});
[keyDim, heads, tokens] = size(Q);
valueDim = size(V, 1);
if isempty(state)
    state = zeros(keyDim, valueDim, heads, 'single');
end
Y = zeros(valueDim, heads, tokens, 'single');
queryScale = single(1) ./ sqrt(single(keyDim));
for first = 1:chunkSize:tokens
    indices = first:min(first + chunkSize - 1, tokens);
    count = numel(indices);
    q = permute(Q(:, :, indices), [3 1 2]);
    k = permute(K(:, :, indices), [3 1 2]);
    v = permute(V(:, :, indices), [3 1 2]);
    b = reshape(beta(:, indices).', count, 1, heads);
    cumulative = reshape(cumsum(logDecay(:, indices), 2).', count, 1, heads);
    difference = cumulative - permute(cumulative, [2 1 3]);
    upper = repmat(triu(true(count), 1), 1, 1, heads);
    difference(upper) = -inf;
    decay = exp(difference);
    prefixDecay = exp(cumulative);

    % Delta corrections solve (I + strict-lower(beta*K*K'*decay))*delta.
    gram = pagemtimes(k, 'none', k, 'transpose');
    lower = single(tril(true(count), -1));
    triangularSystem = eye(count, 'single') + (b .* gram .* decay) .* lower;
    memory = pagemtimes(k, state) .* prefixDecay;
    delta = pagemldivide(triangularSystem, (v - memory) .* b);

    scores = pagemtimes(q, 'none', k, 'transpose') .* decay;
    output = (pagemtimes(q, state) .* prefixDecay + ...
        pagemtimes(scores, delta)) .* queryScale;
    Y(:, :, indices) = permute(output, [2 3 1]);
    tailDecay = exp(cumulative(end, :, :) - cumulative);
    state = state .* exp(cumulative(end, :, :)) + ...
        pagemtimes(k, 'transpose', delta .* tailDecay, 'none');
end
end
