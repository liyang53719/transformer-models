function Y = l2Normalization(X, epsilon)
%l2Normalization - Normalize Gated DeltaNet queries and keys
%   Y = l2Normalization(X,EPSILON) normalizes the first dimension using
%   the RMSNorm and scale order used by the llama.cpp Qwen3.5 graph.
%
%   See also qwen3_5.layer.rmsNormalization, qwen3_5.layer.gatedDeltaNet

d = single(size(X, 1));
Y = qwen3_5.layer.rmsNormalization(X, single(1), single(epsilon) ./ d);
Y = Y .* (single(1) ./ sqrt(d));
end
