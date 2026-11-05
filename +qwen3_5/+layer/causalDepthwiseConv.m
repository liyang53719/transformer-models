function [Y, history] = causalDepthwiseConv(X, weight, history)
%causalDepthwiseConv - Convolve each channel with causal history
%   Y = causalDepthwiseConv(X,WEIGHT,HISTORY) filters channel-by-time X.
%   WEIGHT is channel-by-kernel with the oldest tap first. HISTORY holds
%   kernel-minus-one past samples, oldest first, or is empty on reset.
%
%   [Y,HISTORY] = causalDepthwiseConv(...) also returns updated history.
%
%   See also qwen3_5.layer.linearAttention, qwen3_5.layer.gatedDeltaNet

kernel = size(weight, 2);
if isempty(history)
    history = zeros(size(X, 1), kernel - 1, 'single');
end
combined = [history X];
Y = zeros(size(X), 'single');
for tap = 1:kernel
    Y = Y + combined(:, tap:tap+size(X, 2)-1) .* weight(:, tap);
end
history = combined(:, end-kernel+2:end);
end
