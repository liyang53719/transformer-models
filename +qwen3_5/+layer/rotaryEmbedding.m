function Y = rotaryEmbedding(X, positions, rotaryDim, theta)
%rotaryEmbedding - Rotate the text-position prefix of Qwen3.5 heads
%   Y = rotaryEmbedding(X,POSITIONS,rotaryDim,THETA) rotates the first
%   rotaryDim channels of head-dim-by-head-by-time X. Positions are zero
%   based. Text uses equal temporal, height, and width MRoPE positions.
%
%   See also qwen3_5.layer.fullAttention, qwen3_5.layer.rmsNormalization

half = rotaryDim / 2;
step = single(theta) .^ (single(-2) ./ single(rotaryDim));
frequency = ones(half, 1, 'single');
for i = 2:half
    frequency(i) = frequency(i-1) .* step;
end
angles = reshape(frequency * single(positions(:).'), half, 1, []);
c = cos(angles);
s = sin(angles);
realPart = X(1:half, :, :);
imagPart = X(half+1:rotaryDim, :, :);
Y = X;
Y(1:half, :, :) = realPart .* c - imagPart .* s;
Y(half+1:rotaryDim, :, :) = realPart .* s + imagPart .* c;
end
