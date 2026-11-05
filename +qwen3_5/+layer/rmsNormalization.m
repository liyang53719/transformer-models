function Y = rmsNormalization(X, weight, epsilon)
%rmsNormalization - Normalize along the first dimension
%   Y = rmsNormalization(X,WEIGHT,EPSILON) applies multiplicative RMSNorm.
%   WEIGHT already includes any checkpoint offset. Squares are FP32 and
%   their reduction uses a wide accumulator, as in llama.cpp on CPU.
%
%   See also qwen3_5.layer.l2Normalization, qwen3_5.layer.gatedDeltaNet

meanSquare = single(mean(double(X .* X), 1));
scale = single(1) ./ sqrt(meanSquare + single(epsilon));
Y = (X .* scale) .* weight;
end
