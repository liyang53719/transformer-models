function Y = silu(X)
%SILU - Apply the SiLU activation in FP32
%   Y = SILU(X) returns X times its sigmoid.
%
%   See also qwen3_5.layer.gatedMLP, qwen3_5.layer.linearAttention

Y = X ./ (single(1) + exp(-X));
end
