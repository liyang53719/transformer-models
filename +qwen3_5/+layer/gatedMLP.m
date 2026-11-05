function Y = gatedMLP(X, w)
%gatedMLP - Apply the Qwen3.5 dense SiLU-gated feed-forward network
%   Y = gatedMLP(X,W) computes down(SiLU(gate(X)) times up(X)).
%
%   See also qwen3_5.layer.silu, qwen3_5.layer.block

Y = w.mlp_down_proj * (qwen3_5.layer.silu(w.mlp_gate_proj * X) .* ...
    (w.mlp_up_proj * X));
end
