function Z = gatedMLP(X, weights)
% gatedMLP   Gated MLP (SwiGLU variant)
%
%   Z = gatedMLP(X, weights) implements the Llama/Qwen style MLP.
%
%   Equation: down( silu(gate(x)) .* up(x) )
%
%   Inputs:
%       X       - Input [hiddenDim, seqLen, batch]
%       weights - Struct with fields: 
%                   .gate_proj (hidden -> intermediate)
%                   .up_proj   (hidden -> intermediate)
%                   .down_proj (intermediate -> hidden)
%
%   Outputs:
%       Z       - Output [hiddenDim, seqLen, batch]

    import transformer.layer.silu
    
    % 1. Gate projection
    gate = weights.gate_proj * X;
    
    % 2. Up projection
    up = weights.up_proj * X;
    
    % 3. Activation (SiLU) on Gate and element-wise multiply
    intermediate = silu(gate) .* up;
    
    % 4. Down projection
    Z = weights.down_proj * intermediate;

end
