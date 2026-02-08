function Z = silu(X)
% silu   Sigmoid Linear Unit (SiLU) / Swish activation
%
%   Z = silu(X) computes the SiLU activation: x * sigmoid(x).
%
%   Inputs:
%       X - Input array
%
%   Outputs:
%       Z - Output array same size as X

Z = X .* (1 ./ (1 + exp(-X)));

end
