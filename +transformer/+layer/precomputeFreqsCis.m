function freqs_cis = precomputeFreqsCis(dim, maxSeqLen, theta)
% precomputeFreqsCis   Precompute complex exponentials for RoPE
%
%   freqs_cis = precomputeFreqsCis(dim, maxSeqLen, theta)
%
%   Inputs:
%       dim         - Head dimension (must be even).
%       maxSeqLen   - Maximum sequence length.
%       theta       - Base period (default 10000 for Llama 2, 500000 for Llama 3).
%
%   Outputs:
%       freqs_cis   - [dim/2, maxSeqLen] complex array.

    if nargin < 3
        theta = 10000.0;
    end
    
    dim = single(dim);
    theta = single(theta);
    
    freqs = 1.0 ./ (theta .^ ((0:2:dim-2) / dim)); % [1, dim/2]
    
    t = single((0:maxSeqLen-1).'); % [seqLen, 1]
    
    freqs = freqs(:); % [dim/2, 1]
    
    % Outer product: args = t * freqs' ? 
    % We want [dim/2, seqLen].
    % freqs is per dimension index. t is per position.
    
    args = freqs * t'; % [dim/2, seqLen]
    
    % Euler's formula: exp(i * theta) = cos + i*sin
    freqs_cis = exp(1i * args);

end
