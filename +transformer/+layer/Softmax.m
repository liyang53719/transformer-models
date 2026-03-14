function attn_weights = Softmax(scores, dim)
% Softmax   Numerically stable softmax for attention logits
%
%   attn_weights = Softmax(scores)
%   attn_weights = Softmax(scores, dim)
%
%   Inputs:
%       scores - attention logits tensor
%       dim    - softmax dimension (default: 2)
%
%   Output:
%       attn_weights - softmax-normalized attention weights

    if nargin < 2
        dim = 2;
    end

    max_scores = max(scores, [], dim);
    exps = exp(scores - max_scores);
    attn_weights = exps ./ sum(exps, dim);
end
