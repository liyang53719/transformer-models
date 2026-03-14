function scores = Mask(scores, negInf)
% Mask   Apply causal mask to attention score matrix
%
%   scores = Mask(scores)
%   scores = Mask(scores, negInf)
%
%   Inputs:
%       scores - [seqLen, cacheLen, pages] attention logits
%       negInf - large negative value for masked entries (default: -1e4)
%
%   Output:
%       scores - masked attention logits

    if nargin < 2
        negInf = -1e4;
    end

    [sL, cL, ~] = size(scores);
    if sL > 1 && sL == cL
        mask = tril(true(sL, cL));
        maskVal = zeros(sL, cL, 'like', scores);
        maskVal(~mask) = cast(negInf, 'like', scores);
        scores = scores + maskVal;
    end
end
