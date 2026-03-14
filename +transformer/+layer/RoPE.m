function [xq, xk] = RoPE(xq, xk, freqs_cis)
% RoPE   Apply RoPE rotation to query/key using explicit cos/sin formulas
%
%   [xq, xk] = RoPE(xq, xk, freqs_cis)
%
%   Inputs:
%       xq        - [headDim, numHeads, seqLen, batchSize]
%       xk        - [headDim, numKVHeads, seqLen, batchSize]
%       freqs_cis - [headDim/2, seqLen] complex, exp(i*theta)
%
%   Outputs:
%       xq, xk    - Same shapes as inputs after rotary position embedding

    headDim = size(xq, 1);
    if mod(headDim, 2) ~= 0
        error('RoPE:InvalidHeadDim', 'headDim must be even.');
    end

    half = headDim / 2;
    seqLen = size(xq, 3);

    cosTheta = reshape(real(freqs_cis), [half, 1, seqLen, 1]);
    sinTheta = reshape(imag(freqs_cis), [half, 1, seqLen, 1]);

    xq_r = xq(1:half, :, :, :);
    xq_i = xq(half+1:end, :, :, :);
    xk_r = xk(1:half, :, :, :);
    xk_i = xk(half+1:end, :, :, :);

    xq(1:half, :, :, :) = xq_r .* cosTheta - xq_i .* sinTheta;
    xq(half+1:end, :, :, :) = xq_r .* sinTheta + xq_i .* cosTheta;

    xk(1:half, :, :, :) = xk_r .* cosTheta - xk_i .* sinTheta;
    xk(half+1:end, :, :, :) = xk_r .* sinTheta + xk_i .* cosTheta;
end
