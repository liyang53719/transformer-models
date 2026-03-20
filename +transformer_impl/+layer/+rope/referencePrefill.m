function outStream = referencePrefill(inStream, numTokens, numHeadsThisStream)
% referencePrefill   Golden prefill-only RoPE transform for streamed beats.

cfg = transformer_impl.layer.rope.qwen2_1p5b_config(numHeadsThisStream);
lanes = double(cfg.Lanes);
headDim = double(cfg.HeadDim);
heads = double(cfg.NumStreamHeads);
beatsPerToken = double(cfg.BeatsPerToken);
half = headDim / 2;

assert(size(inStream, 1) == lanes, 'Expected inStream to be lanes-by-totalBeats.');
assert(size(inStream, 2) == beatsPerToken * double(numTokens), 'Unexpected stream length for token/head configuration.');

tokens = zeros(headDim, heads, double(numTokens), 'single');

for tokenIndex = 0:double(numTokens)-1
    tokenBase = tokenIndex * beatsPerToken;
    for headIndex = 0:heads-1
        headBase = tokenBase + headIndex * double(cfg.BeatsPerHead);
        for beatIndex = 0:double(cfg.BeatsPerHead)-1
            dimBase = beatIndex * lanes;
            tokens(dimBase + (1:lanes), headIndex + 1, tokenIndex + 1) = inStream(:, headBase + beatIndex + 1);
        end
    end
end

freqs = transformer.layer.precomputeFreqsCis(headDim, double(numTokens), cfg.RopeTheta);

for tokenIndex = 1:double(numTokens)
    cosTheta = real(freqs(:, tokenIndex));
    sinTheta = imag(freqs(:, tokenIndex));
    for headIndex = 1:heads
        firstHalf = tokens(1:half, headIndex, tokenIndex);
        secondHalf = tokens(half+1:end, headIndex, tokenIndex);
        tokens(1:half, headIndex, tokenIndex) = firstHalf .* cosTheta - secondHalf .* sinTheta;
        tokens(half+1:end, headIndex, tokenIndex) = firstHalf .* sinTheta + secondHalf .* cosTheta;
    end
end

outStream = zeros(size(inStream), 'single');
for tokenIndex = 0:double(numTokens)-1
    tokenBase = tokenIndex * beatsPerToken;
    for headIndex = 0:heads-1
        headBase = tokenBase + headIndex * double(cfg.BeatsPerHead);
        for beatIndex = 0:double(cfg.BeatsPerHead)-1
            dimBase = beatIndex * lanes;
            outStream(:, headBase + beatIndex + 1) = tokens(dimBase + (1:lanes), headIndex + 1, tokenIndex + 1);
        end
    end
end
end