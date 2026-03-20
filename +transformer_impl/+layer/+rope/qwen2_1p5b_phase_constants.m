function [deltaCos, deltaSin] = qwen2_1p5b_phase_constants()
% qwen2_1p5b_phase_constants   Per-dimension RoPE delta for token-to-token updates.

cfg = transformer_impl.layer.rope.qwen2_1p5b_config();
beatsPerHalf = double(cfg.BeatsPerHalf);
lanes = double(cfg.Lanes);
headDim = double(cfg.HeadDim);
theta = double(cfg.RopeTheta);

deltaCos = zeros(beatsPerHalf, lanes, 'single');
deltaSin = zeros(beatsPerHalf, lanes, 'single');

for beatIndex = 0:beatsPerHalf-1
    for laneIndex = 0:lanes-1
        dimIndex = beatIndex * lanes + laneIndex;
        invFreq = 1.0 / (theta ^ ((2.0 * dimIndex) / headDim));
        deltaCos(beatIndex + 1, laneIndex + 1) = single(cos(invFreq));
        deltaSin(beatIndex + 1, laneIndex + 1) = single(sin(invFreq));
    end
end
end