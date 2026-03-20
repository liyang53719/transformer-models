function cfg = qwen2_1p5b_config(numStreamHeads)
% qwen2_1p5b_config   Fixed RoPE accelerator configuration for Qwen2.5-1.5B.

if nargin < 1
    numStreamHeads = uint8(2);
end

cfg = struct();
cfg.Lanes = uint8(8);
cfg.HeadDim = uint16(128);
cfg.HalfDim = uint16(64);
cfg.BeatsPerHalf = uint8(8);
cfg.BeatsPerHead = uint8(16);
cfg.MaxTokens = uint16(1024);
cfg.MinTokens = uint16(64);
cfg.NumHeads = uint8(12);
cfg.NumKVHeads = uint8(2);
cfg.HiddenSize = uint16(1536);
cfg.RopeTheta = single(1000000.0);
cfg.NumStreamHeads = uint8(numStreamHeads);
cfg.BeatsPerToken = uint16(cfg.BeatsPerHead) * uint16(cfg.NumStreamHeads);
end