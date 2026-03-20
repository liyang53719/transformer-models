function stimulus = createRmsNormalizationStimulusPipelined(X, g, epsilon)
% createRmsNormalizationStimulusPipelined   Exercise the overlapped RMSNorm prototype.

if nargin < 3
    epsilon = single(1e-6);
end

numTokens = 64;
hiddenSize = 1536;
lanesPerBeat = 8;
beatsPerToken = hiddenSize / lanesPerBeat;
maxCycles = 1 + beatsPerToken + 1 + numTokens * (beatsPerToken + 4);

resetTrace = false(maxCycles, 1);
startTrace = false(maxCycles, 1);
cfgValidTrace = false(maxCycles, 1);
ddrValidTrace = false(maxCycles, 1);
cfgBeatTrace = zeros(maxCycles, lanesPerBeat, 'single');
ddrBeatTrace = zeros(maxCycles, lanesPerBeat, 'single');

cycleIndex = 1;
resetTrace(cycleIndex) = true;

for beatIndex = 1:beatsPerToken
    cycleIndex = cycleIndex + 1;
    cfgValidTrace(cycleIndex) = true;
    cfgBeatTrace(cycleIndex, :) = g((beatIndex - 1) * lanesPerBeat + (1:lanesPerBeat));
end

cycleIndex = cycleIndex + 1;
startTrace(cycleIndex) = true;

pendingValid = false;
pendingBeat = zeros(1, lanesPerBeat, 'single');
collectedBeats = zeros(numTokens * beatsPerToken, lanesPerBeat, 'single');
collectedCount = 0;

for simCycle = 1:maxCycles
    if pendingValid
        ddrValidTrace(simCycle) = true;
        ddrBeatTrace(simCycle, :) = pendingBeat;
    end

    [outBeat, outValid, ddrReadAddr, ddrReadEn, done] = transformer_impl.layer.rmsnorm.rmsNormalizationPipelined( ...
        resetTrace(simCycle), startTrace(simCycle), cfgBeatTrace(simCycle, :), cfgValidTrace(simCycle), ...
        ddrBeatTrace(simCycle, :), ddrValidTrace(simCycle), epsilon);

    pendingValid = false;

    if ddrReadEn
        beatAddress = double(ddrReadAddr);
        tokenIndex = floor(beatAddress / beatsPerToken) + 1;
        tokenBeatIndex = mod(beatAddress, beatsPerToken) + 1;
        featureRange = (tokenBeatIndex - 1) * lanesPerBeat + (1:lanesPerBeat);
        pendingBeat = X(tokenIndex, featureRange);
        pendingValid = true;
    end

    if outValid
        collectedCount = collectedCount + 1;
        collectedBeats(collectedCount, :) = outBeat;
    end

    if done
        cycleIndex = simCycle;
        break;
    end
end

Y = reshape(collectedBeats(1:collectedCount, :).', hiddenSize, numTokens).';
stimulus = struct();
stimulus.output = Y;
stimulus.stopTime = cycleIndex;
stimulus.cycles = cycleIndex;

end