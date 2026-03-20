function stimulus = createRmsNormalizationStimulus(X, g, epsilon)
% createRmsNormalizationStimulus   Create stream stimulus and capture DUT output.

if nargin < 3
    epsilon = single(1e-6);
end

numTokens = 64;
hiddenSize = 1536;
lanesPerBeat = 8;
beatsPerToken = hiddenSize / lanesPerBeat;
maxCycles = 1 + beatsPerToken + 1 + numTokens * (2 * beatsPerToken + 2);

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

    [outBeat, outValid, ddrReadAddr, ddrReadEn, done] = transformer_impl.layer.rmsnorm.rmsNormalization( ...
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
flushSlackCycles = numTokens * 22 + 8;
tailCycles = flushSlackCycles + 4;
extendedLength = cycleIndex + tailCycles;
time = (0:extendedLength-1).';

resetSeq = [resetTrace(1:cycleIndex); false(tailCycles, 1)];
startSeq = [startTrace(1:cycleIndex); false(tailCycles, 1)];
cfgGammaValidSeq = [cfgValidTrace(1:cycleIndex); false(tailCycles, 1)];
cfgGammaBeatSeq = [cfgBeatTrace(1:cycleIndex, :); zeros(tailCycles, lanesPerBeat, 'single')];
ddrDataValidSeq = [ddrValidTrace(1:cycleIndex); false(tailCycles, 1)];
ddrDataBeatSeq = [ddrBeatTrace(1:cycleIndex, :); zeros(tailCycles, lanesPerBeat, 'single')];

stimulus = struct();
stimulus.resetSeq = timeseries(resetSeq, time);
stimulus.startSeq = timeseries(startSeq, time);
stimulus.cfgGammaValidSeq = timeseries(cfgGammaValidSeq, time);
stimulus.cfgGammaBeatSeq = timeseries(cfgGammaBeatSeq, time);
stimulus.ddrDataValidSeq = timeseries(ddrDataValidSeq, time);
stimulus.ddrDataBeatSeq = timeseries(ddrDataBeatSeq, time);
stimulus.stopTime = extendedLength;
stimulus.output = Y;

end
