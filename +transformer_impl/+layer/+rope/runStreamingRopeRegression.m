function summary = runStreamingRopeRegression(numTokens)
% runStreamingRopeRegression   Compare the streaming RoPE core against the reference model.

if nargin < 1
    numTokens = uint16(64);
end

headOptions = uint8([2 12]);
summary = struct('numTokens', cell(1, numel(headOptions)), ...
    'numHeads', cell(1, numel(headOptions)), ...
    'expectedBeats', cell(1, numel(headOptions)), ...
    'observedBeats', cell(1, numel(headOptions)), ...
    'busyCycles', cell(1, numel(headOptions)), ...
    'outValidCycles', cell(1, numel(headOptions)), ...
    'busyPct', cell(1, numel(headOptions)), ...
    'outValidPct', cell(1, numel(headOptions)), ...
    'maxAbsErr', cell(1, numel(headOptions)), ...
    'success', cell(1, numel(headOptions)));

for caseIndex = 1:numel(headOptions)
    numHeads = headOptions(caseIndex);
    summary(caseIndex) = iRunOneCase(numTokens, numHeads);
    fprintf('ROPE_REG case=%d tokens=%d heads=%d beats=%d busy=%d busy_pct=%.3f out_valid=%d out_valid_pct=%.3f max_abs_err=%.9g\n', ...
        caseIndex, double(summary(caseIndex).numTokens), double(summary(caseIndex).numHeads), ...
        summary(caseIndex).observedBeats, summary(caseIndex).busyCycles, summary(caseIndex).busyPct, ...
        summary(caseIndex).outValidCycles, summary(caseIndex).outValidPct, summary(caseIndex).maxAbsErr);
end

if ~all([summary.success])
    error('StreamingRopeRegressionFailed: at least one regression case mismatched the reference model.');
end

disp('ROPE_REG_OK');

end

function result = iRunOneCase(numTokens, numHeads)
cfg = transformer_impl.layer.rope.qwen2_1p5b_config(numHeads);
lanes = double(cfg.Lanes);
beatsPerToken = double(cfg.BeatsPerToken);
totalBeats = double(numTokens) * beatsPerToken;

inStream = zeros(lanes, totalBeats, 'single');
for tokenIndex = 0:double(numTokens)-1
    tokenBase = tokenIndex * beatsPerToken;
    for headIndex = 0:double(numHeads)-1
        headBase = tokenBase + headIndex * double(cfg.BeatsPerHead);
        for beatIndex = 0:double(cfg.BeatsPerHead)-1
            for laneIndex = 0:lanes-1
                inStream(laneIndex + 1, headBase + beatIndex + 1) = single( ...
                    -0.75 + tokenIndex * 0.03125 + headIndex * 0.0078125 + beatIndex * 0.001953125 + laneIndex * 0.000244140625);
            end
        end
    end
end

expected = transformer_impl.layer.rope.referencePrefill(inStream, numTokens, numHeads);

clear streamingRope_hdl_entry;
clear transformer_impl.layer.rope.streamingRope;

captured = zeros(lanes, totalBeats, 'single');
captureIndex = 0;
busyCycles = 0;
outValidCycles = 0;
simCycles = 0;
maxCycles = totalBeats + 64;

for cycleIndex = 1:maxCycles
    start = cycleIndex == 1;
    if cycleIndex <= totalBeats
        inValid = true;
        inBeat = transpose(inStream(:, cycleIndex));
    else
        inValid = false;
        inBeat = zeros(1, lanes, 'single');
    end

    [outBeat, outValid, busy, done] = streamingRope_hdl_entry(start, uint16(numTokens), uint8(numHeads), inBeat, inValid);

    simCycles = simCycles + 1;
    if busy
        busyCycles = busyCycles + 1;
    end
    if outValid
        outValidCycles = outValidCycles + 1;
        captureIndex = captureIndex + 1;
        captured(:, captureIndex) = transpose(outBeat);
    end

    if done
        break;
    end
end

assert(captureIndex == totalBeats, 'Expected %d output beats, observed %d.', totalBeats, captureIndex);

diffAbs = abs(double(captured(:, 1:captureIndex)) - double(expected(:, 1:captureIndex)));
maxAbsErr = max(diffAbs, [], 'all');

result = struct();
result.numTokens = uint16(numTokens);
result.numHeads = uint8(numHeads);
result.expectedBeats = totalBeats;
result.observedBeats = captureIndex;
result.busyCycles = busyCycles;
result.outValidCycles = outValidCycles;
result.busyPct = (busyCycles * 100.0) / simCycles;
result.outValidPct = (outValidCycles * 100.0) / simCycles;
result.maxAbsErr = maxAbsErr;
result.success = maxAbsErr < 2.0e-5;
end