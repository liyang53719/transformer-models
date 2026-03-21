function summary = runRopeRebuildRegression(numTokens)
% runRopeRebuildRegression   Validate the rebuilt HDL-friendly RoPE model.

if nargin < 1
    numTokens = uint16(64);
end

headOptions = uint8([2 12]);
summary = struct('numTokens', cell(1, numel(headOptions)), ...
    'numHeads', cell(1, numel(headOptions)), ...
    'expectedBeats', cell(1, numel(headOptions)), ...
    'sourceBeats', cell(1, numel(headOptions)), ...
    'sourceMaxAbsErr', cell(1, numel(headOptions)), ...
    'modelBeats', cell(1, numel(headOptions)), ...
    'modelFirstValid', cell(1, numel(headOptions)), ...
    'modelLastValid', cell(1, numel(headOptions)), ...
    'modelMaxAbsErr', cell(1, numel(headOptions)), ...
    'success', cell(1, numel(headOptions)));

for caseIndex = 1:numel(headOptions)
    summary(caseIndex) = iRunOneCase(numTokens, headOptions(caseIndex));
    fprintf('REBUILD_REG case=%d tokens=%d heads=%d src_beats=%d src_max_err=%.9g model_beats=%d model_valid=[%d,%d] model_max_err=%.9g\n', ...
        caseIndex, double(summary(caseIndex).numTokens), double(summary(caseIndex).numHeads), ...
        summary(caseIndex).sourceBeats, summary(caseIndex).sourceMaxAbsErr, ...
        summary(caseIndex).modelBeats, summary(caseIndex).modelFirstValid, ...
        summary(caseIndex).modelLastValid, summary(caseIndex).modelMaxAbsErr);
end

if ~all([summary.success])
    error('RopeRebuildRegressionFailed: at least one rebuilt RoPE case mismatched the reference model.');
end

disp('REBUILD_REG_OK');

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

sourceResult = iRunDirectSourceCase(inStream, expected, numTokens, numHeads);
modelResult = iRunSimulinkCase(inStream, expected, numTokens, numHeads);

result = struct();
result.numTokens = uint16(numTokens);
result.numHeads = uint8(numHeads);
result.expectedBeats = totalBeats;
result.sourceBeats = sourceResult.beats;
result.sourceMaxAbsErr = sourceResult.maxAbsErr;
result.modelBeats = modelResult.beats;
result.modelFirstValid = modelResult.firstValid;
result.modelLastValid = modelResult.lastValid;
result.modelMaxAbsErr = modelResult.maxAbsErr;
result.success = sourceResult.maxAbsErr < 2.0e-5 && modelResult.maxAbsErr < 2.0e-5;
end

function result = iRunDirectSourceCase(inStream, expected, numTokens, numHeads)
lanes = size(inStream, 1);
totalBeats = size(inStream, 2);
captured = zeros(lanes, totalBeats, 'single');
captureIndex = 0;

clear('transformer_impl.layer.rope.streamingRopeSimulink');

for cycleIndex = 1:(totalBeats + 64)
    start = cycleIndex == 1;
    if cycleIndex <= totalBeats
        inValid = true;
        inBeat = transpose(inStream(:, cycleIndex));
    else
        inValid = false;
        inBeat = zeros(1, lanes, 'single');
    end

    [outBeat, outValid, ~, done] = transformer_impl.layer.rope.streamingRopeSimulink( ...
        start, uint16(numTokens), uint8(numHeads), inBeat, inValid);

    if outValid
        captureIndex = captureIndex + 1;
        captured(:, captureIndex) = transpose(outBeat);
    end

    if done
        break;
    end
end

assert(captureIndex == totalBeats, 'Expected %d source beats, observed %d.', totalBeats, captureIndex);

diffAbs = abs(double(captured(:, 1:captureIndex)) - double(expected(:, 1:captureIndex)));
result = struct('beats', captureIndex, 'maxAbsErr', max(diffAbs, [], 'all'));
end

function result = iRunSimulinkCase(inStream, expected, numTokens, numHeads)
repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fileparts(mfilename('fullpath')));
ensureRopeModel(true);

totalBeats = size(inStream, 2);
simCycles = totalBeats + 128;
timeVec = (0:simCycles)';

startSeq = false(simCycles + 1, 1);
startSeq(2) = true;
inValidSeq = false(simCycles + 1, 1);
inBeatSeq = zeros(simCycles + 1, 8, 'single');
for beatIndex = 1:totalBeats
    simIndex = beatIndex + 1;
    inValidSeq(simIndex) = true;
    inBeatSeq(simIndex, :) = transpose(inStream(:, beatIndex));
end

assignin('base', 'simStopTime', simCycles);
assignin('base', 'ropeStartSeq', timeseries(startSeq, timeVec));
assignin('base', 'ropeNumTokensSeq', timeseries(uint16([numTokens; numTokens]), [0; 1]));
assignin('base', 'ropeNumHeadsSeq', timeseries(uint8([numHeads; numHeads]), [0; 1]));
assignin('base', 'ropeInValidSeq', timeseries(inValidSeq, timeVec));
assignin('base', 'ropeInBeatSeq', timeseries(inBeatSeq, timeVec));

simOut = sim('rope', 'ReturnWorkspaceOutputs', 'on');
beatRaw = simOut.get('YBeatOut');
validRaw = logical(simOut.get('YValidOut'));
validIndex = find(validRaw(:));
assert(numel(validIndex) == totalBeats, 'Expected %d model beats, observed %d.', totalBeats, numel(validIndex));

beatMatrix = squeeze(beatRaw);
if size(beatMatrix, 2) ~= 8
    beatMatrix = transpose(beatMatrix);
end

captured = transpose(single(beatMatrix(validIndex, :)));
diffAbs = abs(double(captured) - double(expected(:, 1:numel(validIndex))));

result = struct();
result.beats = numel(validIndex);
result.firstValid = validIndex(1) - 1;
result.lastValid = validIndex(end) - 1;
result.maxAbsErr = max(diffAbs, [], 'all');

if bdIsLoaded('rope')
    close_system('rope', 0);
end

if ~isempty(repoRoot)
    cd(repoRoot);
end
end