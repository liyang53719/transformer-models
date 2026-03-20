function result = runRmsNormalizationHandshakeRegression(numTokens, simCycles)
% runRmsNormalizationHandshakeRegression
% Run the SRAM RMSNorm source model with a closed-loop DDR responder.

if nargin < 1
    numTokens = 64;
end

if nargin < 2
    simCycles = 30000;
end

beatsPerToken = 192;
lanes = 8;
expectedBeats = numTokens * beatsPerToken;
harnessName = 'rmsNormalizationClosedLoopHarness';

addpath(fileparts(mfilename('fullpath')));
ensureRmsNormalizationModel();

harnessPath = iBuildHarnessModel(harnessName, simCycles, numTokens);
load_system(harnessPath);

t = (0:simCycles)';
start = false(simCycles + 1, 1);
cfgValid = false(simCycles + 1, 1);
cfgBeat = zeros(simCycles + 1, lanes, 'single');

for beatIndex = 1:beatsPerToken
    cfgValid(beatIndex + 1) = true;
    cfgBeat(beatIndex + 1, :) = iGammaBeat(beatIndex - 1, lanes);
end

start(beatsPerToken + 5) = true;

assignin('base', 'simStopTime', simCycles);
assignin('base', 'startSeq', timeseries(start, t));
assignin('base', 'cfgGammaValidSeq', timeseries(cfgValid, t));
assignin('base', 'cfgGammaBeatSeq', timeseries(cfgBeat, t));
runOut = sim(harnessName, 'ReturnWorkspaceOutputs', 'on');
readEnRaw = runOut.get('ReadEnOut');
outValidRaw = runOut.get('YValidOut');
doneOutRaw = runOut.get('DoneOut');

plannedReads = nnz(logical(readEnRaw(:)));
outValid = logical(outValidRaw(:));
doneOut = logical(doneOutRaw(:));

result = struct();
result.expectedBeats = expectedBeats;
result.plannedReads = plannedReads;
result.observedBeats = nnz(outValid);
result.doneCount = nnz(doneOut);
result.success = result.observedBeats == expectedBeats && result.doneCount == 1;

fprintf('planned_reads=%d\n', result.plannedReads);
fprintf('observed_beats=%d\n', result.observedBeats);
fprintf('done_count=%d\n', result.doneCount);

if ~result.success
    error('RMSNormHandshakeRegressionFailed: expected %d beats, observed %d, done_count=%d', ...
        expectedBeats, result.observedBeats, result.doneCount);
end

disp('HS64_OK');

end

function harnessPath = iBuildHarnessModel(harnessName, simCycles, numTokens)
repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
harnessDir = fullfile(repoRoot, 'work', 'simulink');
harnessPath = fullfile(harnessDir, [harnessName '.slx']);

if ~isfolder(harnessDir)
    mkdir(harnessDir);
end

if bdIsLoaded(harnessName)
    close_system(harnessName, 0);
end

if isfile(harnessPath)
    delete(harnessPath);
end

new_system(harnessName);
set_param(harnessName, 'Solver', 'FixedStepDiscrete');
set_param(harnessName, 'FixedStep', '1');
set_param(harnessName, 'StopTime', 'simStopTime');

add_block('simulink/Sources/From Workspace', [harnessName '/StartSrc'], ...
    'VariableName', 'startSeq', 'Position', [30 60 120 90]);
add_block('simulink/Sources/From Workspace', [harnessName '/CfgBeatSrc'], ...
    'VariableName', 'cfgGammaBeatSeq', 'Position', [30 110 120 140]);
add_block('simulink/Sources/From Workspace', [harnessName '/CfgValidSrc'], ...
    'VariableName', 'cfgGammaValidSeq', 'Position', [30 160 120 190]);
add_block('rmsNormalization/DUT', [harnessName '/DUT'], 'Position', [250 95 680 360]);
add_block('simulink/Discrete/Unit Delay', [harnessName '/ReadAddrDelay'], ...
    'InitialCondition', 'uint16(0)', 'Position', [720 185 765 209]);
add_block('simulink/Discrete/Unit Delay', [harnessName '/ReadEnDelay'], ...
    'InitialCondition', 'false', 'Position', [720 235 765 259]);
add_block('simulink/User-Defined Functions/MATLAB Function', [harnessName '/DdrResponder'], ...
    'Position', [800 175 1020 305]);
set_param([harnessName '/DdrResponder'], 'SystemSampleTime', '1');
open_system([harnessName '/DdrResponder']);
rt = sfroot;
chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [harnessName '/DdrResponder']);
chart.Script = iBuildDdrResponderScript(numTokens);

add_block('simulink/Sinks/To Workspace', [harnessName '/YValid'], ...
    'VariableName', 'YValidOut', 'SaveFormat', 'Array', 'Position', [1080 125 1170 155]);
add_block('simulink/Sinks/To Workspace', [harnessName '/ReadAddr'], ...
    'VariableName', 'ReadAddrOut', 'SaveFormat', 'Array', 'Position', [1080 175 1170 205]);
add_block('simulink/Sinks/To Workspace', [harnessName '/ReadEn'], ...
    'VariableName', 'ReadEnOut', 'SaveFormat', 'Array', 'Position', [1080 225 1170 255]);
add_block('simulink/Sinks/To Workspace', [harnessName '/Done'], ...
    'VariableName', 'DoneOut', 'SaveFormat', 'Array', 'Position', [1080 275 1170 305]);

add_line(harnessName, 'StartSrc/1', 'DUT/1');
add_line(harnessName, 'CfgBeatSrc/1', 'DUT/2');
add_line(harnessName, 'CfgValidSrc/1', 'DUT/3');
add_line(harnessName, 'DdrResponder/1', 'DUT/4');
add_line(harnessName, 'DdrResponder/2', 'DUT/5');
add_line(harnessName, 'DUT/3', 'ReadAddrDelay/1');
add_line(harnessName, 'DUT/4', 'ReadEnDelay/1');
add_line(harnessName, 'ReadAddrDelay/1', 'DdrResponder/1');
add_line(harnessName, 'ReadEnDelay/1', 'DdrResponder/2');
add_line(harnessName, 'DUT/2', 'YValid/1');
add_line(harnessName, 'DUT/3', 'ReadAddr/1');
add_line(harnessName, 'DUT/4', 'ReadEn/1');
add_line(harnessName, 'DUT/5', 'Done/1');

assignin('base', 'simStopTime', simCycles);
save_system(harnessName, harnessPath);
close_system(harnessName, 0);
end

function scriptText = iBuildDdrResponderScript(numTokens)
scriptText = strjoin({
    'function [ddrBeat, ddrValid] = DdrResponder(readAddr, readEn)'
    '%#codegen'
    'beatsPerToken = uint16(192);'
    sprintf('numTokens = uint16(%d);', numTokens)
    'ddrBeat = zeros(1, 8, ''single'');'
    'ddrValid = false;'
    'if readEn ~= 0'
    '    tokenIndex = uint16(floor(double(readAddr) / double(beatsPerToken)));'
    '    beatIndex = uint16(mod(double(readAddr), double(beatsPerToken)));'
    '    if tokenIndex < numTokens'
    '        for laneIndex = uint16(0):uint16(7)'
    '            ddrBeat(laneIndex + uint16(1)) = single(-0.75 + double(tokenIndex) * 0.01 + double(beatIndex) * 0.001 + double(laneIndex) * 0.0001);'
    '        end'
    '        ddrValid = true;'
    '    end'
    'end'
    'end'}, newline);
end

function beat = iGammaBeat(beatIndex, lanes)
beat = zeros(1, lanes, 'single');
for laneIndex = 0:(lanes - 1)
    beat(laneIndex + 1) = single(0.25 + beatIndex * 0.001 + laneIndex * 0.0001);
end
end