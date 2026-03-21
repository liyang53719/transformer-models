function modelPath = buildRopeModel()
% buildRopeModel   Build a packed streaming RoPE Simulink model.

thisDir = fileparts(mfilename('fullpath'));
modelName = 'rope';
modelPath = fullfile(thisDir, [modelName '.slx']);
rtlVersion = 'v1';

if bdIsLoaded(modelName)
    close_system(modelName, 0);
end

if isfile(modelPath)
    delete(modelPath);
end

new_system(modelName);
set_param(modelName, 'Solver', 'FixedStepDiscrete');
set_param(modelName, 'FixedStep', '1');
set_param(modelName, 'StopTime', 'simStopTime');
set_param(modelName, 'SystemTargetFile', 'grt.tlc');
set_param(modelName, 'SingleTaskRateTransMsg', 'error');
set_param(modelName, 'MultiTaskRateTransMsg', 'error');
set_param(modelName, 'DefaultParameterBehavior', 'Inlined');
set_param(modelName, 'Description', sprintf('RoPE packed RTL %s', rtlVersion));

assignin('base', 'simStopTime', 10);
assignin('base', 'ropeStartSeq', timeseries([false; true; false], [0; 1; 2]));
assignin('base', 'ropeNumTokensSeq', timeseries(uint16([64; 64]), [0; 1]));
assignin('base', 'ropeNumHeadsSeq', timeseries(uint8([2; 2]), [0; 1]));
assignin('base', 'ropeInValidSeq', timeseries([false; false], [0; 1]));
assignin('base', 'ropeInBeatSeq', timeseries(zeros(2, 8, 'single'), [0; 1]));

    iAddFromWorkspace(modelName, 'StartSrc', 'ropeStartSeq', [30 60 120 90]);
    iAddFromWorkspace(modelName, 'NumTokensSrc', 'ropeNumTokensSeq', [30 110 120 140]);
    iAddFromWorkspace(modelName, 'NumHeadsSrc', 'ropeNumHeadsSeq', [30 160 120 190]);
    iAddFromWorkspace(modelName, 'InBeatSrc', 'ropeInBeatSeq', [30 210 120 240]);
    iAddFromWorkspace(modelName, 'InValidSrc', 'ropeInValidSeq', [30 260 120 290]);

add_block('simulink/Ports & Subsystems/Subsystem', [modelName '/DUT'], 'Position', [220 95 650 330]);
set_param([modelName '/DUT'], 'TreatAsAtomicUnit', 'on');
iBuildDutSubsystem(modelName);

add_block('simulink/Ports & Subsystems/Subsystem', [modelName '/DUTPacked'], 'Position', [220 395 650 640]);
set_param([modelName '/DUTPacked'], 'TreatAsAtomicUnit', 'on');
iBuildPackedDutSubsystem(modelName);

iAddToWorkspace(modelName, 'YBeat', 'YBeatOut', [740 125 830 155]);
iAddToWorkspace(modelName, 'YValid', 'YValidOut', [740 175 830 205]);
iAddToWorkspace(modelName, 'Busy', 'BusyOut', [740 225 830 255]);
iAddToWorkspace(modelName, 'Done', 'DoneOut', [740 275 830 305]);

add_line(modelName, 'StartSrc/1', 'DUT/1');
add_line(modelName, 'NumTokensSrc/1', 'DUT/2');
add_line(modelName, 'NumHeadsSrc/1', 'DUT/3');
add_line(modelName, 'InBeatSrc/1', 'DUT/4');
add_line(modelName, 'InValidSrc/1', 'DUT/5');
add_line(modelName, 'DUT/1', 'YBeat/1');
add_line(modelName, 'DUT/2', 'YValid/1');
add_line(modelName, 'DUT/3', 'Busy/1');
add_line(modelName, 'DUT/4', 'Done/1');

save_system(modelName, modelPath);
close_system(modelName, 0);
end

function iBuildDutSubsystem(modelName)
subsystem = [modelName '/DUT'];
iBuildStreamingDutSubsystem(subsystem);
end

function iBuildStreamingDutSubsystem(subsystem)
iDeleteSubsystemContents(subsystem);

controlLatency = '0';

add_block('simulink/Ports & Subsystems/In1', [subsystem '/start'], 'Position', [25 63 55 77], 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgNumTokens'], 'Position', [25 98 55 112], 'Port', '2', 'SampleTime', '1', 'OutDataTypeStr', 'uint16');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgNumHeads'], 'Position', [25 133 55 147], 'Port', '3', 'SampleTime', '1', 'OutDataTypeStr', 'uint8');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/inBeat'], 'Position', [25 168 55 182], 'Port', '4', 'SampleTime', '1', 'OutDataTypeStr', 'single');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/inValid'], 'Position', [25 203 55 217], 'Port', '5', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/User-Defined Functions/MATLAB Function', [subsystem '/RopeCore'], 'Position', [120 70 420 235]);
set_param([subsystem '/RopeCore'], 'SystemSampleTime', '1');
hdlset_param([subsystem '/RopeCore'], 'architecture', 'MATLAB Function');
open_system([subsystem '/RopeCore']);
rt = sfroot;
chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [subsystem '/RopeCore']);
chart.Script = iBuildRopeCoreScript();

add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/BeatSpec'], ...
    'Dimensions', '[1 8]', 'OutDataTypeStr', 'single', 'Position', [455 93 510 117]);
add_block('simulink/Discrete/Integer Delay', [subsystem '/OutValidDelay'], ...
    'Position', [455 133 510 157], 'NumDelays', controlLatency, 'vinit', '0');
add_block('simulink/Discrete/Integer Delay', [subsystem '/DoneDelay'], ...
    'Position', [455 213 510 237], 'NumDelays', controlLatency, 'vinit', '0');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outBeat'], 'Position', [545 98 575 112]);
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outValid'], 'Position', [545 138 575 152], 'Port', '2');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/busy'], 'Position', [545 178 575 192], 'Port', '3');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/done'], 'Position', [545 218 575 232], 'Port', '4');

add_line(subsystem, 'start/1', 'RopeCore/1');
add_line(subsystem, 'cfgNumTokens/1', 'RopeCore/2');
add_line(subsystem, 'cfgNumHeads/1', 'RopeCore/3');
add_line(subsystem, 'inBeat/1', 'RopeCore/4');
add_line(subsystem, 'inValid/1', 'RopeCore/5');
add_line(subsystem, 'RopeCore/1', 'BeatSpec/1');
add_line(subsystem, 'BeatSpec/1', 'outBeat/1');
add_line(subsystem, 'RopeCore/2', 'OutValidDelay/1');
add_line(subsystem, 'OutValidDelay/1', 'outValid/1');
add_line(subsystem, 'RopeCore/3', 'busy/1');
add_line(subsystem, 'RopeCore/4', 'DoneDelay/1');
add_line(subsystem, 'DoneDelay/1', 'done/1');
end

function iBuildPackedDutSubsystem(modelName)
subsystem = [modelName '/DUTPacked'];
iDeleteSubsystemContents(subsystem);

add_block('simulink/Ports & Subsystems/In1', [subsystem '/start'], 'Position', [25 63 55 77], 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgNumTokens'], 'Position', [25 98 55 112], 'Port', '2', 'SampleTime', '1', 'OutDataTypeStr', 'uint16');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgNumHeads'], 'Position', [25 133 55 147], 'Port', '3', 'SampleTime', '1', 'OutDataTypeStr', 'uint8');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/inBeat'], 'Position', [25 168 55 182], 'Port', '4', 'SampleTime', '1', 'OutDataTypeStr', 'fixdt(0,256,0)');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/inValid'], 'Position', [25 203 55 217], 'Port', '5', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');

add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/InBeatUnpack'], 'Position', [105 150 220 200]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/CoreDUT'], 'Position', [255 35 685 255]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/OutBeatPack'], 'Position', [720 80 835 130]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/OutBeatSpec'], ...
    'Dimensions', '1', 'OutDataTypeStr', 'fixdt(0,256,0)', 'Position', [855 93 920 117]);

add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outBeat'], 'Position', [950 98 980 112]);
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outValid'], 'Position', [950 148 980 162], 'Port', '2');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/busy'], 'Position', [950 198 980 212], 'Port', '3');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/done'], 'Position', [950 248 980 262], 'Port', '4');

set_param([subsystem '/InBeatUnpack'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/CoreDUT'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/OutBeatPack'], 'TreatAsAtomicUnit', 'on');

iBuildPackedBeatUnpackSubsystem([subsystem '/InBeatUnpack']);
iBuildStreamingDutSubsystem([subsystem '/CoreDUT']);
iBuildPackedBeatPackSubsystem([subsystem '/OutBeatPack']);

add_line(subsystem, 'start/1', 'CoreDUT/1');
add_line(subsystem, 'cfgNumTokens/1', 'CoreDUT/2');
add_line(subsystem, 'cfgNumHeads/1', 'CoreDUT/3');
add_line(subsystem, 'inBeat/1', 'InBeatUnpack/1');
add_line(subsystem, 'InBeatUnpack/1', 'CoreDUT/4');
add_line(subsystem, 'inValid/1', 'CoreDUT/5');
add_line(subsystem, 'CoreDUT/1', 'OutBeatPack/1');
add_line(subsystem, 'OutBeatPack/1', 'OutBeatSpec/1');
add_line(subsystem, 'OutBeatSpec/1', 'outBeat/1');
add_line(subsystem, 'CoreDUT/2', 'outValid/1');
add_line(subsystem, 'CoreDUT/3', 'busy/1');
add_line(subsystem, 'CoreDUT/4', 'done/1');
end

function iBuildPackedBeatUnpackSubsystem(subsystem)
iDeleteSubsystemContents(subsystem);

add_block('simulink/Ports & Subsystems/In1', [subsystem '/packedBeat'], ...
    'Position', [30 43 60 57], 'OutDataTypeStr', 'fixdt(0,256,0)');
add_block('simulink/User-Defined Functions/MATLAB Function', [subsystem '/Unpack'], ...
    'Position', [100 25 260 75]);
set_param([subsystem '/Unpack'], 'SystemSampleTime', '1');
hdlset_param([subsystem '/Unpack'], 'architecture', 'MATLAB Function');
open_system([subsystem '/Unpack']);
rt = sfroot;
chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [subsystem '/Unpack']);
chart.Script = iBuildPackedBeatUnpackScript();
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/WordSpec'], ...
    'Dimensions', '[1 8]', 'OutDataTypeStr', 'fixdt(0,32,0)', 'Position', [285 18 340 42]);
add_block('hdlsllib/HDL Floating Point Operations/Float Typecast', [subsystem '/WordsToSingle'], ...
    'Position', [365 25 430 55]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/BeatSpec'], ...
    'Dimensions', '[1 8]', 'OutDataTypeStr', 'single', 'Position', [455 25 510 55]);
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/beat'], 'Position', [545 33 575 47]);
add_line(subsystem, 'packedBeat/1', 'Unpack/1');
add_line(subsystem, 'Unpack/1', 'WordSpec/1');
add_line(subsystem, 'WordSpec/1', 'WordsToSingle/1');
add_line(subsystem, 'WordsToSingle/1', 'BeatSpec/1');
add_line(subsystem, 'BeatSpec/1', 'beat/1');
end

function iBuildPackedBeatPackSubsystem(subsystem)
iDeleteSubsystemContents(subsystem);

add_block('simulink/Ports & Subsystems/In1', [subsystem '/beat'], ...
    'Position', [30 43 60 57], 'OutDataTypeStr', 'single');
add_block('hdlsllib/HDL Floating Point Operations/Float Typecast', [subsystem '/SingleToWords'], ...
    'Position', [90 30 155 70]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/WordSpec'], ...
    'Dimensions', '[1 8]', 'OutDataTypeStr', 'fixdt(0,32,0)', 'Position', [180 30 235 70]);
add_block('simulink/User-Defined Functions/MATLAB Function', [subsystem '/Pack'], ...
    'Position', [270 25 430 75]);
set_param([subsystem '/Pack'], 'SystemSampleTime', '1');
hdlset_param([subsystem '/Pack'], 'architecture', 'MATLAB Function');
open_system([subsystem '/Pack']);
rt = sfroot;
chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [subsystem '/Pack']);
chart.Script = iBuildPackedBeatPackScript();
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/PackedSpec'], ...
    'Dimensions', '1', 'OutDataTypeStr', 'fixdt(0,256,0)', 'Position', [455 38 520 62]);
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/packedBeat'], 'Position', [555 43 585 57]);
add_line(subsystem, 'beat/1', 'SingleToWords/1');
add_line(subsystem, 'SingleToWords/1', 'WordSpec/1');
add_line(subsystem, 'WordSpec/1', 'Pack/1');
add_line(subsystem, 'Pack/1', 'PackedSpec/1');
add_line(subsystem, 'PackedSpec/1', 'packedBeat/1');
end

function iDeleteSubsystemContents(subsystem)
if ~bdIsLoaded(bdroot(subsystem))
    return;
end

lines = find_system(subsystem, 'SearchDepth', 1, 'FindAll', 'on', 'Type', 'line');
for lineIndex = 1:numel(lines)
    delete_line(lines(lineIndex));
end

blocks = find_system(subsystem, 'SearchDepth', 1, 'Type', 'Block');
for blockIndex = 1:numel(blocks)
    if strcmp(blocks{blockIndex}, subsystem)
        continue;
    end
    delete_block(blocks{blockIndex});
end
end

function iAddFromWorkspace(modelName, blockName, variableName, position)
blockPath = [modelName '/' blockName];
add_block('simulink/Sources/From Workspace', blockPath, 'Position', position);
set_param(blockPath, 'VariableName', variableName, 'SampleTime', '1');
end

function iAddToWorkspace(modelName, blockName, variableName, position)
blockPath = [modelName '/' blockName];
add_block('simulink/Sinks/To Workspace', blockPath, 'Position', position);
set_param(blockPath, 'VariableName', variableName, 'SaveFormat', 'Array');
end

function scriptText = iBuildRopeCoreScript()
thisDir = fileparts(mfilename('fullpath'));
sourcePath = fullfile(thisDir, '..', '..', '..', '+transformer_impl', '+layer', '+rope', 'streamingRopeSimulink.m');
scriptText = fileread(sourcePath);
end

function scriptText = iBuildPackedBeatUnpackScript()
scriptText = strjoin({ ...
    'function words = Unpack(packedBeat)', ...
    'words = fi(zeros(1, 8), 0, 32, 0);', ...
    'words(1) = fi(bitsliceget(packedBeat, 32, 1), 0, 32, 0);', ...
    'words(2) = fi(bitsliceget(packedBeat, 64, 33), 0, 32, 0);', ...
    'words(3) = fi(bitsliceget(packedBeat, 96, 65), 0, 32, 0);', ...
    'words(4) = fi(bitsliceget(packedBeat, 128, 97), 0, 32, 0);', ...
    'words(5) = fi(bitsliceget(packedBeat, 160, 129), 0, 32, 0);', ...
    'words(6) = fi(bitsliceget(packedBeat, 192, 161), 0, 32, 0);', ...
    'words(7) = fi(bitsliceget(packedBeat, 224, 193), 0, 32, 0);', ...
    'words(8) = fi(bitsliceget(packedBeat, 256, 225), 0, 32, 0);', ...
    'end'}, newline);
end

function scriptText = iBuildPackedBeatPackScript()
scriptText = strjoin({ ...
    'function packedBeat = Pack(words)', ...
    'packedBeat = fi(0, 0, 256, 0);', ...
    'packedBeat = bitconcat(words(8), words(7), words(6), words(5), words(4), words(3), words(2), words(1));', ...
    'end'}, newline);
end