function modelPath = buildRmsNormalizationModel()
% buildRmsNormalizationModel   Build the streaming RMSNorm Simulink DUT model.

thisDir = fileparts(mfilename('fullpath'));
modelName = 'rmsNormalization';
modelPath = fullfile(thisDir, [modelName '.slx']);
rtlVersion = 'v2';

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
set_param(modelName, 'Description', sprintf('RMSNorm packed RTL %s', rtlVersion));

assignin('base', 'simStopTime', 10);
assignin('base', 'resetSeq', timeseries([true; false], [0; 1]));
assignin('base', 'startSeq', timeseries([false; true; false], [0; 1; 2]));
assignin('base', 'cfgGammaValidSeq', timeseries([false; false], [0; 1]));
assignin('base', 'cfgGammaBeatSeq', timeseries(zeros(2, 8, 'single'), [0; 1]));
assignin('base', 'ddrDataValidSeq', timeseries([false; false], [0; 1]));
assignin('base', 'ddrDataBeatSeq', timeseries(zeros(2, 8, 'single'), [0; 1]));

iAddFromWorkspace(modelName, 'ResetSrc', 'resetSeq', [30 40 120 70]);
iAddFromWorkspace(modelName, 'StartSrc', 'startSeq', [30 90 120 120]);
iAddFromWorkspace(modelName, 'CfgBeatSrc', 'cfgGammaBeatSeq', [30 140 120 170]);
iAddFromWorkspace(modelName, 'CfgValidSrc', 'cfgGammaValidSeq', [30 190 120 220]);
iAddFromWorkspace(modelName, 'DdrBeatSrc', 'ddrDataBeatSeq', [30 240 120 270]);
iAddFromWorkspace(modelName, 'DdrValidSrc', 'ddrDataValidSeq', [30 290 120 320]);

add_block('simulink/Ports & Subsystems/Subsystem', [modelName '/DUT'], 'Position', [220 95 650 360]);
set_param([modelName '/DUT'], 'TreatAsAtomicUnit', 'on');
iBuildDutSubsystem(modelName);

add_block('simulink/Ports & Subsystems/Subsystem', [modelName '/DUTPacked'], 'Position', [220 395 650 660]);
set_param([modelName '/DUTPacked'], 'TreatAsAtomicUnit', 'on');
iBuildPackedDutSubsystem(modelName);

iAddToWorkspace(modelName, 'YBeat', 'YBeatOut', [740 125 830 155]);
iAddToWorkspace(modelName, 'YValid', 'YValidOut', [740 175 830 205]);
iAddToWorkspace(modelName, 'ReadAddr', 'ReadAddrOut', [740 225 830 255]);
iAddToWorkspace(modelName, 'ReadEn', 'ReadEnOut', [740 275 830 305]);
iAddToWorkspace(modelName, 'Done', 'DoneOut', [740 325 830 355]);

add_line(modelName, 'ResetSrc/1', 'DUT/1');
add_line(modelName, 'StartSrc/1', 'DUT/2');
add_line(modelName, 'CfgBeatSrc/1', 'DUT/3');
add_line(modelName, 'CfgValidSrc/1', 'DUT/4');
add_line(modelName, 'DdrBeatSrc/1', 'DUT/5');
add_line(modelName, 'DdrValidSrc/1', 'DUT/6');
add_line(modelName, 'DUT/1', 'YBeat/1');
add_line(modelName, 'DUT/2', 'YValid/1');
add_line(modelName, 'DUT/3', 'ReadAddr/1');
add_line(modelName, 'DUT/4', 'ReadEn/1');
add_line(modelName, 'DUT/5', 'Done/1');

save_system(modelName, modelPath);
close_system(modelName, 0);

end

function iBuildDutSubsystem(modelName)
subsystem = [modelName '/DUT'];
iBuildStreamingDutSubsystem(subsystem);
end

function iBuildStreamingDutSubsystem(subsystem)
iDeleteSubsystemContents(subsystem);

add_block('simulink/Ports & Subsystems/In1', [subsystem '/reset'], 'Position', [25 28 55 42], 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/start'], 'Position', [25 63 55 77], 'Port', '2', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgGammaBeat'], 'Position', [25 98 55 112], 'Port', '3', 'SampleTime', '1', 'OutDataTypeStr', 'single');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgGammaValid'], 'Position', [25 133 55 147], 'Port', '4', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/ddrDataBeat'], 'Position', [25 168 55 182], 'Port', '5', 'SampleTime', '1', 'OutDataTypeStr', 'single');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/ddrDataValid'], 'Position', [25 203 55 217], 'Port', '6', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');

add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/SquareBeat'], 'Position', [105 150 230 250]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/BeatReduce'], 'Position', [255 150 390 250]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/BeatAccumulator'], 'Position', [415 150 585 255]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/ScalarRsqrt'], 'Position', [415 75 570 145]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/InvRmsLatch'], 'Position', [415 10 585 70]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/LaneMultiply'], 'Position', [640 165 825 290]);
add_block('simulink/User-Defined Functions/MATLAB Function', [subsystem '/Controller'], 'Position', [105 20 340 125]);
set_param([subsystem '/Controller'], 'SystemSampleTime', '1');
open_system([subsystem '/Controller']);
rt = sfroot;
chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [subsystem '/Controller']);
chart.Script = iBuildControllerScript();

add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/CfgBeatSpec'], ...
    'Dimensions', '[1 8]', 'OutDataTypeStr', 'single', 'Position', [80 90 130 120]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/DdrBeatSpec'], ...
    'Dimensions', '[1 8]', 'OutDataTypeStr', 'single', 'Position', [80 160 130 190]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/CurrentSumSpec'], ...
    'Dimensions', '1', 'OutDataTypeStr', 'single', 'Position', [600 198 655 222]);

add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outBeat'], 'Position', [865 208 895 222]);
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outValid'], 'Position', [865 248 895 262], 'Port', '2');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/ddrReadAddr'], 'Position', [865 58 895 72], 'Port', '3');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/ddrReadEn'], 'Position', [865 93 895 107], 'Port', '4');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/done'], 'Position', [865 128 895 142], 'Port', '5');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/busy'], 'Position', [865 163 895 177], 'Port', '6');

set_param([subsystem '/SquareBeat'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/BeatReduce'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/BeatAccumulator'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/ScalarRsqrt'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/InvRmsLatch'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/LaneMultiply'], 'TreatAsAtomicUnit', 'on');
hdlset_param([subsystem '/SquareBeat'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/BeatReduce'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/BeatAccumulator'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/ScalarRsqrt'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/InvRmsLatch'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/LaneMultiply'], 'BalanceDelays', 'on');

iBuildSquareBeatSubsystem(subsystem);
iBuildBeatReduceSubsystem(subsystem);
iBuildBeatAccumulatorSubsystem(subsystem);
iBuildScalarRsqrtSubsystem(subsystem);
iBuildInvRmsLatchSubsystem(subsystem);
iBuildLaneMultiplySubsystem(subsystem);

add_line(subsystem, 'ddrDataBeat/1', 'SquareBeat/1');
add_line(subsystem, 'ddrDataValid/1', 'SquareBeat/2');
add_line(subsystem, 'SquareBeat/1', 'BeatReduce/1');
add_line(subsystem, 'SquareBeat/2', 'BeatReduce/2');
add_line(subsystem, 'BeatReduce/1', 'BeatAccumulator/1');
add_line(subsystem, 'BeatReduce/2', 'BeatAccumulator/2');
add_line(subsystem, 'reset/1', 'BeatAccumulator/3');
add_line(subsystem, 'BeatAccumulator/1', 'CurrentSumSpec/1');
add_line(subsystem, 'CurrentSumSpec/1', 'ScalarRsqrt/1');
add_line(subsystem, 'BeatAccumulator/2', 'ScalarRsqrt/2');
add_line(subsystem, 'InvRmsLatch/3', 'BeatAccumulator/4');
add_line(subsystem, 'ScalarRsqrt/1', 'InvRmsLatch/1');
add_line(subsystem, 'ScalarRsqrt/2', 'InvRmsLatch/2');
add_line(subsystem, 'Controller/5', 'InvRmsLatch/3');
add_line(subsystem, 'reset/1', 'InvRmsLatch/4');
add_line(subsystem, 'InvRmsLatch/1', 'LaneMultiply/5');
add_line(subsystem, 'InvRmsLatch/2', 'LaneMultiply/6');

add_line(subsystem, 'reset/1', 'Controller/1');
add_line(subsystem, 'start/1', 'Controller/2');
add_line(subsystem, 'cfgGammaBeat/1', 'CfgBeatSpec/1');
add_line(subsystem, 'CfgBeatSpec/1', 'Controller/3');
add_line(subsystem, 'cfgGammaValid/1', 'Controller/4');
add_line(subsystem, 'ddrDataBeat/1', 'DdrBeatSpec/1');
add_line(subsystem, 'DdrBeatSpec/1', 'Controller/5');
add_line(subsystem, 'ddrDataValid/1', 'Controller/6');
add_line(subsystem, 'Controller/1', 'LaneMultiply/1');
add_line(subsystem, 'Controller/2', 'LaneMultiply/2');
add_line(subsystem, 'Controller/3', 'LaneMultiply/3');
add_line(subsystem, 'Controller/4', 'LaneMultiply/4');
add_line(subsystem, 'LaneMultiply/1', 'outBeat/1');
add_line(subsystem, 'LaneMultiply/2', 'outValid/1');
add_line(subsystem, 'Controller/6', 'ddrReadAddr/1');
add_line(subsystem, 'Controller/7', 'ddrReadEn/1');
add_line(subsystem, 'Controller/8', 'done/1');
add_line(subsystem, 'Controller/9', 'busy/1');
end

function iBuildPackedDutSubsystem(modelName)
subsystem = [modelName '/DUTPacked'];
iDeleteSubsystemContents(subsystem);

add_block('simulink/Ports & Subsystems/In1', [subsystem '/reset'], 'Position', [25 28 55 42], 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/start'], 'Position', [25 63 55 77], 'Port', '2', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgGammaBeat'], 'Position', [25 98 55 112], 'Port', '3', 'SampleTime', '1', 'OutDataTypeStr', 'fixdt(0,256,0)');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgGammaValid'], 'Position', [25 133 55 147], 'Port', '4', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/ddrDataBeat'], 'Position', [25 168 55 182], 'Port', '5', 'SampleTime', '1', 'OutDataTypeStr', 'fixdt(0,256,0)');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/ddrDataValid'], 'Position', [25 203 55 217], 'Port', '6', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');

add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/CfgBeatUnpack'], 'Position', [105 80 220 130]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/DdrBeatUnpack'], 'Position', [105 150 220 200]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/CoreDUT'], 'Position', [255 35 685 300]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/OutBeatPack'], 'Position', [720 80 835 130]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/OutBeatSpec'], ...
    'Dimensions', '1', 'OutDataTypeStr', 'fixdt(0,256,0)', 'Position', [855 93 920 117]);

add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outBeat'], 'Position', [950 98 980 112]);
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outValid'], 'Position', [950 173 980 187], 'Port', '2');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/ddrReadAddr'], 'Position', [950 208 980 222], 'Port', '3');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/ddrReadEn'], 'Position', [950 243 980 257], 'Port', '4');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/done'], 'Position', [950 278 980 292], 'Port', '5');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/busy'], 'Position', [950 313 980 327], 'Port', '6');

set_param([subsystem '/CfgBeatUnpack'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/DdrBeatUnpack'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/CoreDUT'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/OutBeatPack'], 'TreatAsAtomicUnit', 'on');

iBuildPackedBeatUnpackSubsystem([subsystem '/CfgBeatUnpack']);
iBuildPackedBeatUnpackSubsystem([subsystem '/DdrBeatUnpack']);
iBuildStreamingDutSubsystem([subsystem '/CoreDUT']);
iBuildPackedBeatPackSubsystem([subsystem '/OutBeatPack']);

add_line(subsystem, 'reset/1', 'CoreDUT/1');
add_line(subsystem, 'start/1', 'CoreDUT/2');
add_line(subsystem, 'cfgGammaBeat/1', 'CfgBeatUnpack/1');
add_line(subsystem, 'CfgBeatUnpack/1', 'CoreDUT/3');
add_line(subsystem, 'cfgGammaValid/1', 'CoreDUT/4');
add_line(subsystem, 'ddrDataBeat/1', 'DdrBeatUnpack/1');
add_line(subsystem, 'DdrBeatUnpack/1', 'CoreDUT/5');
add_line(subsystem, 'ddrDataValid/1', 'CoreDUT/6');
add_line(subsystem, 'CoreDUT/1', 'OutBeatPack/1');
add_line(subsystem, 'OutBeatPack/1', 'OutBeatSpec/1');
add_line(subsystem, 'OutBeatSpec/1', 'outBeat/1');
add_line(subsystem, 'CoreDUT/2', 'outValid/1');
add_line(subsystem, 'CoreDUT/3', 'ddrReadAddr/1');
add_line(subsystem, 'CoreDUT/4', 'ddrReadEn/1');
add_line(subsystem, 'CoreDUT/5', 'done/1');
add_line(subsystem, 'CoreDUT/6', 'busy/1');
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

function iBuildSquareBeatSubsystem(subsystem)
target = [subsystem '/SquareBeat'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/ddrDataBeat'], 'Position', [30 38 60 52]);
add_block('simulink/Ports & Subsystems/In1', [target '/ddrDataBeatValid'], 'Position', [30 83 60 97], 'Port', '2', 'OutDataTypeStr', 'boolean');
add_block('simulink/Math Operations/Product', [target '/Square'], 'Inputs', '2', ...
    'Multiplication', 'Element-wise(.*)', 'Position', [100 27 170 63]);
add_block('simulink/Ports & Subsystems/Out1', [target '/squaredBeat'], 'Position', [210 38 240 52]);
add_block('simulink/Ports & Subsystems/Out1', [target '/squaredBeatValid'], 'Position', [210 88 240 102], 'Port', '2');
add_line(target, 'ddrDataBeat/1', 'Square/1');
add_line(target, 'ddrDataBeat/1', 'Square/2', 'autorouting', 'on');
add_line(target, 'Square/1', 'squaredBeat/1');
add_line(target, 'ddrDataBeatValid/1', 'squaredBeatValid/1');
end

function iBuildBeatReduceSubsystem(subsystem)
target = [subsystem '/BeatReduce'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/squaredBeat'], 'Position', [30 38 60 52]);
add_block('simulink/Ports & Subsystems/In1', [target '/squaredBeatValid'], 'Position', [30 118 60 132], 'Port', '2', 'OutDataTypeStr', 'boolean');
add_block('simulink/Signal Routing/Demux', [target '/Demux'], 'Outputs', '8', 'Position', [90 20 95 100]);
add_block('simulink/Math Operations/Add', [target '/Add01'], 'Inputs', '++', 'Position', [125 18 160 42]);
add_block('simulink/Math Operations/Add', [target '/Add23'], 'Inputs', '++', 'Position', [125 48 160 72]);
add_block('simulink/Math Operations/Add', [target '/Add45'], 'Inputs', '++', 'Position', [125 78 160 102]);
add_block('simulink/Math Operations/Add', [target '/Add67'], 'Inputs', '++', 'Position', [125 108 160 132]);
add_block('simulink/Math Operations/Add', [target '/Add0123'], 'Inputs', '++', 'Position', [190 33 225 57]);
add_block('simulink/Math Operations/Add', [target '/Add4567'], 'Inputs', '++', 'Position', [190 93 225 117]);
add_block('simulink/Math Operations/Add', [target '/AddAll'], 'Inputs', '++', 'Position', [255 63 290 87]);
add_block('simulink/Ports & Subsystems/Out1', [target '/beatSum'], 'Position', [325 68 355 82]);
add_block('simulink/Ports & Subsystems/Out1', [target '/beatSumValid'], 'Position', [325 118 355 132], 'Port', '2');
add_line(target, 'squaredBeat/1', 'Demux/1');
add_line(target, 'Demux/1', 'Add01/1');
add_line(target, 'Demux/2', 'Add01/2');
add_line(target, 'Demux/3', 'Add23/1');
add_line(target, 'Demux/4', 'Add23/2');
add_line(target, 'Demux/5', 'Add45/1');
add_line(target, 'Demux/6', 'Add45/2');
add_line(target, 'Demux/7', 'Add67/1');
add_line(target, 'Demux/8', 'Add67/2');
add_line(target, 'Add01/1', 'Add0123/1');
add_line(target, 'Add23/1', 'Add0123/2');
add_line(target, 'Add45/1', 'Add4567/1');
add_line(target, 'Add67/1', 'Add4567/2');
add_line(target, 'Add0123/1', 'AddAll/1');
add_line(target, 'Add4567/1', 'AddAll/2');
add_line(target, 'AddAll/1', 'beatSum/1');
add_line(target, 'squaredBeatValid/1', 'beatSumValid/1');
end

function iBuildScalarRsqrtSubsystem(subsystem)
target = [subsystem '/ScalarRsqrt'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/currentSum'], 'Position', [30 38 60 52]);
add_block('simulink/Ports & Subsystems/In1', [target '/currentSumValid'], 'Position', [30 83 60 97], 'Port', '2', 'OutDataTypeStr', 'boolean');
add_block('simulink/Sources/Constant', [target '/Epsilon'], 'Value', 'single(1536e-6)', ...
    'SampleTime', '1', 'Position', [90 78 160 102]);
add_block('simulink/Math Operations/Add', [target '/RsqrtOperandAdd'], 'Inputs', '++', ...
    'OutDataTypeStr', 'single', 'Position', [90 28 150 62]);
add_block('simulink/Math Operations/Math Function', [target '/Sqrt'], 'Operator', 'sqrt', ...
    'Position', [175 28 225 62]);
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/SqrtSingle'], ...
    'OutDataTypeStr', 'single', 'Position', [240 28 290 62]);
add_block('simulink/Sources/Constant', [target '/One'], 'Value', 'single(1)', ...
    'SampleTime', '1', 'Position', [240 78 300 102]);
add_block('simulink/Math Operations/Divide', [target '/Reciprocal'], 'OutDataTypeStr', 'single', ...
    'SaturateOnIntegerOverflow', 'on', 'RndMeth', 'Zero', 'Position', [315 28 370 62]);
add_block('simulink/Ports & Subsystems/Out1', [target '/invRms'], 'Position', [405 38 435 52]);
add_block('simulink/Ports & Subsystems/Out1', [target '/invRmsValid'], 'Position', [405 88 435 102], 'Port', '2');
add_line(target, 'currentSum/1', 'RsqrtOperandAdd/1');
add_line(target, 'Epsilon/1', 'RsqrtOperandAdd/2');
add_line(target, 'RsqrtOperandAdd/1', 'Sqrt/1');
add_line(target, 'Sqrt/1', 'SqrtSingle/1');
add_line(target, 'One/1', 'Reciprocal/1');
add_line(target, 'SqrtSingle/1', 'Reciprocal/2');
add_line(target, 'Reciprocal/1', 'invRms/1');
add_line(target, 'currentSumValid/1', 'invRmsValid/1');
end

function iBuildBeatAccumulatorSubsystem(subsystem)
target = [subsystem '/BeatAccumulator'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/beatSum'], 'Position', [30 28 60 42]);
add_block('simulink/Ports & Subsystems/In1', [target '/beatSumValid'], 'Position', [30 63 60 77], 'Port', '2', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [target '/reset'], 'Position', [30 98 60 112], 'Port', '3', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [target '/clearAccumulator'], 'Position', [30 133 60 147], 'Port', '4', 'OutDataTypeStr', 'boolean');
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/BeatSumSingle'], ...
    'OutDataTypeStr', 'single', 'Position', [80 18 115 42]);
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/BeatSumValidBool'], ...
    'OutDataTypeStr', 'boolean', 'Position', [80 53 115 77]);
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/BeatSumValidSingle'], ...
    'OutDataTypeStr', 'single', 'Position', [120 53 155 77]);
add_block('simulink/Discrete/Unit Delay', [target '/ClearAccumulatorDelay'], ...
    'InitialCondition', 'false', 'Position', [80 133 115 157]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [target '/ResetOrClear'], ...
    'Operator', 'OR', 'Position', [130 108 165 142]);
add_block('simulink/Sources/Constant', [target '/ZeroSingle'], ...
    'Value', 'single(0)', 'SampleTime', '1', 'Position', [365 18 425 42]);
add_block('simulink/Discrete/Unit Delay', [target '/BankIndexState'], ...
    'InitialCondition', 'uint8(0)', 'Position', [250 178 295 202]);
add_block('simulink/Sources/Constant', [target '/BankIndexOne'], ...
    'Value', 'uint8(1)', 'SampleTime', '1', 'Position', [135 148 185 172]);
add_block('simulink/Sources/Constant', [target '/BankIndexZero'], ...
    'Value', 'uint8(0)', 'SampleTime', '1', 'Position', [135 238 185 262]);
add_block('simulink/Math Operations/Add', [target '/BankIndexAdd'], ...
    'Inputs', '++', 'OutDataTypeStr', 'uint8', 'Position', [205 148 240 172]);
add_block('simulink/Logic and Bit Operations/Compare To Constant', [target '/BankIndexIsLast'], ...
    'relop', '==', 'const', '10', 'Position', [205 198 255 222]);
add_block('simulink/Signal Routing/Switch', [target '/BankIndexWrap'], ...
    'Criteria', 'u2 ~= 0', 'Threshold', '0.5', 'Position', [280 133 315 187]);
add_block('simulink/Signal Routing/Switch', [target '/BankIndexAdvance'], ...
    'Criteria', 'u2 ~= 0', 'Threshold', '0.5', 'Position', [340 153 375 207]);
add_block('simulink/Signal Routing/Switch', [target '/BankIndexClear'], ...
    'Criteria', 'u2 ~= 0', 'Threshold', '0.5', 'Position', [400 173 435 227]);
add_line(target, 'beatSum/1', 'BeatSumSingle/1');
add_line(target, 'beatSumValid/1', 'BeatSumValidBool/1');
add_line(target, 'beatSumValid/1', 'BeatSumValidSingle/1');
add_line(target, 'clearAccumulator/1', 'ClearAccumulatorDelay/1');
add_line(target, 'reset/1', 'ResetOrClear/1');
add_line(target, 'ClearAccumulatorDelay/1', 'ResetOrClear/2');
add_line(target, 'BankIndexState/1', 'BankIndexAdd/1');
add_line(target, 'BankIndexOne/1', 'BankIndexAdd/2');
add_line(target, 'BankIndexState/1', 'BankIndexIsLast/1');
add_line(target, 'BankIndexZero/1', 'BankIndexWrap/1');
add_line(target, 'BankIndexIsLast/1', 'BankIndexWrap/2');
add_line(target, 'BankIndexAdd/1', 'BankIndexWrap/3');
add_line(target, 'BankIndexWrap/1', 'BankIndexAdvance/1');
add_line(target, 'BeatSumValidBool/1', 'BankIndexAdvance/2');
add_line(target, 'BankIndexState/1', 'BankIndexAdvance/3');
add_line(target, 'BankIndexZero/1', 'BankIndexClear/1');
add_line(target, 'ResetOrClear/1', 'BankIndexClear/2');
add_line(target, 'BankIndexAdvance/1', 'BankIndexClear/3');
add_line(target, 'BankIndexClear/1', 'BankIndexState/1');

for bankIndex = 0:10
    suffix = sprintf('%02d', bankIndex);
    compareName = ['Sel' suffix];
    selectName = ['SelSingle' suffix];
    updateName = ['UpdateEn' suffix];
    gateName = ['Gate' suffix];
    addName = ['AccumulateAdd' suffix];
    holdName = ['HoldOrAcc' suffix];
    clearName = ['ClearPipe' suffix];
    delayName = ['FeedbackDelay' suffix];
    stateName = ['BankState' suffix];
    add_block('simulink/Logic and Bit Operations/Compare To Constant', [target '/' compareName], ...
        'relop', '==', 'const', sprintf('%d', bankIndex), ...
        'Position', [150 20 + bankIndex*35 200 44 + bankIndex*35]);
    add_block('simulink/Signal Attributes/Data Type Conversion', [target '/' selectName], ...
        'OutDataTypeStr', 'single', 'Position', [220 20 + bankIndex*35 255 44 + bankIndex*35]);
    add_block('simulink/Logic and Bit Operations/Logical Operator', [target '/' updateName], ...
        'Operator', 'AND', 'Position', [220 50 + bankIndex*35 255 74 + bankIndex*35]);
    add_block('simulink/Math Operations/Product', [target '/' gateName], ...
        'Inputs', '3', 'Multiplication', 'Element-wise(.*)', ...
        'Position', [275 12 + bankIndex*35 335 52 + bankIndex*35]);
    add_block('simulink/Math Operations/Add', [target '/' addName], ...
        'Inputs', '++', 'Position', [495 17 + bankIndex*35 530 41 + bankIndex*35]);
    add_block('simulink/Signal Routing/Switch', [target '/' holdName], ...
        'Criteria', 'u2 ~= 0', 'Threshold', '0.5', ...
        'Position', [540 10 + bankIndex*35 575 48 + bankIndex*35]);
    add_block('simulink/Signal Routing/Switch', [target '/' clearName], ...
        'Criteria', 'u2 ~= 0', 'Threshold', '0.5', ...
        'Position', [590 10 + bankIndex*35 625 48 + bankIndex*35]);
    add_block('simulink/Discrete/Delay', [target '/' delayName], ...
        'DelayLength', '10', 'InitialCondition', 'single(0)', ...
        'Position', [385 10 + bankIndex*35 440 48 + bankIndex*35]);
    add_block('simulink/Discrete/Unit Delay', [target '/' stateName], ...
        'InitialCondition', 'single(0)', 'Position', [650 17 + bankIndex*35 695 41 + bankIndex*35]);
    add_line(target, 'BankIndexState/1', [compareName '/1']);
    add_line(target, [compareName '/1'], [selectName '/1']);
    add_line(target, [compareName '/1'], [updateName '/1']);
    add_line(target, 'BeatSumValidBool/1', [updateName '/2']);
    add_line(target, 'BeatSumSingle/1', [gateName '/1']);
    add_line(target, 'BeatSumValidSingle/1', [gateName '/2']);
    add_line(target, [selectName '/1'], [gateName '/3']);
    add_line(target, [delayName '/1'], [addName '/1']);
    add_line(target, [gateName '/1'], [addName '/2']);
    add_line(target, [addName '/1'], [holdName '/1']);
    add_line(target, [updateName '/1'], [holdName '/2']);
    add_line(target, [stateName '/1'], [holdName '/3']);
    add_line(target, 'ZeroSingle/1', [clearName '/1']);
    add_line(target, 'ResetOrClear/1', [clearName '/2']);
    add_line(target, [holdName '/1'], [clearName '/3']);
    add_line(target, [clearName '/1'], [stateName '/1']);
    add_line(target, [stateName '/1'], [delayName '/1']);
end

add_block('simulink/Math Operations/Add', [target '/Sum01'], 'Inputs', '++', 'OutDataTypeStr', 'single', 'Position', [690 22 725 46]);
add_block('simulink/Math Operations/Add', [target '/Sum23'], 'Inputs', '++', 'OutDataTypeStr', 'single', 'Position', [690 92 725 116]);
add_block('simulink/Math Operations/Add', [target '/Sum45'], 'Inputs', '++', 'OutDataTypeStr', 'single', 'Position', [690 162 725 186]);
add_block('simulink/Math Operations/Add', [target '/Sum67'], 'Inputs', '++', 'OutDataTypeStr', 'single', 'Position', [690 232 725 256]);
add_block('simulink/Math Operations/Add', [target '/Sum89'], 'Inputs', '++', 'OutDataTypeStr', 'single', 'Position', [690 302 725 326]);
add_block('simulink/Math Operations/Add', [target '/Sum0123'], 'Inputs', '++', 'OutDataTypeStr', 'single', 'Position', [760 57 795 81]);
add_block('simulink/Math Operations/Add', [target '/Sum4567'], 'Inputs', '++', 'OutDataTypeStr', 'single', 'Position', [760 197 795 221]);
add_block('simulink/Math Operations/Add', [target '/Sum8910'], 'Inputs', '++', 'OutDataTypeStr', 'single', 'Position', [760 317 795 341]);
add_block('simulink/Math Operations/Add', [target '/Sum01234567'], 'Inputs', '++', 'OutDataTypeStr', 'single', 'Position', [830 122 870 146]);
add_block('simulink/Math Operations/Add', [target '/SumAll'], 'Inputs', '++', 'OutDataTypeStr', 'single', 'Position', [900 212 940 236]);
add_block('simulink/Ports & Subsystems/Out1', [target '/currentSum'], 'Position', [980 217 1010 231]);
add_block('simulink/Ports & Subsystems/Out1', [target '/currentSumValid'], 'Position', [980 257 1010 271], 'Port', '2');
add_line(target, 'BankState00/1', 'Sum01/1');
add_line(target, 'BankState01/1', 'Sum01/2');
add_line(target, 'BankState02/1', 'Sum23/1');
add_line(target, 'BankState03/1', 'Sum23/2');
add_line(target, 'BankState04/1', 'Sum45/1');
add_line(target, 'BankState05/1', 'Sum45/2');
add_line(target, 'BankState06/1', 'Sum67/1');
add_line(target, 'BankState07/1', 'Sum67/2');
add_line(target, 'BankState08/1', 'Sum89/1');
add_line(target, 'BankState09/1', 'Sum89/2');
add_line(target, 'Sum01/1', 'Sum0123/1');
add_line(target, 'Sum23/1', 'Sum0123/2');
add_line(target, 'Sum45/1', 'Sum4567/1');
add_line(target, 'Sum67/1', 'Sum4567/2');
add_line(target, 'Sum89/1', 'Sum8910/1');
add_line(target, 'BankState10/1', 'Sum8910/2');
add_line(target, 'Sum0123/1', 'Sum01234567/1');
add_line(target, 'Sum4567/1', 'Sum01234567/2');
add_line(target, 'Sum01234567/1', 'SumAll/1');
add_line(target, 'Sum8910/1', 'SumAll/2');
add_line(target, 'SumAll/1', 'currentSum/1');
add_line(target, 'BeatSumValidBool/1', 'currentSumValid/1');
end

function iBuildInvRmsLatchSubsystem(subsystem)
target = [subsystem '/InvRmsLatch'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/invRmsIn'], 'Position', [30 33 60 47]);
add_block('simulink/Ports & Subsystems/In1', [target '/invRmsInValid'], 'Position', [30 68 60 82], 'Port', '2', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [target '/capture'], 'Position', [30 103 60 117], 'Port', '3', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [target '/reset'], 'Position', [30 138 60 152], 'Port', '4', 'OutDataTypeStr', 'boolean');
add_block('simulink/Logic and Bit Operations/Logical Operator', [target '/CaptureAccepted'], ...
    'Operator', 'AND', 'Position', [85 83 120 107]);
add_block('simulink/Discrete/Unit Delay', [target '/StoredInvRms'], ...
    'InitialCondition', 'single(0)', 'Position', [310 28 355 52]);
add_block('simulink/Signal Routing/Switch', [target '/SelectNew'], ...
    'Criteria', 'u2 ~= 0', 'Threshold', '0.5', 'Position', [145 22 180 88]);
add_block('simulink/Sources/Constant', [target '/ZeroSingle'], ...
    'Value', 'single(0)', 'SampleTime', '1', 'Position', [145 138 205 162]);
add_block('simulink/Signal Routing/Switch', [target '/ResetStored'], ...
    'Criteria', 'u2 ~= 0', 'Threshold', '0.5', 'Position', [225 32 260 98]);
add_block('simulink/Discrete/Unit Delay', [target '/CaptureAcceptedDelay'], ...
    'InitialCondition', 'false', 'Position', [310 88 355 112]);
add_block('simulink/Discrete/Unit Delay', [target '/LatchedValidState'], ...
    'InitialCondition', 'false', 'Position', [310 133 355 157]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [target '/HoldValid'], ...
    'Operator', 'OR', 'Position', [225 118 260 142]);
add_block('simulink/Sources/Constant', [target '/FalseConst'], ...
    'Value', 'false', 'SampleTime', '1', 'OutDataTypeStr', 'boolean', 'Position', [225 168 280 192]);
add_block('simulink/Signal Routing/Switch', [target '/ResetValid'], ...
    'Criteria', 'u2 ~= 0', 'Threshold', '0.5', 'Position', [385 118 420 184]);
add_block('simulink/Ports & Subsystems/Out1', [target '/invRmsLatched'], 'Position', [455 38 485 52]);
add_block('simulink/Ports & Subsystems/Out1', [target '/invRmsLatchedValid'], 'Position', [455 98 485 112], 'Port', '2');
add_block('simulink/Ports & Subsystems/Out1', [target '/clearAccumulator'], 'Position', [455 158 485 172], 'Port', '3');
add_line(target, 'capture/1', 'CaptureAccepted/1');
add_line(target, 'invRmsInValid/1', 'CaptureAccepted/2');
add_line(target, 'invRmsIn/1', 'SelectNew/1');
add_line(target, 'CaptureAccepted/1', 'SelectNew/2');
add_line(target, 'StoredInvRms/1', 'SelectNew/3');
add_line(target, 'ZeroSingle/1', 'ResetStored/1');
add_line(target, 'reset/1', 'ResetStored/2');
add_line(target, 'SelectNew/1', 'ResetStored/3');
add_line(target, 'ResetStored/1', 'StoredInvRms/1');
add_line(target, 'StoredInvRms/1', 'invRmsLatched/1');
add_line(target, 'CaptureAccepted/1', 'CaptureAcceptedDelay/1');
add_line(target, 'CaptureAccepted/1', 'HoldValid/1');
add_line(target, 'LatchedValidState/1', 'HoldValid/2');
add_line(target, 'FalseConst/1', 'ResetValid/1');
add_line(target, 'reset/1', 'ResetValid/2');
add_line(target, 'HoldValid/1', 'ResetValid/3');
add_line(target, 'ResetValid/1', 'LatchedValidState/1');
add_line(target, 'LatchedValidState/1', 'invRmsLatchedValid/1');
add_line(target, 'CaptureAcceptedDelay/1', 'clearAccumulator/1');
end

function iBuildLaneMultiplySubsystem(subsystem)
target = [subsystem '/LaneMultiply'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/xBeat'], 'Position', [30 43 60 57]);
add_block('simulink/Ports & Subsystems/In1', [target '/xBeatValid'], 'Position', [30 78 60 92], 'Port', '2', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [target '/gBeat'], 'Position', [30 123 60 137], 'Port', '3');
add_block('simulink/Ports & Subsystems/In1', [target '/gBeatValid'], 'Position', [30 158 60 172], 'Port', '4', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [target '/invRms'], 'Position', [30 203 60 217], 'Port', '5');
add_block('simulink/Ports & Subsystems/In1', [target '/invRmsValid'], 'Position', [30 238 60 252], 'Port', '6', 'OutDataTypeStr', 'boolean');
add_block('simulink/Signal Attributes/Signal Specification', [target '/InvRmsScalar'], ...
    'Dimensions', '1', 'Position', [90 198 145 222]);
add_block('simulink/Math Operations/Gain', [target '/ScaleGamma'], 'Gain', 'single(sqrt(1536))', ...
    'Position', [90 118 150 142]);
add_block('simulink/Math Operations/Product', [target '/WeightMultiply'], 'Inputs', '2', ...
    'Multiplication', 'Element-wise(.*)', 'Position', [165 83 235 117]);
add_block('simulink/Math Operations/Product', [target '/ScaleMultiply'], 'Inputs', '2', ...
    'Multiplication', 'Element-wise(.*)', 'Position', [270 123 345 167]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [target '/InputBeatValid'], ...
    'Operator', 'AND', 'Inputs', '2', 'Position', [165 18 200 52]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [target '/YBeatValid'], ...
    'Operator', 'AND', 'Inputs', '2', 'Position', [270 18 305 52]);
add_block('simulink/Ports & Subsystems/Out1', [target '/yBeat'], 'Position', [380 138 410 152]);
add_block('simulink/Ports & Subsystems/Out1', [target '/yBeatValid'], 'Position', [380 33 410 47], 'Port', '2');
add_line(target, 'xBeat/1', 'WeightMultiply/1');
add_line(target, 'gBeat/1', 'ScaleGamma/1');
add_line(target, 'ScaleGamma/1', 'WeightMultiply/2');
add_line(target, 'WeightMultiply/1', 'ScaleMultiply/1');
add_line(target, 'invRms/1', 'InvRmsScalar/1');
add_line(target, 'InvRmsScalar/1', 'ScaleMultiply/2');
add_line(target, 'ScaleMultiply/1', 'yBeat/1');
add_line(target, 'xBeatValid/1', 'InputBeatValid/1');
add_line(target, 'gBeatValid/1', 'InputBeatValid/2');
add_line(target, 'InputBeatValid/1', 'YBeatValid/1');
add_line(target, 'invRmsValid/1', 'YBeatValid/2');
add_line(target, 'YBeatValid/1', 'yBeatValid/1');
end

function iAddFromWorkspace(modelName, blockName, variableName, position)
add_block('simulink/Sources/From Workspace', [modelName '/' blockName], 'VariableName', variableName, 'Position', position);
end

function iAddToWorkspace(modelName, blockName, variableName, position)
add_block('simulink/Sinks/To Workspace', [modelName '/' blockName], ...
    'VariableName', variableName, 'SaveFormat', 'Array', 'Position', position);
end

function iDeleteSubsystemContents(subsystem)
children = find_system(subsystem, 'SearchDepth', 1, 'Type', 'Block');
for childIndex = 1:numel(children)
    if ~strcmp(children{childIndex}, subsystem)
        delete_block(children{childIndex});
    end
end
lines = find_system(subsystem, 'FindAll', 'on', 'SearchDepth', 1, 'Type', 'Line');
for lineIndex = 1:numel(lines)
    delete_line(lines(lineIndex));
end
end

function scriptText = iBuildControllerScript()
templatePath = fullfile(fileparts(mfilename('fullpath')), 'private', 'rmsNormalizationControllerTemplate.txt');
scriptText = fileread(templatePath);
end

function scriptText = iBuildPackedBeatUnpackScript()
scriptText = strjoin({
    'function words = Unpack(packedBeat)'
    '%#codegen'
    'words = fi(zeros(1,8), 0, 32, 0);'
    'words(1) = fi(bitsliceget(packedBeat, 32, 1), 0, 32, 0);'
    'words(2) = fi(bitsliceget(packedBeat, 64, 33), 0, 32, 0);'
    'words(3) = fi(bitsliceget(packedBeat, 96, 65), 0, 32, 0);'
    'words(4) = fi(bitsliceget(packedBeat, 128, 97), 0, 32, 0);'
    'words(5) = fi(bitsliceget(packedBeat, 160, 129), 0, 32, 0);'
    'words(6) = fi(bitsliceget(packedBeat, 192, 161), 0, 32, 0);'
    'words(7) = fi(bitsliceget(packedBeat, 224, 193), 0, 32, 0);'
    'words(8) = fi(bitsliceget(packedBeat, 256, 225), 0, 32, 0);'
    'end'}, newline);
end

function scriptText = iBuildPackedBeatPackScript()
scriptText = strjoin({
    'function packedBeat = Pack(words)'
    '%#codegen'
    'packedBeat = bitconcat(words(8), words(7), words(6), words(5), words(4), words(3), words(2), words(1));'
    'end'}, newline);
end
