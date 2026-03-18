function modelPath = buildRmsNormalizationModel()
% buildRmsNormalizationModel   Build the streaming RMSNorm Simulink DUT model.

thisDir = fileparts(mfilename('fullpath'));
modelName = 'rmsNormalization';
modelPath = fullfile(thisDir, [modelName '.slx']);

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
iDeleteSubsystemContents(subsystem);

add_block('simulink/Ports & Subsystems/In1', [subsystem '/reset'], 'Position', [25 28 55 42], 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/start'], 'Position', [25 63 55 77], 'Port', '2', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgGammaBeat'], 'Position', [25 98 55 112], 'Port', '3', 'SampleTime', '1', 'OutDataTypeStr', 'single');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgGammaValid'], 'Position', [25 133 55 147], 'Port', '4', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/ddrDataBeat'], 'Position', [25 168 55 182], 'Port', '5', 'SampleTime', '1', 'OutDataTypeStr', 'single');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/ddrDataValid'], 'Position', [25 203 55 217], 'Port', '6', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');

add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/SquareBeat'], 'Position', [105 150 210 230]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/BeatReduce'], 'Position', [235 150 340 230]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/BeatAccumulator'], 'Position', [370 150 540 230]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/ScalarRsqrt'], 'Position', [370 85 485 145]);
add_block('simulink/Discrete/Unit Delay', [subsystem '/InvRmsDelay'], 'Position', [510 93 545 117]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/InvRmsLatch'], 'Position', [370 20 540 70]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/LaneMultiply'], 'Position', [370 185 540 285]);
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
    'Dimensions', '1', 'OutDataTypeStr', 'single', 'Position', [555 178 610 202]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/InvRmsSpec'], ...
    'Dimensions', '1', 'OutDataTypeStr', 'single', 'Position', [555 92 610 116]);
add_block('simulink/Discrete/Unit Delay', [subsystem '/CaptureClearDelay'], ...
    'InitialCondition', 'false', 'Position', [350 238 385 262]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [subsystem '/ClearAccumulator'], ...
    'Operator', 'OR', 'Position', [280 235 315 265]);

add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outBeat'], 'Position', [595 208 625 222]);
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outValid'], 'Position', [595 248 625 262], 'Port', '2');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/ddrReadAddr'], 'Position', [595 58 625 72], 'Port', '3');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/ddrReadEn'], 'Position', [595 93 625 107], 'Port', '4');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/done'], 'Position', [595 128 625 142], 'Port', '5');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/busy'], 'Position', [595 163 625 177], 'Port', '6');

set_param([subsystem '/SquareBeat'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/BeatReduce'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/BeatAccumulator'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/ScalarRsqrt'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/InvRmsLatch'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/LaneMultiply'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/InvRmsDelay'], 'InitialCondition', 'single(0)');

iBuildSquareBeatSubsystem(subsystem);
iBuildBeatReduceSubsystem(subsystem);
iBuildBeatAccumulatorSubsystem(subsystem);
iBuildScalarRsqrtSubsystem(subsystem);
iBuildInvRmsLatchSubsystem(subsystem);
iBuildLaneMultiplySubsystem(subsystem);
add_line(subsystem, 'ddrDataBeat/1', 'SquareBeat/1');
add_line(subsystem, 'SquareBeat/1', 'BeatReduce/1');
add_line(subsystem, 'reset/1', 'Controller/1');
add_line(subsystem, 'start/1', 'Controller/2');
add_line(subsystem, 'cfgGammaBeat/1', 'CfgBeatSpec/1');
add_line(subsystem, 'CfgBeatSpec/1', 'Controller/3');
add_line(subsystem, 'cfgGammaValid/1', 'Controller/4');
add_line(subsystem, 'ddrDataBeat/1', 'DdrBeatSpec/1');
add_line(subsystem, 'DdrBeatSpec/1', 'Controller/5');
add_line(subsystem, 'ddrDataValid/1', 'Controller/6');
add_line(subsystem, 'BeatReduce/1', 'BeatAccumulator/1');
add_line(subsystem, 'ddrDataValid/1', 'BeatAccumulator/2');
add_line(subsystem, 'reset/1', 'ClearAccumulator/1');
add_line(subsystem, 'Controller/4', 'CaptureClearDelay/1');
add_line(subsystem, 'CaptureClearDelay/1', 'ClearAccumulator/2');
add_line(subsystem, 'ClearAccumulator/1', 'BeatAccumulator/3');
add_line(subsystem, 'BeatAccumulator/1', 'CurrentSumSpec/1');
add_line(subsystem, 'CurrentSumSpec/1', 'Controller/7');
add_line(subsystem, 'ScalarRsqrt/1', 'InvRmsDelay/1');
add_line(subsystem, 'InvRmsDelay/1', 'InvRmsSpec/1');
add_line(subsystem, 'InvRmsSpec/1', 'Controller/8');
add_line(subsystem, 'Controller/1', 'ScalarRsqrt/1');
add_line(subsystem, 'Controller/2', 'LaneMultiply/1');
add_line(subsystem, 'Controller/3', 'LaneMultiply/2');
add_line(subsystem, 'InvRmsDelay/1', 'InvRmsLatch/1');
add_line(subsystem, 'Controller/4', 'InvRmsLatch/2');
add_line(subsystem, 'InvRmsLatch/1', 'LaneMultiply/3');
add_line(subsystem, 'LaneMultiply/1', 'outBeat/1');
add_line(subsystem, 'Controller/5', 'outValid/1');
add_line(subsystem, 'Controller/6', 'ddrReadAddr/1');
add_line(subsystem, 'Controller/7', 'ddrReadEn/1');
add_line(subsystem, 'Controller/8', 'done/1');
add_line(subsystem, 'Controller/9', 'busy/1');
end

function iBuildSquareBeatSubsystem(subsystem)
target = [subsystem '/SquareBeat'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/ddrDataBeat'], 'Position', [30 38 60 52]);
add_block('simulink/Math Operations/Product', [target '/Square'], 'Inputs', '2', ...
    'Multiplication', 'Element-wise(.*)', 'Position', [100 27 170 63]);
add_block('simulink/Ports & Subsystems/Out1', [target '/squaredBeat'], 'Position', [210 38 240 52]);
add_line(target, 'ddrDataBeat/1', 'Square/1');
add_line(target, 'ddrDataBeat/1', 'Square/2', 'autorouting', 'on');
add_line(target, 'Square/1', 'squaredBeat/1');
end

function iBuildBeatReduceSubsystem(subsystem)
target = [subsystem '/BeatReduce'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/squaredBeat'], 'Position', [30 38 60 52]);
add_block('simulink/Signal Routing/Demux', [target '/Demux'], 'Outputs', '8', 'Position', [90 20 95 100]);
add_block('simulink/Math Operations/Add', [target '/Add01'], 'Inputs', '++', 'Position', [125 18 160 42]);
add_block('simulink/Math Operations/Add', [target '/Add23'], 'Inputs', '++', 'Position', [125 48 160 72]);
add_block('simulink/Math Operations/Add', [target '/Add45'], 'Inputs', '++', 'Position', [125 78 160 102]);
add_block('simulink/Math Operations/Add', [target '/Add67'], 'Inputs', '++', 'Position', [125 108 160 132]);
add_block('simulink/Math Operations/Add', [target '/Add0123'], 'Inputs', '++', 'Position', [190 33 225 57]);
add_block('simulink/Math Operations/Add', [target '/Add4567'], 'Inputs', '++', 'Position', [190 93 225 117]);
add_block('simulink/Math Operations/Add', [target '/AddAll'], 'Inputs', '++', 'Position', [255 63 290 87]);
add_block('simulink/Ports & Subsystems/Out1', [target '/beatSum'], 'Position', [325 68 355 82]);
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
end

function iBuildScalarRsqrtSubsystem(subsystem)
target = [subsystem '/ScalarRsqrt'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/rsqrtOperand'], 'Position', [30 38 60 52]);
add_block('simulink/Math Operations/Math Function', [target '/Sqrt'], 'Operator', 'sqrt', ...
    'Position', [90 28 140 62]);
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/SqrtSingle'], ...
    'OutDataTypeStr', 'single', 'Position', [155 28 205 62]);
add_block('simulink/Sources/Constant', [target '/One'], 'Value', 'single(1)', ...
    'SampleTime', '1', ...
    'Position', [155 78 215 102]);
add_block('simulink/Math Operations/Divide', [target '/Reciprocal'], 'OutDataTypeStr', 'single', ...
    'SaturateOnIntegerOverflow', 'on', 'RndMeth', 'Zero', 'Position', [225 28 280 62]);
add_block('simulink/Ports & Subsystems/Out1', [target '/invRms'], 'Position', [315 38 345 52]);
add_line(target, 'rsqrtOperand/1', 'Sqrt/1');
add_line(target, 'Sqrt/1', 'SqrtSingle/1');
add_line(target, 'One/1', 'Reciprocal/1');
add_line(target, 'SqrtSingle/1', 'Reciprocal/2');
add_line(target, 'Reciprocal/1', 'invRms/1');
end

function iBuildBeatAccumulatorSubsystem(subsystem)
target = [subsystem '/BeatAccumulator'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/beatSum'], 'Position', [30 28 60 42]);
add_block('simulink/Ports & Subsystems/In1', [target '/accumulate'], 'Position', [30 63 60 77], 'Port', '2');
add_block('simulink/Ports & Subsystems/In1', [target '/clear'], 'Position', [30 98 60 112], 'Port', '3');
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/BeatSumSingle'], ...
    'OutDataTypeStr', 'single', 'Position', [80 18 115 42]);
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/AccumulateBool'], ...
    'OutDataTypeStr', 'boolean', 'Position', [80 53 115 77]);
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/AccumulateSingle'], ...
    'OutDataTypeStr', 'single', 'Position', [120 53 155 77]);
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/ClearBool'], ...
    'OutDataTypeStr', 'boolean', 'Position', [80 88 115 112]);
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
add_line(target, 'accumulate/1', 'AccumulateBool/1');
add_line(target, 'accumulate/1', 'AccumulateSingle/1');
add_line(target, 'clear/1', 'ClearBool/1');
add_line(target, 'BankIndexState/1', 'BankIndexAdd/1');
add_line(target, 'BankIndexOne/1', 'BankIndexAdd/2');
add_line(target, 'BankIndexState/1', 'BankIndexIsLast/1');
add_line(target, 'BankIndexZero/1', 'BankIndexWrap/1');
add_line(target, 'BankIndexIsLast/1', 'BankIndexWrap/2');
add_line(target, 'BankIndexAdd/1', 'BankIndexWrap/3');
add_line(target, 'BankIndexWrap/1', 'BankIndexAdvance/1');
add_line(target, 'AccumulateBool/1', 'BankIndexAdvance/2');
add_line(target, 'BankIndexState/1', 'BankIndexAdvance/3');
add_line(target, 'BankIndexZero/1', 'BankIndexClear/1');
add_line(target, 'ClearBool/1', 'BankIndexClear/2');
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
        'OutDataTypeStr', 'single', ...
        'Position', [220 20 + bankIndex*35 255 44 + bankIndex*35]);
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
    add_line(target, 'AccumulateBool/1', [updateName '/2']);
    add_line(target, 'BeatSumSingle/1', [gateName '/1']);
    add_line(target, 'AccumulateSingle/1', [gateName '/2']);
    add_line(target, [selectName '/1'], [gateName '/3']);
    add_line(target, [delayName '/1'], [addName '/1']);
    add_line(target, [gateName '/1'], [addName '/2']);
    add_line(target, [addName '/1'], [holdName '/1']);
    add_line(target, [updateName '/1'], [holdName '/2']);
    add_line(target, [stateName '/1'], [holdName '/3']);
    add_line(target, 'ZeroSingle/1', [clearName '/1']);
    add_line(target, 'ClearBool/1', [clearName '/2']);
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
end

function iBuildInvRmsLatchSubsystem(subsystem)
target = [subsystem '/InvRmsLatch'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/invRmsIn'], 'Position', [30 33 60 47]);
add_block('simulink/Ports & Subsystems/In1', [target '/capture'], 'Position', [30 83 60 97], 'Port', '2');
add_block('simulink/Discrete/Unit Delay', [target '/StoredInvRms'], ...
    'InitialCondition', 'single(0)', 'Position', [190 28 235 52]);
add_block('simulink/Signal Routing/Switch', [target '/SelectNew'], ...
    'Criteria', 'u2 ~= 0', 'Threshold', '0.5', 'Position', [120 32 155 98]);
add_block('simulink/Ports & Subsystems/Out1', [target '/invRmsLatched'], 'Position', [280 38 310 52]);
add_line(target, 'invRmsIn/1', 'SelectNew/1');
add_line(target, 'capture/1', 'SelectNew/2');
add_line(target, 'StoredInvRms/1', 'SelectNew/3');
add_line(target, 'SelectNew/1', 'StoredInvRms/1');
add_line(target, 'StoredInvRms/1', 'invRmsLatched/1');
end

function iBuildLaneMultiplySubsystem(subsystem)
target = [subsystem '/LaneMultiply'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/xBeat'], 'Position', [30 43 60 57]);
add_block('simulink/Ports & Subsystems/In1', [target '/gBeat'], 'Position', [30 93 60 107], 'Port', '2');
add_block('simulink/Ports & Subsystems/In1', [target '/invRms'], 'Position', [30 143 60 157], 'Port', '3');
add_block('simulink/Signal Attributes/Signal Specification', [target '/InvRmsScalar'], ...
    'Dimensions', '1', 'Position', [85 138 140 162]);
add_block('simulink/Math Operations/Gain', [target '/ScaleGamma'], 'Gain', 'single(sqrt(1536))', ...
    'Position', [85 88 145 112]);
add_block('simulink/Math Operations/Product', [target '/WeightMultiply'], 'Inputs', '2', ...
    'Multiplication', 'Element-wise(.*)', 'Position', [160 58 230 92]);
add_block('simulink/Math Operations/Product', [target '/ScaleMultiply'], 'Inputs', '2', ...
    'Multiplication', 'Element-wise(.*)', 'Position', [260 78 335 122]);
add_block('simulink/Ports & Subsystems/Out1', [target '/yBeat'], 'Position', [370 93 400 107]);
add_line(target, 'xBeat/1', 'WeightMultiply/1');
add_line(target, 'gBeat/1', 'ScaleGamma/1');
add_line(target, 'ScaleGamma/1', 'WeightMultiply/2');
add_line(target, 'WeightMultiply/1', 'ScaleMultiply/1');
add_line(target, 'invRms/1', 'InvRmsScalar/1');
add_line(target, 'InvRmsScalar/1', 'ScaleMultiply/2');
add_line(target, 'ScaleMultiply/1', 'yBeat/1');
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
scriptText = strjoin({
    'function [rsqrtOperand, xBeatOut, gBeatOut, captureInvRms, outValid, ddrReadAddr, ddrReadEn, done, busy] = Controller(reset,start,cfgGammaBeat,cfgGammaValid,ddrDataBeat,ddrDataValid,currentSumIn,invRmsIn)'
    '%#codegen'
    'persistent state gammaMem tokenMem;'
    'persistent gammaWriteBeat gammaLoaded tokenIndex;'
    'persistent requestBeatIndex receiveBeatIndex outputBeatIndex sumFlushCount;'
    'beatsPerToken = uint16(192); epsilonScaled = single(1536e-6);'
    'if isempty(state), state=uint8(0); gammaMem=zeros(1,1536,''single''); tokenMem=zeros(1,1536,''single''); gammaWriteBeat=uint16(0); gammaLoaded=false; tokenIndex=uint8(0); requestBeatIndex=uint16(0); receiveBeatIndex=uint16(0); outputBeatIndex=uint16(0); sumFlushCount=uint8(0); end'
    'currentSum = single(currentSumIn); rsqrtOperand = currentSum + epsilonScaled; xBeatOut = zeros(1,8,''single''); gBeatOut = zeros(1,8,''single''); captureInvRms = false; outValid = false; ddrReadAddr = uint16(0); ddrReadEn = false; done = false; busy = state ~= uint8(0);'
    'if reset, state=uint8(0); gammaMem=zeros(1,1536,''single''); tokenMem=zeros(1,1536,''single''); gammaWriteBeat=uint16(0); gammaLoaded=false; tokenIndex=uint8(0); requestBeatIndex=uint16(0); receiveBeatIndex=uint16(0); outputBeatIndex=uint16(0); sumFlushCount=uint8(0); else'
    'if cfgGammaValid, startIdx = gammaWriteBeat * uint16(8); gammaMem(startIdx+uint16(1))=cfgGammaBeat(1); gammaMem(startIdx+uint16(2))=cfgGammaBeat(2); gammaMem(startIdx+uint16(3))=cfgGammaBeat(3); gammaMem(startIdx+uint16(4))=cfgGammaBeat(4); gammaMem(startIdx+uint16(5))=cfgGammaBeat(5); gammaMem(startIdx+uint16(6))=cfgGammaBeat(6); gammaMem(startIdx+uint16(7))=cfgGammaBeat(7); gammaMem(startIdx+uint16(8))=cfgGammaBeat(8); if gammaWriteBeat == beatsPerToken-1, gammaWriteBeat=uint16(0); gammaLoaded=true; else, gammaWriteBeat=gammaWriteBeat+uint16(1); end; end'
    'switch state'
    'case uint8(0)'
    '    if start && gammaLoaded, state=uint8(1); tokenIndex=uint8(0); requestBeatIndex=uint16(0); receiveBeatIndex=uint16(0); outputBeatIndex=uint16(0); sumFlushCount=uint8(0); busy=true; end'
    'case uint8(1)'
    '    busy=true; if requestBeatIndex < beatsPerToken, ddrReadEn=true; ddrReadAddr=uint16(uint16(tokenIndex)*beatsPerToken + requestBeatIndex); requestBeatIndex=requestBeatIndex+uint16(1); end'
    '    if ddrDataValid, startIdx = receiveBeatIndex * uint16(8); tokenMem(startIdx+uint16(1))=ddrDataBeat(1); tokenMem(startIdx+uint16(2))=ddrDataBeat(2); tokenMem(startIdx+uint16(3))=ddrDataBeat(3); tokenMem(startIdx+uint16(4))=ddrDataBeat(4); tokenMem(startIdx+uint16(5))=ddrDataBeat(5); tokenMem(startIdx+uint16(6))=ddrDataBeat(6); tokenMem(startIdx+uint16(7))=ddrDataBeat(7); tokenMem(startIdx+uint16(8))=ddrDataBeat(8); if receiveBeatIndex == beatsPerToken-1, receiveBeatIndex=uint16(0); requestBeatIndex=uint16(0); sumFlushCount=uint8(22); state=uint8(2); else, receiveBeatIndex=receiveBeatIndex+uint16(1); end; end'
    'case uint8(2)'
    '    busy=true; if sumFlushCount == uint8(0), state=uint8(3); else, sumFlushCount = sumFlushCount - uint8(1); end'
    'case uint8(3)'
    '    busy=true; captureInvRms=true; outputBeatIndex=uint16(0); state=uint8(4);'
    'otherwise'
    '    busy=true; outValid=true; startIdx=outputBeatIndex * uint16(8); xBeatOut(1)=tokenMem(startIdx+uint16(1)); xBeatOut(2)=tokenMem(startIdx+uint16(2)); xBeatOut(3)=tokenMem(startIdx+uint16(3)); xBeatOut(4)=tokenMem(startIdx+uint16(4)); xBeatOut(5)=tokenMem(startIdx+uint16(5)); xBeatOut(6)=tokenMem(startIdx+uint16(6)); xBeatOut(7)=tokenMem(startIdx+uint16(7)); xBeatOut(8)=tokenMem(startIdx+uint16(8)); gBeatOut(1)=gammaMem(startIdx+uint16(1)); gBeatOut(2)=gammaMem(startIdx+uint16(2)); gBeatOut(3)=gammaMem(startIdx+uint16(3)); gBeatOut(4)=gammaMem(startIdx+uint16(4)); gBeatOut(5)=gammaMem(startIdx+uint16(5)); gBeatOut(6)=gammaMem(startIdx+uint16(6)); gBeatOut(7)=gammaMem(startIdx+uint16(7)); gBeatOut(8)=gammaMem(startIdx+uint16(8)); if outputBeatIndex == beatsPerToken-1, if tokenIndex == uint8(63), done=true; state=uint8(0); tokenIndex=uint8(0); outputBeatIndex=uint16(0); busy=false; else, tokenIndex=tokenIndex+uint8(1); outputBeatIndex=uint16(0); requestBeatIndex=uint16(0); receiveBeatIndex=uint16(0); state=uint8(1); end; else, outputBeatIndex=outputBeatIndex+uint16(1); end'
    'end'
    'end'
    'end'}, newline);
end
