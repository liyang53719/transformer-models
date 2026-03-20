function modelPath = buildRmsNormalizationModel()
% buildRmsNormalizationModel   Build the streaming RMSNorm Simulink DUT model.

thisDir = fileparts(mfilename('fullpath'));
modelName = 'rmsNormalization';
modelPath = fullfile(thisDir, [modelName '.slx']);
rtlVersion = 'v3';

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
assignin('base', 'startSeq', timeseries([false; true; false], [0; 1; 2]));
assignin('base', 'cfgGammaValidSeq', timeseries([false; false], [0; 1]));
assignin('base', 'cfgGammaBeatSeq', timeseries(zeros(2, 8, 'single'), [0; 1]));
assignin('base', 'ddrDataValidSeq', timeseries([false; false], [0; 1]));
assignin('base', 'ddrDataBeatSeq', timeseries(zeros(2, 8, 'single'), [0; 1]));

iAddFromWorkspace(modelName, 'StartSrc', 'startSeq', [30 60 120 90]);
iAddFromWorkspace(modelName, 'CfgBeatSrc', 'cfgGammaBeatSeq', [30 110 120 140]);
iAddFromWorkspace(modelName, 'CfgValidSrc', 'cfgGammaValidSeq', [30 160 120 190]);
iAddFromWorkspace(modelName, 'DdrBeatSrc', 'ddrDataBeatSeq', [30 210 120 240]);
iAddFromWorkspace(modelName, 'DdrValidSrc', 'ddrDataValidSeq', [30 260 120 290]);

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

add_line(modelName, 'StartSrc/1', 'DUT/1');
add_line(modelName, 'CfgBeatSrc/1', 'DUT/2');
add_line(modelName, 'CfgValidSrc/1', 'DUT/3');
add_line(modelName, 'DdrBeatSrc/1', 'DUT/4');
add_line(modelName, 'DdrValidSrc/1', 'DUT/5');
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
iBuildStreamingDutSubsystem(subsystem, true);
end

function iBuildStreamingDutSubsystem(subsystem, enableLogs)
iDeleteSubsystemContents(subsystem);

add_block('simulink/Ports & Subsystems/In1', [subsystem '/start'], 'Position', [25 63 55 77], 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgGammaBeat'], 'Position', [25 98 55 112], 'Port', '2', 'SampleTime', '1', 'OutDataTypeStr', 'single');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgGammaValid'], 'Position', [25 133 55 147], 'Port', '3', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/ddrDataBeat'], 'Position', [25 168 55 182], 'Port', '4', 'SampleTime', '1', 'OutDataTypeStr', 'single');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/ddrDataValid'], 'Position', [25 203 55 217], 'Port', '5', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');

add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/SquareBeat'], 'Position', [105 150 220 220]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/BeatReduce'], 'Position', [245 150 360 220]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/TokenSram'], 'Position', [255 255 435 415]);
add_block('simulink/Discrete/Delay', [subsystem '/WriteBankAccumDelay'], ...
    'DelayLength', '41', 'InitialCondition', 'false', 'Position', [365 300 430 324]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [subsystem '/WriteBankNot'], ...
    'Operator', 'NOT', 'Position', [385 238 420 262]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [subsystem '/BeatSumValidBank0'], ...
    'Operator', 'AND', 'Inputs', '2', 'Position', [445 118 480 142]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [subsystem '/BeatSumValidBank1'], ...
    'Operator', 'AND', 'Inputs', '2', 'Position', [445 258 480 282]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/BeatAccumulator0'], 'Position', [510 70 720 185]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/BeatAccumulator1'], 'Position', [510 210 720 325]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/ScalarRsqrt0'], 'Position', [750 40 905 110]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/ScalarRsqrt1'], 'Position', [750 180 905 250]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [subsystem '/CaptureBankNot'], ...
    'Operator', 'NOT', 'Position', [760 330 795 354]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [subsystem '/CaptureInvRmsBank0'], ...
    'Operator', 'AND', 'Inputs', '2', 'Position', [820 320 855 344]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [subsystem '/CaptureInvRmsBank1'], ...
    'Operator', 'AND', 'Inputs', '2', 'Position', [820 355 855 379]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/InvRmsLatch0'], 'Position', [930 35 1095 125]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/InvRmsLatch1'], 'Position', [930 175 1095 265]);
add_block('simulink/Discrete/Unit Delay', [subsystem '/ClearAccumulatorFeedback0'], ...
    'InitialCondition', 'false', 'Position', [1110 58 1155 82]);
add_block('simulink/Discrete/Unit Delay', [subsystem '/ClearAccumulatorFeedback1'], ...
    'InitialCondition', 'false', 'Position', [1110 198 1155 222]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [subsystem '/ReadBankNot'], ...
    'Operator', 'NOT', 'Position', [1120 300 1155 324]);
add_block('simulink/Logic and Bit Operations/Relational Operator', [subsystem '/ReadAddrIsZero'], ...
    'Operator', '==', 'Position', [1120 430 1155 454]);
add_block('simulink/Sources/Constant', [subsystem '/ReadAddrZeroConst'], ...
    'Value', 'uint16(0)', 'OutDataTypeStr', 'uint16', 'Position', [1060 430 1100 454]);
add_block('simulink/Signal Routing/Switch', [subsystem '/SelectedInvRms'], ...
    'Criteria', 'u2 ~= 0', 'Threshold', '0.5', 'Position', [1180 292 1215 346]);
add_block('simulink/Signal Routing/Switch', [subsystem '/SelectedInvRmsValid'], ...
    'Criteria', 'u2 ~= 0', 'Threshold', '0.5', 'Position', [1180 352 1215 406]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [subsystem '/OutputTokenStart'], ...
    'Operator', 'AND', 'Inputs', '2', 'Position', [1180 430 1215 454]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/OutputInvRmsLatch'], 'Position', [1245 300 1410 390]);
add_block('simulink/Ports & Subsystems/Subsystem', [subsystem '/LaneMultiply'], 'Position', [870 225 1055 350]);
add_block('simulink/User-Defined Functions/MATLAB Function', [subsystem '/Controller'], 'Position', [105 20 255 330]);
set_param([subsystem '/Controller'], 'SystemSampleTime', '1');
open_system([subsystem '/Controller']);
rt = sfroot;
chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [subsystem '/Controller']);
chart.Script = iBuildControllerScript();

add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/CfgBeatSpec'], ...
    'Dimensions', '[1 8]', 'OutDataTypeStr', 'single', 'Position', [80 90 130 120]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/WriteAddrSpec'], ...
    'Dimensions', '1', 'OutDataTypeStr', 'uint16', 'Position', [285 300 340 324]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/WriteBankSpec'], ...
    'Dimensions', '1', 'OutDataTypeStr', 'boolean', 'Position', [285 330 340 354]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/ReadAddrSpec'], ...
    'Dimensions', '1', 'OutDataTypeStr', 'uint16', 'Position', [285 360 340 384]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/ReadBankSpec'], ...
    'Dimensions', '1', 'OutDataTypeStr', 'boolean', 'Position', [285 390 340 414]);
add_block('simulink/Signal Attributes/Signal Specification', [subsystem '/CaptureBankSpec'], ...
    'Dimensions', '1', 'OutDataTypeStr', 'boolean', 'Position', [285 420 340 444]);

add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outBeat'], 'Position', [1090 178 1120 192]);
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/outValid'], 'Position', [1090 218 1120 232], 'Port', '2');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/ddrReadAddr'], 'Position', [1090 58 1120 72], 'Port', '3');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/ddrReadEn'], 'Position', [1090 93 1120 107], 'Port', '4');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/done'], 'Position', [1090 128 1120 142], 'Port', '5');
add_block('simulink/Ports & Subsystems/Out1', [subsystem '/busy'], 'Position', [1090 163 1120 177], 'Port', '6');

set_param([subsystem '/SquareBeat'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/BeatReduce'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/BeatAccumulator0'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/BeatAccumulator1'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/TokenSram'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/ScalarRsqrt0'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/ScalarRsqrt1'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/InvRmsLatch0'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/InvRmsLatch1'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/OutputInvRmsLatch'], 'TreatAsAtomicUnit', 'on');
set_param([subsystem '/LaneMultiply'], 'TreatAsAtomicUnit', 'on');
hdlset_param([subsystem '/SquareBeat'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/BeatReduce'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/BeatAccumulator0'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/BeatAccumulator1'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/TokenSram'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/ScalarRsqrt0'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/ScalarRsqrt1'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/InvRmsLatch0'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/InvRmsLatch1'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/OutputInvRmsLatch'], 'BalanceDelays', 'on');
hdlset_param([subsystem '/LaneMultiply'], 'BalanceDelays', 'on');

iBuildSquareBeatSubsystem(subsystem);
iBuildBeatReduceSubsystem(subsystem);
iBuildBeatAccumulatorSubsystem(subsystem, 'BeatAccumulator0');
iBuildBeatAccumulatorSubsystem(subsystem, 'BeatAccumulator1');
iBuildTokenSramSubsystem(subsystem);
iBuildScalarRsqrtSubsystem(subsystem, 'ScalarRsqrt0');
iBuildScalarRsqrtSubsystem(subsystem, 'ScalarRsqrt1');
iBuildInvRmsLatchSubsystem(subsystem, 'InvRmsLatch0');
iBuildInvRmsLatchSubsystem(subsystem, 'InvRmsLatch1');
iBuildInvRmsLatchSubsystem(subsystem, 'OutputInvRmsLatch');
iBuildLaneMultiplySubsystem(subsystem);

add_line(subsystem, 'ddrDataBeat/1', 'TokenSram/1');
add_line(subsystem, 'ddrDataBeat/1', 'SquareBeat/1');
add_line(subsystem, 'ddrDataValid/1', 'SquareBeat/2');
add_line(subsystem, 'SquareBeat/1', 'BeatReduce/1');
add_line(subsystem, 'SquareBeat/2', 'BeatReduce/2');
add_line(subsystem, 'Controller/5', 'WriteAddrSpec/1');
add_line(subsystem, 'WriteAddrSpec/1', 'TokenSram/2');
add_line(subsystem, 'Controller/6', 'WriteBankSpec/1');
add_line(subsystem, 'WriteBankSpec/1', 'TokenSram/3');
add_line(subsystem, 'WriteBankSpec/1', 'WriteBankAccumDelay/1');
add_line(subsystem, 'Controller/7', 'TokenSram/4');
add_line(subsystem, 'BeatReduce/1', 'BeatAccumulator0/1');
add_line(subsystem, 'BeatReduce/1', 'BeatAccumulator1/1');
add_line(subsystem, 'BeatReduce/2', 'BeatSumValidBank0/1');
add_line(subsystem, 'BeatReduce/2', 'BeatSumValidBank1/1');
add_line(subsystem, 'WriteBankAccumDelay/1', 'WriteBankNot/1');
add_line(subsystem, 'WriteBankNot/1', 'BeatSumValidBank0/2');
add_line(subsystem, 'WriteBankAccumDelay/1', 'BeatSumValidBank1/2');
add_line(subsystem, 'BeatSumValidBank0/1', 'BeatAccumulator0/2');
add_line(subsystem, 'BeatSumValidBank1/1', 'BeatAccumulator1/2');
add_line(subsystem, 'ClearAccumulatorFeedback0/1', 'BeatAccumulator0/3');
add_line(subsystem, 'ClearAccumulatorFeedback1/1', 'BeatAccumulator1/3');
add_line(subsystem, 'BeatAccumulator0/1', 'ScalarRsqrt0/1');
add_line(subsystem, 'BeatAccumulator0/2', 'ScalarRsqrt0/2');
add_line(subsystem, 'BeatAccumulator1/1', 'ScalarRsqrt1/1');
add_line(subsystem, 'BeatAccumulator1/2', 'ScalarRsqrt1/2');
add_line(subsystem, 'Controller/4', 'CaptureBankSpec/1');
add_line(subsystem, 'CaptureBankSpec/1', 'CaptureBankNot/1');
add_line(subsystem, 'Controller/3', 'CaptureInvRmsBank0/1');
add_line(subsystem, 'Controller/3', 'CaptureInvRmsBank1/1');
add_line(subsystem, 'CaptureBankNot/1', 'CaptureInvRmsBank0/2');
add_line(subsystem, 'CaptureBankSpec/1', 'CaptureInvRmsBank1/2');
add_line(subsystem, 'ScalarRsqrt0/1', 'InvRmsLatch0/1');
add_line(subsystem, 'ScalarRsqrt0/2', 'InvRmsLatch0/2');
add_line(subsystem, 'CaptureInvRmsBank0/1', 'InvRmsLatch0/3');
add_line(subsystem, 'ScalarRsqrt1/1', 'InvRmsLatch1/1');
add_line(subsystem, 'ScalarRsqrt1/2', 'InvRmsLatch1/2');
add_line(subsystem, 'CaptureInvRmsBank1/1', 'InvRmsLatch1/3');
add_line(subsystem, 'InvRmsLatch0/3', 'ClearAccumulatorFeedback0/1');
add_line(subsystem, 'InvRmsLatch1/3', 'ClearAccumulatorFeedback1/1');
add_line(subsystem, 'Controller/8', 'ReadAddrSpec/1');
add_line(subsystem, 'ReadAddrSpec/1', 'TokenSram/5');
add_line(subsystem, 'Controller/9', 'ReadBankSpec/1');
add_line(subsystem, 'ReadBankSpec/1', 'TokenSram/6');
add_line(subsystem, 'ReadBankSpec/1', 'ReadBankNot/1');
add_line(subsystem, 'Controller/10', 'TokenSram/7');
add_line(subsystem, 'InvRmsLatch1/1', 'SelectedInvRms/1');
add_line(subsystem, 'ReadBankSpec/1', 'SelectedInvRms/2');
add_line(subsystem, 'InvRmsLatch0/1', 'SelectedInvRms/3');
add_line(subsystem, 'InvRmsLatch1/2', 'SelectedInvRmsValid/1');
add_line(subsystem, 'ReadBankSpec/1', 'SelectedInvRmsValid/2');
add_line(subsystem, 'InvRmsLatch0/2', 'SelectedInvRmsValid/3');
add_line(subsystem, 'ReadAddrSpec/1', 'ReadAddrIsZero/1');
add_line(subsystem, 'ReadAddrZeroConst/1', 'ReadAddrIsZero/2');
add_line(subsystem, 'Controller/10', 'OutputTokenStart/1');
add_line(subsystem, 'ReadAddrIsZero/1', 'OutputTokenStart/2');
add_line(subsystem, 'SelectedInvRms/1', 'OutputInvRmsLatch/1');
add_line(subsystem, 'SelectedInvRmsValid/1', 'OutputInvRmsLatch/2');
add_line(subsystem, 'OutputTokenStart/1', 'OutputInvRmsLatch/3');
add_line(subsystem, 'OutputInvRmsLatch/1', 'LaneMultiply/5');
add_line(subsystem, 'OutputInvRmsLatch/2', 'LaneMultiply/6');

add_line(subsystem, 'start/1', 'Controller/1');
add_line(subsystem, 'cfgGammaBeat/1', 'CfgBeatSpec/1');
add_line(subsystem, 'CfgBeatSpec/1', 'Controller/2');
add_line(subsystem, 'cfgGammaValid/1', 'Controller/3');
add_line(subsystem, 'ddrDataValid/1', 'Controller/4');
add_line(subsystem, 'ClearAccumulatorFeedback0/1', 'Controller/5');
add_line(subsystem, 'ClearAccumulatorFeedback1/1', 'Controller/6');
add_line(subsystem, 'TokenSram/1', 'LaneMultiply/1');
add_line(subsystem, 'TokenSram/2', 'LaneMultiply/2');
add_line(subsystem, 'Controller/1', 'LaneMultiply/3');
add_line(subsystem, 'Controller/2', 'LaneMultiply/4');
add_line(subsystem, 'LaneMultiply/1', 'outBeat/1');
add_line(subsystem, 'LaneMultiply/2', 'outValid/1');
add_line(subsystem, 'Controller/11', 'ddrReadAddr/1');
add_line(subsystem, 'Controller/12', 'ddrReadEn/1');
add_line(subsystem, 'Controller/13', 'done/1');
add_line(subsystem, 'Controller/14', 'busy/1');

if enableLogs
    iAddStreamingDebugLogs(subsystem);
end
end

function iBuildPackedDutSubsystem(modelName)
subsystem = [modelName '/DUTPacked'];
iDeleteSubsystemContents(subsystem);

add_block('simulink/Ports & Subsystems/In1', [subsystem '/start'], 'Position', [25 63 55 77], 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgGammaBeat'], 'Position', [25 98 55 112], 'Port', '2', 'SampleTime', '1', 'OutDataTypeStr', 'fixdt(0,256,0)');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/cfgGammaValid'], 'Position', [25 133 55 147], 'Port', '3', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/ddrDataBeat'], 'Position', [25 168 55 182], 'Port', '4', 'SampleTime', '1', 'OutDataTypeStr', 'fixdt(0,256,0)');
add_block('simulink/Ports & Subsystems/In1', [subsystem '/ddrDataValid'], 'Position', [25 203 55 217], 'Port', '5', 'SampleTime', '1', 'OutDataTypeStr', 'boolean');

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
iBuildStreamingDutSubsystem([subsystem '/CoreDUT'], false);
iBuildPackedBeatPackSubsystem([subsystem '/OutBeatPack']);

add_line(subsystem, 'start/1', 'CoreDUT/1');
add_line(subsystem, 'cfgGammaBeat/1', 'CfgBeatUnpack/1');
add_line(subsystem, 'CfgBeatUnpack/1', 'CoreDUT/2');
add_line(subsystem, 'cfgGammaValid/1', 'CoreDUT/3');
add_line(subsystem, 'ddrDataBeat/1', 'DdrBeatUnpack/1');
add_line(subsystem, 'DdrBeatUnpack/1', 'CoreDUT/4');
add_line(subsystem, 'ddrDataValid/1', 'CoreDUT/5');
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

function iBuildTokenSramSubsystem(subsystem)
target = [subsystem '/TokenSram'];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/writeBeat'], 'Position', [30 28 60 42]);
add_block('simulink/Ports & Subsystems/In1', [target '/writeAddr'], 'Position', [30 63 60 77], 'Port', '2', 'OutDataTypeStr', 'uint16');
add_block('simulink/Ports & Subsystems/In1', [target '/writeBank'], 'Position', [30 98 60 112], 'Port', '3', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [target '/writeValid'], 'Position', [30 133 60 147], 'Port', '4', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [target '/readAddr'], 'Position', [30 168 60 182], 'Port', '5', 'OutDataTypeStr', 'uint16');
add_block('simulink/Ports & Subsystems/In1', [target '/readBank'], 'Position', [30 203 60 217], 'Port', '6', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [target '/readReq'], 'Position', [30 238 60 252], 'Port', '7', 'OutDataTypeStr', 'boolean');
add_block('simulink/User-Defined Functions/MATLAB Function', [target '/SramCore'], ...
    'Position', [100 60 315 225]);
set_param([target '/SramCore'], 'SystemSampleTime', '1');
hdlset_param([target '/SramCore'], 'architecture', 'MATLAB Function');
open_system([target '/SramCore']);
rt = sfroot;
chart = rt.find('-isa', 'Stateflow.EMChart', 'Path', [target '/SramCore']);
chart.Script = iBuildTokenSramScript();
add_block('simulink/Ports & Subsystems/Out1', [target '/readBeat'], 'Position', [355 103 385 117]);
add_block('simulink/Ports & Subsystems/Out1', [target '/readValid'], 'Position', [355 173 385 187], 'Port', '2');
add_line(target, 'writeBeat/1', 'SramCore/1');
add_line(target, 'writeAddr/1', 'SramCore/2');
add_line(target, 'writeBank/1', 'SramCore/3');
add_line(target, 'writeValid/1', 'SramCore/4');
add_line(target, 'readAddr/1', 'SramCore/5');
add_line(target, 'readBank/1', 'SramCore/6');
add_line(target, 'readReq/1', 'SramCore/7');
add_line(target, 'SramCore/1', 'readBeat/1');
add_line(target, 'SramCore/2', 'readValid/1');
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

function iBuildBeatAccumulatorSubsystem(subsystem, blockName)
if nargin < 2
    blockName = 'BeatAccumulator';
end
target = [subsystem '/' blockName];
iDeleteSubsystemContents(target);
numBanks = 32;
feedbackDelayLength = numBanks - 1;

add_block('simulink/Ports & Subsystems/In1', [target '/beatSum'], 'Position', [30 28 60 42]);
add_block('simulink/Ports & Subsystems/In1', [target '/beatSumValid'], 'Position', [30 63 60 77], 'Port', '2', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [target '/clearAccumulator'], 'Position', [30 98 60 112], 'Port', '3', 'OutDataTypeStr', 'boolean');
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/BeatSumSingle'], ...
    'OutDataTypeStr', 'single', 'Position', [80 18 115 42]);
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/BeatSumValidBool'], ...
    'OutDataTypeStr', 'boolean', 'Position', [80 53 115 77]);
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/BeatSumValidSingle'], ...
    'OutDataTypeStr', 'single', 'Position', [120 53 155 77]);
add_block('simulink/Discrete/Unit Delay', [target '/ClearAccumulatorDelay'], ...
    'InitialCondition', 'false', 'Position', [80 98 115 122]);
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
    'relop', '==', 'const', sprintf('%d', numBanks - 1), 'Position', [205 198 255 222]);
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
add_line(target, 'ClearAccumulatorDelay/1', 'BankIndexClear/2');
add_line(target, 'BankIndexAdvance/1', 'BankIndexClear/3');
add_line(target, 'BankIndexClear/1', 'BankIndexState/1');

bankOutputs = cell(1, numBanks);
for bankIndex = 0:numBanks-1
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
        'DelayLength', sprintf('%d', feedbackDelayLength), 'InitialCondition', 'single(0)', ...
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
    add_line(target, 'ClearAccumulatorDelay/1', [clearName '/2']);
    add_line(target, [holdName '/1'], [clearName '/3']);
    add_line(target, [clearName '/1'], [stateName '/1']);
    add_line(target, [stateName '/1'], [delayName '/1']);
    bankOutputs{bankIndex + 1} = [stateName '/1'];
end

levelSignals = bankOutputs;
levelIndex = 0;
while numel(levelSignals) > 1
    nextSignals = {};
    for addIndex = 1:2:numel(levelSignals)
        nextIndex = floor((addIndex - 1) / 2) + 1;
        if addIndex == numel(levelSignals)
            nextSignals{nextIndex} = levelSignals{addIndex}; %#ok<AGROW>
        else
            addName = sprintf('SumL%d_%02d', levelIndex, nextIndex - 1);
            add_block('simulink/Math Operations/Add', [target '/' addName], ...
                'Inputs', '++', 'OutDataTypeStr', 'single', ...
                'Position', [720 + levelIndex*75 22 + (nextIndex-1)*70 760 + levelIndex*75 46 + (nextIndex-1)*70]);
            add_line(target, levelSignals{addIndex}, [addName '/1']);
            add_line(target, levelSignals{addIndex + 1}, [addName '/2']);
            nextSignals{nextIndex} = [addName '/1']; %#ok<AGROW>
        end
    end
    levelSignals = nextSignals;
    levelIndex = levelIndex + 1;
end

outX = 720 + levelIndex*75 + 80;
add_block('simulink/Ports & Subsystems/Out1', [target '/currentSum'], 'Position', [outX 217 outX+30 231]);
add_block('simulink/Discrete/Unit Delay', [target '/CurrentSumValidAlign'], ...
    'InitialCondition', 'false', 'Position', [outX-70 252 outX-25 276]);
add_block('simulink/Ports & Subsystems/Out1', [target '/currentSumValid'], 'Position', [outX 257 outX+30 271], 'Port', '2');
add_line(target, levelSignals{1}, 'currentSum/1');
add_line(target, 'BeatSumValidBool/1', 'CurrentSumValidAlign/1');
add_line(target, 'CurrentSumValidAlign/1', 'currentSumValid/1');
end

function iBuildScalarRsqrtSubsystem(subsystem, blockName)
if nargin < 2
    blockName = 'ScalarRsqrt';
end
target = [subsystem '/' blockName];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/currentSum'], 'Position', [30 38 60 52]);
add_block('simulink/Ports & Subsystems/In1', [target '/currentSumValid'], 'Position', [30 83 60 97], 'Port', '2', 'OutDataTypeStr', 'boolean');
add_block('simulink/Discrete/Unit Delay', [target '/CurrentSumPipe'], ...
    'InitialCondition', 'single(0)', 'Position', [80 28 125 52]);
add_block('simulink/Discrete/Unit Delay', [target '/CurrentSumValidPipe'], ...
    'InitialCondition', 'false', 'Position', [80 83 125 107]);
add_block('simulink/Sources/Constant', [target '/Epsilon'], 'Value', 'single(1536e-6)', ...
    'SampleTime', '1', 'Position', [90 78 160 102]);
add_block('simulink/Math Operations/Add', [target '/RsqrtOperandAdd'], 'Inputs', '++', ...
    'OutDataTypeStr', 'single', 'Position', [90 28 150 62]);
add_block('simulink/Math Operations/Reciprocal Sqrt', [target '/ReciprocalSqrt'], ...
    'Position', [180 28 240 62]);
add_block('simulink/Signal Attributes/Data Type Conversion', [target '/ReciprocalSqrtSingle'], ...
    'OutDataTypeStr', 'single', 'Position', [260 28 315 62]);
add_block('simulink/Discrete/Unit Delay', [target '/InvRmsReg'], ...
    'InitialCondition', 'single(0)', 'Position', [340 28 385 52]);
add_block('simulink/Discrete/Unit Delay', [target '/InvRmsValidReg'], ...
    'InitialCondition', 'false', 'Position', [340 83 385 107]);
add_block('simulink/Discrete/Unit Delay', [target '/InvRmsValidAlign'], ...
    'InitialCondition', 'false', 'Position', [390 83 435 107]);
add_block('simulink/Ports & Subsystems/Out1', [target '/invRms'], 'Position', [420 38 450 52]);
add_block('simulink/Ports & Subsystems/Out1', [target '/invRmsValid'], 'Position', [420 88 450 102], 'Port', '2');
add_line(target, 'currentSum/1', 'CurrentSumPipe/1');
add_line(target, 'CurrentSumPipe/1', 'RsqrtOperandAdd/1');
add_line(target, 'Epsilon/1', 'RsqrtOperandAdd/2');
add_line(target, 'RsqrtOperandAdd/1', 'ReciprocalSqrt/1');
add_line(target, 'ReciprocalSqrt/1', 'ReciprocalSqrtSingle/1');
add_line(target, 'ReciprocalSqrtSingle/1', 'InvRmsReg/1');
add_line(target, 'InvRmsReg/1', 'invRms/1');
add_line(target, 'currentSumValid/1', 'CurrentSumValidPipe/1');
add_line(target, 'CurrentSumValidPipe/1', 'InvRmsValidReg/1');
add_line(target, 'InvRmsValidReg/1', 'InvRmsValidAlign/1');
add_line(target, 'InvRmsValidAlign/1', 'invRmsValid/1');
end

function iBuildInvRmsLatchSubsystem(subsystem, blockName)
if nargin < 2
    blockName = 'InvRmsLatch';
end
target = [subsystem '/' blockName];
iDeleteSubsystemContents(target);
add_block('simulink/Ports & Subsystems/In1', [target '/invRmsIn'], 'Position', [30 33 60 47]);
add_block('simulink/Ports & Subsystems/In1', [target '/invRmsInValid'], 'Position', [30 68 60 82], 'Port', '2', 'OutDataTypeStr', 'boolean');
add_block('simulink/Ports & Subsystems/In1', [target '/capture'], 'Position', [30 103 60 117], 'Port', '3', 'OutDataTypeStr', 'boolean');
add_block('simulink/Logic and Bit Operations/Logical Operator', [target '/CaptureAccepted'], ...
    'Operator', 'AND', 'Position', [85 83 120 107]);
add_block('simulink/Discrete/Unit Delay', [target '/StoredInvRms'], ...
    'InitialCondition', 'single(0)', 'Position', [310 28 355 52]);
add_block('simulink/Discrete/Unit Delay', [target '/CaptureAcceptedDelay'], ...
    'InitialCondition', 'false', 'Position', [310 88 355 112]);
add_block('simulink/Discrete/Unit Delay', [target '/LatchedValidState'], ...
    'InitialCondition', 'false', 'Position', [310 133 355 157]);
add_block('simulink/Signal Routing/Switch', [target '/SelectNew'], ...
    'Criteria', 'u2 ~= 0', 'Threshold', '0.5', 'Position', [145 22 180 88]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [target '/HoldValid'], ...
    'Operator', 'OR', 'Position', [225 118 260 142]);
add_block('simulink/Ports & Subsystems/Out1', [target '/invRmsLatched'], 'Position', [455 38 485 52]);
add_block('simulink/Ports & Subsystems/Out1', [target '/invRmsLatchedValid'], 'Position', [455 98 485 112], 'Port', '2');
add_block('simulink/Ports & Subsystems/Out1', [target '/clearAccumulator'], 'Position', [455 158 485 172], 'Port', '3');
add_line(target, 'capture/1', 'CaptureAccepted/1');
add_line(target, 'invRmsInValid/1', 'CaptureAccepted/2');
add_line(target, 'invRmsIn/1', 'SelectNew/1');
add_line(target, 'CaptureAccepted/1', 'SelectNew/2');
add_line(target, 'StoredInvRms/1', 'SelectNew/3');
add_line(target, 'SelectNew/1', 'StoredInvRms/1');
add_line(target, 'StoredInvRms/1', 'invRmsLatched/1');
add_line(target, 'CaptureAccepted/1', 'CaptureAcceptedDelay/1');
add_line(target, 'CaptureAccepted/1', 'HoldValid/1');
add_line(target, 'LatchedValidState/1', 'HoldValid/2');
add_line(target, 'LatchedValidState/1', 'invRmsLatchedValid/1');
add_line(target, 'HoldValid/1', 'LatchedValidState/1');
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
add_block('simulink/Discrete/Unit Delay', [target '/WeightReg'], ...
    'InitialCondition', 'single(zeros(1, 8))', 'Position', [255 83 300 107]);
add_block('simulink/Math Operations/Product', [target '/ScaleMultiply'], 'Inputs', '2', ...
    'Multiplication', 'Element-wise(.*)', 'Position', [325 123 400 167]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [target '/InputBeatValid'], ...
    'Operator', 'AND', 'Inputs', '2', 'Position', [165 18 200 52]);
add_block('simulink/Discrete/Unit Delay', [target '/WeightValidReg'], ...
    'InitialCondition', 'false', 'Position', [220 18 265 42]);
add_block('simulink/Logic and Bit Operations/Logical Operator', [target '/ScaledBeatValid'], ...
    'Operator', 'AND', 'Inputs', '2', 'Position', [325 18 360 52]);
add_block('simulink/Discrete/Unit Delay', [target '/YBeatReg'], ...
    'InitialCondition', 'single(zeros(1, 8))', 'Position', [420 128 465 152]);
add_block('simulink/Discrete/Unit Delay', [target '/YBeatValidReg'], ...
    'InitialCondition', 'false', 'Position', [385 18 430 42]);
add_block('simulink/Ports & Subsystems/Out1', [target '/yBeat'], 'Position', [500 138 530 152]);
add_block('simulink/Ports & Subsystems/Out1', [target '/yBeatValid'], 'Position', [500 33 530 47], 'Port', '2');
add_line(target, 'xBeat/1', 'WeightMultiply/1');
add_line(target, 'gBeat/1', 'ScaleGamma/1');
add_line(target, 'ScaleGamma/1', 'WeightMultiply/2');
add_line(target, 'WeightMultiply/1', 'WeightReg/1');
add_line(target, 'WeightReg/1', 'ScaleMultiply/1');
add_line(target, 'invRms/1', 'InvRmsScalar/1');
add_line(target, 'InvRmsScalar/1', 'ScaleMultiply/2');
add_line(target, 'ScaleMultiply/1', 'YBeatReg/1');
add_line(target, 'YBeatReg/1', 'yBeat/1');
add_line(target, 'xBeatValid/1', 'InputBeatValid/1');
add_line(target, 'gBeatValid/1', 'InputBeatValid/2');
add_line(target, 'InputBeatValid/1', 'WeightValidReg/1');
add_line(target, 'WeightValidReg/1', 'ScaledBeatValid/1');
add_line(target, 'invRmsValid/1', 'ScaledBeatValid/2');
add_line(target, 'ScaledBeatValid/1', 'YBeatValidReg/1');
add_line(target, 'YBeatValidReg/1', 'yBeatValid/1');
end

function iAddStreamingDebugLogs(subsystem)
logSpecs = {
    'BeatAccumulator0/1', 'CurrentSumLog0', 'currentSumLog0', [930 60 1020 80];
    'BeatAccumulator0/2', 'CurrentSumValidLog0', 'currentSumValidLog0', [930 85 1020 105];
    'BeatAccumulator1/1', 'CurrentSumLog1', 'currentSumLog1', [930 110 1020 130];
    'BeatAccumulator1/2', 'CurrentSumValidLog1', 'currentSumValidLog1', [930 135 1020 155];
    'Controller/3', 'CaptureInvRmsLog', 'captureInvRmsLog', [930 160 1020 180];
    'Controller/4', 'CaptureInvRmsBankLog', 'captureInvRmsBankLog', [930 185 1020 205];
    'TokenSram/1', 'TokenSramReadBeatLog', 'tokenSramReadBeatLog', [930 210 1020 230];
    'TokenSram/2', 'TokenSramReadValidLog', 'tokenSramReadValidLog', [930 235 1020 255];
    'ScalarRsqrt0/1', 'ScalarRsqrtDataLog0', 'scalarRsqrtDataLog0', [930 260 1020 280];
    'ScalarRsqrt0/2', 'ScalarRsqrtValidLog0', 'scalarRsqrtValidLog0', [930 285 1020 305];
    'ScalarRsqrt1/1', 'ScalarRsqrtDataLog1', 'scalarRsqrtDataLog1', [930 310 1020 330];
    'ScalarRsqrt1/2', 'ScalarRsqrtValidLog1', 'scalarRsqrtValidLog1', [930 335 1020 355];
    'SelectedInvRms/1', 'InvRmsDataLog', 'invRmsDataLog', [930 360 1020 380];
    'SelectedInvRmsValid/1', 'InvRmsValidLog', 'invRmsValidLog', [930 385 1020 405];
    'LaneMultiply/1', 'LaneMultiplyDataLog', 'laneMultiplyDataLog', [930 410 1020 430];
    'LaneMultiply/2', 'LaneMultiplyValidLog', 'laneMultiplyValidLog', [930 435 1020 455]
    };

for logIndex = 1:size(logSpecs, 1)
    add_block('simulink/Sinks/To Workspace', [subsystem '/' logSpecs{logIndex, 2}], ...
        'VariableName', logSpecs{logIndex, 3}, 'SaveFormat', 'Structure With Time', ...
        'Position', logSpecs{logIndex, 4});
    add_line(subsystem, logSpecs{logIndex, 1}, [logSpecs{logIndex, 2} '/1'], 'autorouting', 'on');
end
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

function scriptText = iBuildTokenSramScript()
scriptText = strjoin({
    'function [readBeat, readValid] = SramCore(writeBeat, writeAddr, writeBank, writeValid, readAddr, readBank, readReq)'
    '%#codegen'
    'persistent bankA bankB'
    'if isempty(bankA)'
    '    bankA = zeros(1, 1536, ''single'');'
    '    bankB = zeros(1, 1536, ''single'');'
    'end'
    'if writeValid ~= 0'
    '    baseIndex = uint16(bitshift(writeAddr, 3));'
    '    for laneIndex = uint16(0):uint16(7)'
    '        if writeBank ~= 0'
    '            bankB(baseIndex + laneIndex + uint16(1)) = writeBeat(laneIndex + uint16(1));'
    '        else'
    '            bankA(baseIndex + laneIndex + uint16(1)) = writeBeat(laneIndex + uint16(1));'
    '        end'
    '    end'
    'end'
    'readBeat = zeros(1, 8, ''single'');'
    'readValid = false;'
    'if readReq ~= 0'
    '    baseIndex = uint16(bitshift(readAddr, 3));'
    '    for laneIndex = uint16(0):uint16(7)'
    '        if readBank ~= 0'
    '            readBeat(laneIndex + uint16(1)) = bankB(baseIndex + laneIndex + uint16(1));'
    '        else'
    '            readBeat(laneIndex + uint16(1)) = bankA(baseIndex + laneIndex + uint16(1));'
    '        end'
    '    end'
    '    readValid = true;'
    'end'
    'end'}, newline);
end
