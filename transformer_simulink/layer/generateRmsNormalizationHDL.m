function targetDir = generateRmsNormalizationHDL()
% generateRmsNormalizationHDL   Generate Verilog for the Simulink streaming DUT.

thisDir = fileparts(mfilename('fullpath'));
targetDir = fullfile(thisDir, '..', '..', 'work', 'hdl', 'simulink_rmsNormalization');
targetDir = char(java.io.File(targetDir).getCanonicalPath());

if ~isfolder(targetDir)
    mkdir(targetDir);
end

buildRmsNormalizationModel();
load_system('rmsNormalization');
cleaner = onCleanup(@() close_system('rmsNormalization', 0));

hdlset_param('rmsNormalization', 'TargetLanguage', 'Verilog');
hdlset_param('rmsNormalization', 'UseFloatingPoint', 'on');
hdlset_param('rmsNormalization', 'TreatRealsInGeneratedCodeAs', 'None');
hdlset_param('rmsNormalization', 'ClockRatePipelining', 'off');
hdlset_param('rmsNormalization', 'DistributedPipelining', 'off');
hdlset_param('rmsNormalization', 'AdaptivePipelining', 'off');
hdlset_param('rmsNormalization', 'FloatingPointTargetConfiguration', ...
    hdlcoder.createFloatingPointTargetConfig('NativeFloatingPoint'));

makehdl('rmsNormalization', ...
    'HDLSubsystem', 'rmsNormalization/DUT', ...
    'TargetLanguage', 'Verilog', ...
    'TargetDirectory', targetDir, ...
    'GenerateHDLTestBench', 'off', ...
    'GenerateValidationModel', 'off', ...
    'Traceability', 'on', ...
    'ResourceReport', 'on', ...
    'OptimizationReport', 'on');

clear cleaner;

end
