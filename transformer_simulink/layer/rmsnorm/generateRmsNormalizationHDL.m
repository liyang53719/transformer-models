function targetDir = generateRmsNormalizationHDL()
% generateRmsNormalizationHDL   Generate HDL for the packed Simulink DUT.

thisDir = fileparts(mfilename('fullpath'));
targetDir = fullfile(thisDir, '..', '..', '..', 'work', 'hdl', 'simulink_rmsNormalization');
targetDir = char(java.io.File(targetDir).getCanonicalPath());

if ~isfolder(targetDir)
    mkdir(targetDir);
end

generatedDir = fullfile(targetDir, 'rmsNormalization');
if isfolder(generatedDir)
    rmdir(generatedDir, 's');
end

legacyPackedWrapper = fullfile(targetDir, 'DUTPacked.v');
if isfile(legacyPackedWrapper)
    delete(legacyPackedWrapper);
end

ensureRmsNormalizationModel();

hdlset_param('rmsNormalization', 'TargetLanguage', 'SystemVerilog');
hdlset_param('rmsNormalization', 'ScalarizePorts', 'off');
hdlset_param('rmsNormalization', 'UseFloatingPoint', 'on');
hdlset_param('rmsNormalization', 'TreatRealsInGeneratedCodeAs', 'None');
hdlset_param('rmsNormalization', 'ClockRatePipelining', 'off');
hdlset_param('rmsNormalization', 'DistributedPipelining', 'off');
hdlset_param('rmsNormalization', 'AdaptivePipelining', 'off');
hdlset_param('rmsNormalization', 'BalanceDelays', 'off');
hdlset_param('rmsNormalization', 'GenerateModel', 'off');
hdlset_param('rmsNormalization', 'HDLGenerateWebview', 'off');
hdlset_param('rmsNormalization', 'FloatingPointTargetConfiguration', ...
    hdlcoder.createFloatingPointTargetConfig('NativeFloatingPoint'));

makehdl('rmsNormalization', ...
    'HDLSubsystem', 'rmsNormalization/DUTPacked', ...
    'TargetLanguage', 'SystemVerilog', ...
    'TargetDirectory', targetDir, ...
    'GenerateHDLTestBench', 'off', ...
    'GenerateValidationModel', 'off', ...
    'Traceability', 'off', ...
    'ResourceReport', 'off', ...
    'OptimizationReport', 'off', ...
    'HDLGenerateWebview', 'off', ...
    'CodeGenerationOutput', 'GenerateHDLCode');

iWriteFileLists(targetDir);

end

function iWriteFileLists(targetDir)
statusPath = fullfile(targetDir, 'rmsNormalization', 'hdlcodegenstatus.json');
statusInfo = jsondecode(fileread(statusPath));

rtlFiles = cell(numel(statusInfo.GenFileList), 1);
for fileIndex = 1:numel(statusInfo.GenFileList)
    rtlFiles{fileIndex} = fullfile(targetDir, 'rmsNormalization', statusInfo.GenFileList{fileIndex});
end

tbPath = fullfile(targetDir, 'tb_rmsNormalization_fsdb.sv');
assert(isfile(tbPath), 'Missing VCS testbench: %s', tbPath);

iWriteFileList(fullfile(targetDir, 'rtl_filelist.f'), rtlFiles);
iWriteFileList(fullfile(targetDir, 'filelist.f'), [rtlFiles; {tbPath}]);
end

function iWriteFileList(fileListPath, sourceFiles)
fid = fopen(fileListPath, 'w');
assert(fid >= 0, 'Failed to open filelist for writing: %s', fileListPath);
cleaner = onCleanup(@() fclose(fid));

fprintf(fid, '-timescale=1ns/1ps\n');
for fileIndex = 1:numel(sourceFiles)
    fprintf(fid, '%s\n', sourceFiles{fileIndex});
end
end
