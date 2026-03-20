function targetDir = generateRopeHDL()
% generateRopeHDL   Generate HDL for the packed RoPE Simulink DUT.

thisDir = fileparts(mfilename('fullpath'));
targetDir = fullfile(thisDir, '..', '..', '..', 'work', 'hdl', 'simulink_rope');
targetDir = char(java.io.File(targetDir).getCanonicalPath());

if ~isfolder(targetDir)
    mkdir(targetDir);
end

generatedDir = fullfile(targetDir, 'rope');
if isfolder(generatedDir)
    rmdir(generatedDir, 's');
end

legacyPackedWrapper = fullfile(targetDir, 'DUTPacked.v');
if isfile(legacyPackedWrapper)
    delete(legacyPackedWrapper);
end

assignin('base', 'simStopTime', 10);
assignin('base', 'ropeStartSeq', timeseries([false; true; false], [0; 1; 2]));
assignin('base', 'ropeNumTokensSeq', timeseries(uint16([64; 64]), [0; 1]));
assignin('base', 'ropeNumHeadsSeq', timeseries(uint8([2; 2]), [0; 1]));
assignin('base', 'ropeInValidSeq', timeseries([false; false], [0; 1]));
assignin('base', 'ropeInBeatSeq', timeseries(zeros(2, 8, 'single'), [0; 1]));

ensureRopeModel();

hdlset_param('rope', 'TargetLanguage', 'SystemVerilog');
hdlset_param('rope', 'ScalarizePorts', 'off');
hdlset_param('rope', 'UseFloatingPoint', 'on');
hdlset_param('rope', 'TreatRealsInGeneratedCodeAs', 'None');
hdlset_param('rope', 'ClockRatePipelining', 'off');
hdlset_param('rope', 'DistributedPipelining', 'off');
hdlset_param('rope', 'AdaptivePipelining', 'off');
hdlset_param('rope', 'BalanceDelays', 'off');
hdlset_param('rope', 'GenerateModel', 'off');
hdlset_param('rope', 'HDLGenerateWebview', 'off');
hdlset_param('rope', 'FloatingPointTargetConfiguration', ...
    hdlcoder.createFloatingPointTargetConfig('NativeFloatingPoint'));

makehdl('rope', ...
    'HDLSubsystem', 'rope/DUTPacked', ...
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
statusPath = fullfile(targetDir, 'rope', 'hdlcodegenstatus.json');
statusInfo = jsondecode(fileread(statusPath));

rtlFiles = cell(numel(statusInfo.GenFileList), 1);
for fileIndex = 1:numel(statusInfo.GenFileList)
    rtlFiles{fileIndex} = fullfile(targetDir, 'rope', statusInfo.GenFileList{fileIndex});
end

tbPath = fullfile(targetDir, 'tb_rope_fsdb.sv');
assert(isfile(tbPath), 'Missing VCS testbench: %s', tbPath);

iWriteFileList(fullfile(targetDir, 'rtl_filelist.f'), rtlFiles);
iWriteFileList(fullfile(targetDir, 'filelist.f'), [rtlFiles; {tbPath}]);
end

function iWriteFileList(fileListPath, sourceFiles)
fid = fopen(fileListPath, 'w');
assert(fid >= 0, 'Failed to open filelist for writing: %s', fileListPath);
cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, '-timescale=1ns/1ps\n');
for fileIndex = 1:numel(sourceFiles)
    fprintf(fid, '%s\n', sourceFiles{fileIndex});
end
end