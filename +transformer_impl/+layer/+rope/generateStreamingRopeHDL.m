function targetDir = generateStreamingRopeHDL()
% generateStreamingRopeHDL   Generate HDL Coder RTL for the MATLAB RoPE DUT.

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
repoRoot = char(java.io.File(repoRoot).getCanonicalPath());
targetDir = fullfile(repoRoot, 'work', 'hdl', 'matlab_rope');
if ~isfolder(targetDir)
    mkdir(targetDir);
end

generatedDir = fullfile(targetDir, 'streamingRope_hdl_entry');
if isfolder(generatedDir)
    rmdir(generatedDir, 's');
end

cfg = coder.config('hdl');
cfg.TargetLanguage = 'Verilog';
cfg.GenerateReport = true;
cfg.GenerateOptimizationReport = false;
cfg.GenerateResourceReport = false;
cfg.GenerateHDLCode = true;
cfg.GenerateHDLTestBench = false;
cfg.GenerateCosimTestBench = false;
cfg.LaunchReport = false;
cfg.UseFloatingPoint = true;
cfg.FloatingPointTargetConfiguration = hdlcoder.createFloatingPointTargetConfig('NativeFloatingPoint');
cfg.TreatRealsInGeneratedCodeAs = 'None';
cfg.CheckConformance = false;
cfg.LaunchConformanceReport = false;
cfg.GenerateSimulinkModel = false;
cfg.ScalarizePorts = 'on';
cfg.ShareMultipliers = true;
cfg.ShareAdders = true;
cfg.ResourceSharing = int32(1);
cfg.TargetFrequency = 200;

codegen('-config', cfg, ...
    'streamingRope_hdl_entry', ...
    '-args', {false, uint16(64), uint8(2), zeros(1, 8, 'single'), false}, ...
    '-d', targetDir);

iWriteFileLists(targetDir);

end

function iWriteFileLists(targetDir)
rtlFiles = [dir(fullfile(targetDir, '**', '*.v')); dir(fullfile(targetDir, '**', '*.sv'))];
rtlPaths = cell(0, 1);
for fileIndex = 1:numel(rtlFiles)
    if rtlFiles(fileIndex).isdir
        continue;
    end
    fullPath = fullfile(rtlFiles(fileIndex).folder, rtlFiles(fileIndex).name);
    if contains(fullPath, [filesep 'html' filesep])
        continue;
    end
    if contains(rtlFiles(fileIndex).name, 'tb_')
        continue;
    end
    rtlPaths{end + 1, 1} = fullPath; %#ok<AGROW>
end

assert(~isempty(rtlPaths), 'No MATLAB HDL RTL files were generated under %s', targetDir);
rtlPaths = sort(rtlPaths);

tbPath = fullfile(targetDir, 'tb_streamingRope_fsdb.sv');
assert(isfile(tbPath), 'Missing VCS testbench: %s', tbPath);

iWriteFileList(fullfile(targetDir, 'rtl_filelist.f'), rtlPaths);
iWriteFileList(fullfile(targetDir, 'filelist.f'), [rtlPaths; {tbPath}]);
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