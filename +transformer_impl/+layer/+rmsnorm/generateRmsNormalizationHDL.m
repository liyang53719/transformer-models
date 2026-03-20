function targetDir = generateRmsNormalizationHDL()
% generateRmsNormalizationHDL   Generate Verilog for the MATLAB streaming DUT.

thisDir = fileparts(mfilename('fullpath'));
repoRoot = fullfile(thisDir, '..', '..', '..');
repoRoot = char(java.io.File(repoRoot).getCanonicalPath());
targetDir = fullfile(repoRoot, 'work', 'hdl', 'matlab_rmsNormalization');

if ~isfolder(targetDir)
    mkdir(targetDir);
end

cfg = coder.config('hdl');
cfg.TargetLanguage = 'Verilog';
cfg.GenerateReport = true;
cfg.GenerateOptimizationReport = true;
cfg.GenerateResourceReport = true;
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
cfg.ScalarizePorts = 'off';
cfg.ShareMultipliers = true;
cfg.ShareAdders = true;
cfg.ResourceSharing = int32(1);
cfg.TargetFrequency = 200;

codegen('-config', cfg, ...
    'rmsNormalization_hdl_entry', ...
    '-args', {false, false, zeros(1, 8, 'single'), false, zeros(1, 8, 'single'), false, single(1e-6)}, ...
    '-d', targetDir);

end