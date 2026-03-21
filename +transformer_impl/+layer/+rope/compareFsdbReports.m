function summary = compareFsdbReports(reportDir, numTokens, numHeads)
% compareFsdbReports   Compare fsdbreport CSV exports against golden RoPE outputs.

if nargin < 1 || strlength(string(reportDir)) == 0
    reportDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', 'work', 'tmp', 'fsdb_compare');
end
if nargin < 2
    numTokens = uint16(64);
end
if nargin < 3
    numHeads = uint8(12);
end

reportDir = char(java.io.File(reportDir).getCanonicalPath());
cfg = transformer_impl.layer.rope.qwen2_1p5b_config(numHeads);
lanes = double(cfg.Lanes);
expectedBeats = double(numTokens) * double(cfg.BeatsPerToken);

matlabValid = iReadValidCsv(fullfile(reportDir, 'm_valid.csv'));
simulinkValid = iReadValidCsv(fullfile(reportDir, 's_valid.csv'));

matlabValidIdx = find(matlabValid);
simulinkValidIdx = find(simulinkValid);

assert(numel(matlabValidIdx) == expectedBeats, ...
    'MATLAB FSDB valid-beat count mismatch: expected %d, observed %d.', expectedBeats, numel(matlabValidIdx));
assert(numel(simulinkValidIdx) == expectedBeats, ...
    'Simulink FSDB valid-beat count mismatch: expected %d, observed %d.', expectedBeats, numel(simulinkValidIdx));

matlabOut = zeros(lanes, expectedBeats);
simulinkOut = zeros(lanes, expectedBeats);
for laneIndex = 0:lanes-1
    matlabBits = iReadBitCsv(fullfile(reportDir, sprintf('m_lane%d.csv', laneIndex)));
    simulinkBits = iReadBitCsv(fullfile(reportDir, sprintf('s_lane%d.csv', laneIndex)));

    matlabOut(laneIndex + 1, :) = iDecodeDoubleBits(matlabBits(matlabValidIdx));
    simulinkOut(laneIndex + 1, :) = iDecodeSingleBits(simulinkBits(simulinkValidIdx));
end

inStream = iBuildInputStream(numTokens, numHeads, lanes, double(cfg.BeatsPerHead), double(cfg.BeatsPerToken));
golden = double(transformer_impl.layer.rope.referencePrefill(inStream, numTokens, numHeads));

matlabErr = abs(matlabOut - golden);
simulinkErr = abs(simulinkOut - golden);
rtlErr = abs(matlabOut - simulinkOut);

summary = struct();
summary.reportDir = string(reportDir);
summary.numTokens = uint16(numTokens);
summary.numHeads = uint8(numHeads);
summary.expectedBeats = expectedBeats;
summary.validCycleCountMatlab = numel(matlabValidIdx);
summary.validCycleCountSimulink = numel(simulinkValidIdx);
summary.firstValidCycleIndexMatlab = matlabValidIdx(1) - 1;
summary.lastValidCycleIndexMatlab = matlabValidIdx(end) - 1;
summary.firstValidCycleIndexSimulink = simulinkValidIdx(1) - 1;
summary.lastValidCycleIndexSimulink = simulinkValidIdx(end) - 1;
summary.maxAbsErrMatlabVsGolden = max(matlabErr, [], 'all');
summary.maxAbsErrSimulinkVsGolden = max(simulinkErr, [], 'all');
summary.maxAbsErrMatlabVsSimulink = max(rtlErr, [], 'all');
summary.matlabPass = summary.maxAbsErrMatlabVsGolden < 2.0e-5;
summary.simulinkPass = summary.maxAbsErrSimulinkVsGolden < 2.0e-5;
summary.rtlMatch = summary.maxAbsErrMatlabVsSimulink < 2.0e-5;

fprintf(['FSDB_CMP beats=%d matlab_valid=[%d,%d] simulink_valid=[%d,%d] ' ...
    'matlab_vs_golden=%.9g simulink_vs_golden=%.9g rtl_vs_rtl=%.9g\n'], ...
    summary.expectedBeats, summary.firstValidCycleIndexMatlab, summary.lastValidCycleIndexMatlab, ...
    summary.firstValidCycleIndexSimulink, summary.lastValidCycleIndexSimulink, ...
    summary.maxAbsErrMatlabVsGolden, summary.maxAbsErrSimulinkVsGolden, summary.maxAbsErrMatlabVsSimulink);

assert(summary.matlabPass, 'MATLAB RTL deviates from golden by %.9g.', summary.maxAbsErrMatlabVsGolden);
assert(summary.simulinkPass, 'Simulink RTL deviates from golden by %.9g.', summary.maxAbsErrSimulinkVsGolden);
assert(summary.rtlMatch, 'MATLAB RTL and Simulink RTL differ by %.9g.', summary.maxAbsErrMatlabVsSimulink);

disp('FSDB_CMP_OK');

end

function values = iReadValidCsv(filePath)
[~, bitStrings] = iReadCsvColumns(filePath);
values = strcmp(bitStrings, '1');
end

function bitStrings = iReadBitCsv(filePath)
[~, bitStrings] = iReadCsvColumns(filePath);
end

function [times, bitStrings] = iReadCsvColumns(filePath)
fid = fopen(filePath, 'r');
assert(fid >= 0, 'Failed to open %s', filePath);
cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>

header = fgetl(fid); %#ok<NASGU>
data = textscan(fid, '%f%s', 'Delimiter', ',', 'CollectOutput', false);
times = data{1};
bitStrings = data{2};
end

function values = iDecodeDoubleBits(bitStrings)
values = zeros(1, numel(bitStrings));
for idx = 1:numel(bitStrings)
    values(idx) = typecast(iBinaryStringToUint64(bitStrings{idx}), 'double');
end
end

function values = iDecodeSingleBits(bitStrings)
values = zeros(1, numel(bitStrings));
for idx = 1:numel(bitStrings)
    values(idx) = double(typecast(iBinaryStringToUint32(bitStrings{idx}), 'single'));
end
end

function value = iBinaryStringToUint64(bitString)
value = uint64(0);
for charIndex = 1:numel(bitString)
    value = bitshift(value, 1);
    if bitString(charIndex) == '1'
        value = bitor(value, uint64(1));
    end
end
end

function value = iBinaryStringToUint32(bitString)
value = uint32(0);
for charIndex = 1:numel(bitString)
    value = bitshift(value, 1);
    if bitString(charIndex) == '1'
        value = bitor(value, uint32(1));
    end
end
end

function inStream = iBuildInputStream(numTokens, numHeads, lanes, beatsPerHead, beatsPerToken)
totalBeats = double(numTokens) * beatsPerToken;
inStream = zeros(lanes, totalBeats, 'single');
for tokenIndex = 0:double(numTokens)-1
    tokenBase = tokenIndex * beatsPerToken;
    for headIndex = 0:double(numHeads)-1
        headBase = tokenBase + headIndex * beatsPerHead;
        for beatIndex = 0:beatsPerHead-1
            for laneIndex = 0:lanes-1
                inStream(laneIndex + 1, headBase + beatIndex + 1) = single( ...
                    -0.75 + tokenIndex * 0.03125 + headIndex * 0.0078125 + beatIndex * 0.001953125 + laneIndex * 0.000244140625);
            end
        end
    end
end
end