function reports = TestQwen3_5Numerics(modelFolder, referenceRoot)
%TestQwen3_5Numerics - Run FP32 Qwen3.5 operator and reference comparisons
%   REPORTS = TestQwen3_5Numerics(modelFolder,referenceRoot) loads the model
%   once and compares every Transformers/PyTorch case below referenceRoot.
%   Reference cases are prepared by tools/prepare_qwen3_5.py.
%
%   See also qwen3_5.compareTransformers, qwen3_5.load

repoRoot = fileparts(mfilename('fullpath'));
addpath(repoRoot);
results = runtests(fullfile(repoRoot, 'test', 'qwen3_5', 'qwen3_5OperatorsTest.m'));
assert(all([results.Passed]), 'qwen3_5:OperatorTestsFailed', 'Operator tests failed.');
parameters = qwen3_5.load(modelFolder);
listing = dir(referenceRoot);
reports = {};
for i = 1:numel(listing)
    referenceDir = fullfile(referenceRoot, listing(i).name);
    if listing(i).isdir && isfile(fullfile(referenceDir, 'case_metadata.json'))
        reports{end+1} = qwen3_5.compareTransformers(parameters, referenceDir); %#ok<AGROW>
    end
end
assert(~isempty(reports), 'qwen3_5:MissingReference', 'No reference cases found.');
fprintf('QWEN3_5_ALL_PASS cases=%d\n', numel(reports));
end
