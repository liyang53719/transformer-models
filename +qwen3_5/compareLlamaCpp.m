function report = compareLlamaCpp(parameters, referenceDir)
%compareLlamaCpp - Compare the FP32 decoder against llama.cpp dumps
%   REPORT = compareLlamaCpp(PARAMETERS,referenceDir) replays reference IDs
%   through MATLAB and compares all dumped operators, every layer,
%   and all vocabulary logits at every prefill and decode position.
%   The elementwise bound is 1e-3 + 1e-4 times the reference magnitude.
%
%   See also qwen3_5.model, qwen3_5.load

tokens = int32(readmatrix(fullfile(referenceDir, 'tokens.txt')));
generated = int32(readmatrix(fullfile(referenceDir, 'generated.txt')));
manifest = readtable(fullfile(referenceDir, 'tensors.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t', 'TextType', 'string');
stages = unique(manifest.stage, 'stable');
absTol = 1e-3;
relTol = 1e-4;
state = [];
position = 0;
rows = struct('Stage', {}, 'Tensor', {}, 'Count', {}, ...
    'MaxAbsError', {}, 'RMSE', {}, 'RelativeL2', {}, 'Pass', {});
top1Matches = 0;
top1Count = 0;
actualGenerated = zeros(1, 0, 'int32');
nextId = int32(-1);
for step = 1:numel(stages)
    stage = stages(step);
    selection = manifest(manifest.stage == stage, :);
    logitRow = selection(selection.name == "logits", :);
    assert(height(logitRow) == 1, 'qwen3_5:MissingReference', ...
        'Expected one logits record per stage.');
    if startsWith(stage, 'prefill')
        input = tokens(position + (1:logitRow.d1));
    else
        decodeIndex = sscanf(stage, 'decode_%d') + 1;
        input = generated(decodeIndex);
        actualGenerated(end+1) = nextId; %#ok<AGROW>
    end
    [logits, state, trace] = qwen3_5.model(input, parameters, state);
    position = position + numel(input);
    [~, nextIndex] = max(logits(:, end));
    nextId = int32(nextIndex - 1);
    layersChecked = false(1, parameters.Hyperparameters.NumLayers);
    stageStart = numel(rows) + 1;
    for r = 1:height(selection)
        name = selection.name(r);
        if name == "logits"
            actual = logits;
        elseif name == "result_norm"
            actual = trace.result_norm;
        else
            parts = regexp(name, '^(.*)-([0-9]+)$', 'tokens', 'once');
            assert(~isempty(parts), 'qwen3_5:UnknownTensor', ...
                'Unexpected reference tensor %s.', name);
            layer = str2double(parts{2}) + 1;
            actual = trace.Layers{layer}.(parts{1});
            if strcmp(parts{1}, 'l_out')
                layersChecked(layer) = true;
            end
        end
        expectedCount = prod(selection{r, {'d0', 'd1', 'd2', 'd3'}});
        assert(numel(actual) == expectedCount, 'qwen3_5:ShapeMismatch', ...
            'Element count differs for %s/%s.', stage, name);
        fid = fopen(fullfile(referenceDir, stage, name + '.bin'), 'r', 'ieee-le');
        assert(fid >= 0, 'qwen3_5:MissingReference', 'Missing %s/%s.', stage, name);
        closer = onCleanup(@() fclose(fid));
        expected = fread(fid, inf, '*single');
        clear closer
        assert(numel(expected) == expectedCount, 'qwen3_5:TruncatedReference', ...
            'Truncated reference %s/%s.', stage, name);
        reference = double(expected);
        observed = double(actual(:));
        delta = observed - reference;
        passed = all(isfinite(observed)) && all(isfinite(reference)) && ...
            all(abs(delta) <= absTol + relTol .* abs(reference));
        rows(end+1) = struct('Stage', char(stage), 'Tensor', char(name), ...
            'Count', expectedCount, 'MaxAbsError', max(abs(delta)), ...
            'RMSE', sqrt(mean(delta.^2)), ...
            'RelativeL2', norm(delta) / max(norm(reference), realmin), ...
            'Pass', passed); %#ok<AGROW>
        if name == "logits"
            expected = reshape(expected, size(logits));
            [~, referenceTop1] = max(expected, [], 1);
            [~, actualTop1] = max(logits, [], 1);
            top1Matches = top1Matches + nnz(referenceTop1 == actualTop1);
            top1Count = top1Count + numel(actualTop1);
        end
        if ~passed
            fprintf('MISMATCH %s/%s max_abs=%.9g rmse=%.9g rel_l2=%.9g\n', ...
                stage, name, rows(end).MaxAbsError, rows(end).RMSE, rows(end).RelativeL2);
        end
    end
    assert(all(layersChecked), 'qwen3_5:IncompleteReference', ...
        'Reference must include all decoder layer outputs.');
    stageRows = rows(stageStart:end);
    fprintf('MATLAB_COMPARE %s tokens=%d tensors=%d max_abs=%.9g pass=%d\n', ...
        stage, numel(input), numel(stageRows), max([stageRows.MaxAbsError]), ...
        all([stageRows.Pass]));
end
report = struct('ReferenceDirectory', char(referenceDir), ...
    'AbsoluteTolerance', absTol, 'RelativeTolerance', relTol, ...
    'PrefillTokens', numel(tokens), 'DecodeTokens', numel(generated), ...
    'Stages', numel(stages), 'TensorComparisons', numel(rows), ...
    'Top1Matches', top1Matches, 'Top1Count', top1Count, ...
    'ReferenceGeneratedTokens', generated(:).', ...
    'MatlabGeneratedTokens', actualGenerated, ...
    'GeneratedTokensEqual', isequal(actualGenerated, generated(:).'), ...
    'MaxAbsError', max([rows.MaxAbsError]), ...
    'Pass', all([rows.Pass]) && top1Matches == top1Count && ...
        isequal(actualGenerated, generated(:).'), ...
    'Tensors', rows);
fid = fopen(fullfile(referenceDir, 'matlab_comparison.json'), 'w');
assert(fid >= 0, 'qwen3_5:ReportWriteFailed', 'Cannot write comparison report.');
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(report, 'PrettyPrint', true));
fprintf('QWEN3_5_COMPARE pass=%d tensors=%d top1=%d/%d max_abs=%.9g\n', ...
    report.Pass, report.TensorComparisons, top1Matches, top1Count, report.MaxAbsError);
assert(report.Pass, 'qwen3_5:NumericalMismatch', 'MATLAB/llama.cpp comparison failed.');
end
