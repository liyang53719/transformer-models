function report = compareTransformers(parameters, referenceDir)
%compareTransformers - Compare decoder outputs with PyTorch FP32 references
%   REPORT = compareTransformers(PARAMETERS,referenceDir) checks all layers,
%   vocabulary logits, recurrent state, convolution history, and KV caches.
%   Each element must meet 1e-3 + 1e-4 times the reference magnitude.
%
%   See also qwen3_5.model, TestQwen3_5Numerics

metadata = jsondecode(fileread(fullfile(referenceDir, 'case_metadata.json')));
state = [];
top1Matches = 0;
top1Count = 0;
nextId = int32(-1);
generated = zeros(1, 0, 'int32');
rows = struct('Stage', {}, 'Tensor', {}, 'MaxAbsError', {}, 'RMSE', {}, 'Pass', {});
for stage = 1:numel(metadata.Stages)
    reference = builtin('load', fullfile(referenceDir, metadata.Stages{stage}));
    input = int32(reference.input_ids(:).');
    if stage > 1
        generated(end+1) = nextId; %#ok<AGROW>
    end
    chunk = metadata.MatlabPrefillChunk;
    if stage == 1 && chunk > 0
        actualLogits = zeros(parameters.Hyperparameters.VocabSize, 0, 'single');
        values = struct();
        finalNorm = zeros(parameters.Hyperparameters.HiddenSize, 0, 'single');
        for offset = 1:chunk:numel(input)
            [part, state, trace] = qwen3_5.model( ...
                input(offset:min(offset+chunk-1, end)), parameters, state);
            actualLogits = [actualLogits part]; %#ok<AGROW>
            partValues = iLayerValues(trace);
            names = fieldnames(partValues);
            for i = 1:numel(names)
                name = names{i};
                if offset == 1
                    values.(name) = partValues.(name);
                else
                    values.(name) = [values.(name) partValues.(name)];
                end
            end
            finalNorm = [finalNorm trace.result_norm]; %#ok<AGROW>
        end
    else
        [actualLogits, state, trace] = qwen3_5.model(input, parameters, state);
        values = iLayerValues(trace);
        finalNorm = trace.result_norm;
    end
    [~, predicted] = max(actualLogits, [], 1);
    [~, expected] = max(reference.logits, [], 1);
    top1Matches = top1Matches + nnz(predicted == expected);
    top1Count = top1Count + numel(expected);
    nextId = int32(predicted(end) - 1);
    values.logits = actualLogits;
    values.result_norm = finalNorm;
    for i = 1:parameters.Hyperparameters.NumLayers
        cache = state.Layers{i};
        if isfield(cache, 'SSM')
            values.(sprintf('ssm_%d', i-1)) = cache.SSM;
            values.(sprintf('conv_%d', i-1)) = cache.History;
        else
            values.(sprintf('keys_%d', i-1)) = cache.Keys;
            values.(sprintf('values_%d', i-1)) = cache.Values;
        end
    end
    names = fieldnames(values);
    first = numel(rows) + 1;
    for i = 1:numel(names)
        name = names{i};
        a = double(values.(name)(:));
        b = double(reference.(name)(:));
        assert(numel(a) == numel(b), 'qwen3_5:ShapeMismatch', 'Shape differs for %s.', name);
        delta = abs(a - b);
        passed = all(isfinite(a)) && all(isfinite(b)) && ...
            all(delta <= 1e-3 + 1e-4 .* abs(b));
        rows(end+1) = struct('Stage', stage, 'Tensor', name, ...
            'MaxAbsError', max(delta), 'RMSE', sqrt(mean(delta.^2)), ...
            'Pass', passed); %#ok<AGROW>
        if ~passed
            fprintf('MISMATCH stage=%d %s max_abs=%.9g\n', stage, name, max(delta));
        end
    end
    fprintf('PYTORCH_COMPARE case=%s stage=%d tensors=%d max_abs=%.9g pass=%d\n', ...
        metadata.name, stage, numel(names), max([rows(first:end).MaxAbsError]), ...
        all([rows(first:end).Pass]));
end

report = struct('Case', metadata.name, 'Backend', 'Transformers/PyTorch FP32 CPU', ...
    'PrefillTokens', metadata.PrefillTokens, 'DecodeTokens', numel(generated), ...
    'Top1Matches', top1Matches, 'Top1Count', top1Count, ...
    'TensorComparisons', numel(rows), 'MaxAbsError', max([rows.MaxAbsError]), ...
    'GeneratedTokensEqual', isequal(generated(:), int32(metadata.GeneratedTokens(:))), ...
    'Tensors', rows);
report.Pass = all([rows.Pass]) && top1Matches == top1Count && report.GeneratedTokensEqual;
fid = fopen(fullfile(referenceDir, 'matlab_comparison.json'), 'w');
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(report, 'PrettyPrint', true));
assert(report.Pass, 'qwen3_5:NumericalMismatch', 'MATLAB/PyTorch numerical comparison failed.');
end

function values = iLayerValues(trace)
values = struct();
for i = 1:numel(trace.Layers)
    layer = trace.Layers{i};
    for name = {'l_out', 'attn_norm', 'attn_post_norm', 'ffn_out'}
        values.(sprintf('%s_%d', name{1}, i-1)) = layer.(name{1});
    end
    if isfield(layer, 'linear_attn_out')
        attention = layer.linear_attn_out;
    else
        attention = layer.attn_output;
    end
    values.(sprintf('attention_%d', i-1)) = attention;
end
end
