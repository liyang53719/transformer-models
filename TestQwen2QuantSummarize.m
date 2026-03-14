%% TestQwen2QuantSummarize
% Compare Qwen2 GGUF quantization precisions with one top-level script.
% Uses generateSummary_Qwen2_quant.m for all formats.

%% 1. Configuration
repoId = "Qwen/Qwen2.5-1.5B-Instruct-GGUF";
modelFolder = "qwen_gguf";
quantFormats = ["q8_0", "q4_0", "q4_k_m"];
useHfMirror = true;
setenv('TOKENIZERS_PARALLELISM', 'false');
dequantizeNow = false; % true: faster inference, higher RAM; false: lower RAM, slower

promptText = "The process of photosynthesis is how plants convert light energy into chemical energy. Chlorophyll absorbs sunlight and uses it to convert carbon dioxide and water into glucose.";
promptTemplate = "Summarize this: %s Summary:";
maxNewTokens = 30;
topK = 1;

if ~exist('qwen_model', 'dir')
    error('Tokenizer folder qwen_model not found. Please prepare tokenizer first.');
end

if ~exist(modelFolder, 'dir')
    mkdir(modelFolder);
end

%% 2. Run per-quantization comparison
numFormats = numel(quantFormats);
loadTime = zeros(numFormats, 1);
inferTime = zeros(numFormats, 1);
outputs = strings(numFormats, 1);
modelPaths = strings(numFormats, 1);

for i = 1:numFormats
    fmt = quantFormats(i);
    ggufFile = fullfile(modelFolder, "qwen2.5-1.5b-instruct-" + fmt + ".gguf");
    modelPaths(i) = string(ggufFile);

    ensureGGUF(repoId, ggufFile, useHfMirror);

    fprintf('\n===================================\n');
    fprintf('Format: %s\n', fmt);
    fprintf('File  : %s\n', ggufFile);

    mdl = struct();
    tic;
    mdl.Parameters = qwen2_quant.load_gguf(ggufFile, 'DequantizeNow', dequantizeNow, 'Verbose', false);
    if ~dequantizeNow
        linearMode = "float";
        if fmt == "q8_0"
            linearMode = "q8_0_block_sim";
        elseif fmt == "q4_0"
            linearMode = "q4_0_block_sim";
        elseif fmt == "q4_k_m"
            linearMode = "q4_k_m_block_sim";
        end
        mdl.Parameters.RuntimeConfig = struct('LinearMode', char(linearMode), 'TracePrecision', false);
        fprintf('LinearMode: %s\n', linearMode);
    end
    mdl.Tokenizer = qwen2.tokenizer.QwenTokenizer('qwen_model');
    loadTime(i) = toc;

    tic;
    outputs(i) = string(generateSummary_Qwen2_quant(mdl, promptText, ...
        'TopK', topK, ...
        'MaxNewTokens', maxNewTokens, ...
        'PromptTemplate', promptTemplate));
    inferTime(i) = toc;

    fprintf('Load Time : %.3fs\n', loadTime(i));
    fprintf('Infer Time: %.3fs\n', inferTime(i));
    fprintf('Summary   : %s\n', outputs(i));
end

%% 3. Report
fprintf('\n===================================\n');
fprintf('Quantization Comparison Summary\n');
fprintf('===================================\n');
for i = 1:numFormats
    fprintf('%-7s | load: %7.3fs | infer: %7.3fs\n', quantFormats(i), loadTime(i), inferTime(i));
end

baseText = outputs(1);
for i = 2:numFormats
    fprintf('Text Equal (%s vs %s): %s\n', quantFormats(1), quantFormats(i), string(strcmp(baseText, outputs(i))));
end

%% Local helpers
function ensureGGUF(repoId, ggufFile, useHfMirror)
    if exist(ggufFile, 'file')
        return;
    end

    fprintf('Missing GGUF: %s\n', ggufFile);

    [hasHf, ~] = system('command -v hf');
    [hasHfCli, ~] = system('command -v huggingface-cli');
    if hasHf ~= 0 && hasHfCli ~= 0
        error(['Neither hf nor huggingface-cli found. Install with: ', ...
            'pip install -U huggingface_hub']);
    end

    [ggufDir, ggufName, ggufExt] = fileparts(ggufFile);
    ggufDir = string(ggufDir);
    ggufName = string(ggufName);
    ggufExt = string(ggufExt);
    targetName = ggufName + ggufExt;
    repoIdChar = char(strip(string(repoId)));
    targetNameChar = char(strip(targetName));
    ggufDirChar = char(strip(ggufDir));

    if useHfMirror
        setenv('HF_ENDPOINT', 'https://hf-mirror.com');
        fprintf('Using HF mirror endpoint: %s\n', getenv('HF_ENDPOINT'));
    else
        setenv('HF_ENDPOINT', '');
    end

    cmdList = strings(0, 1);
    if hasHf == 0
        cmdList(end+1) = sprintf('hf download "%s" "%s" --local-dir "%s"', ...
            repoIdChar, targetNameChar, ggufDirChar);
    end
    if hasHfCli == 0
        cmdList(end+1) = sprintf(['huggingface-cli download "%s" "%s" ', ...
            '--local-dir "%s" --local-dir-use-symlinks False'], ...
            repoIdChar, targetNameChar, ggufDirChar);
    end

    status = 1;
    out = '';

    for ci = 1:numel(cmdList)
        [status, out] = system(cmdList(ci));
        if status == 0 && exist(ggufFile, 'file')
            return;
        end
    end

    if useHfMirror
        fprintf('Mirror download failed, retrying with official endpoint...\n');
        setenv('HF_ENDPOINT', '');
        for ci = 1:numel(cmdList)
            [status, out] = system(cmdList(ci));
            if status == 0 && exist(ggufFile, 'file')
                return;
            end
        end
    end

    if status ~= 0 || ~exist(ggufFile, 'file')
        error('Failed to download GGUF file. Command output:\n%s', out);
    end
end
