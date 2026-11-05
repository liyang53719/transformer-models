%TestQwen3_5Summarize - Compare MATLAB and llama.cpp greedy summaries
%   Run this script from MATLAB to print both summaries, generation times,
%   and exact token/text equality. Edit the configuration below as needed.
%   Model data and the prepared reference stay under the ignored work folder.
%
%   See also TestQwen3_5Numerics, generateSummary_Qwen3_5

%% Configuration
modelFolder = fullfile('work', 'models', 'Qwen3.5-0.8B-FULL');
referenceWork = fullfile('work', 'tmp', 'qwen3_5');
inputText = "The process of photosynthesis is how plants convert light " + ...
    "energy into chemical energy. Chlorophyll absorbs sunlight and uses " + ...
    "it to convert carbon dioxide and water into glucose.";
promptTemplate = "<|im_start|>user\nSummarize the following text in one sentence:\n" + ...
    "%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
maxNewTokens = 30;

%% Run independent greedy generation and compare
comparison = iCompareSummaries(modelFolder, referenceWork, ...
    inputText, promptTemplate, maxNewTokens);

function result = iCompareSummaries(modelFolder, referenceWork, text, template, count)
repoRoot = fileparts(mfilename('fullpath'));
previousPath = path;
pathCleanup = onCleanup(@() path(previousPath));
addpath(repoRoot);
previousThreads = maxNumCompThreads(8);
threadCleanup = onCleanup(@() maxNumCompThreads(previousThreads));
modelPath = fullfile(repoRoot, modelFolder);
assert(isfile(fullfile(modelPath, 'config.json')), ...
    'qwen3_5:MissingModel', 'Place the local model files in %s.', modelFolder);
outputDir = fullfile(referenceWork, 'summarize');
outputPath = fullfile(repoRoot, outputDir);
if ~isfolder(outputPath)
    mkdir(outputPath);
end

%% MATLAB summary
disp('Loading MATLAB FP32 model...');
mdl.Parameters = qwen3_5.load(modelPath);
mdl.Tokenizer = qwen3_5.tokenizer.QwenTokenizer(modelPath);
prompt = sprintf(template, text);
inputIds = mdl.Tokenizer.encode(prompt);
disp('Running MATLAB greedy generation...');
started = tic;
[summaryMatlab, matlabIds] = generateSummary_Qwen3_5(mdl, text, ...
    'TopK', 1, 'MaxNewTokens', count, 'PromptTemplate', template);
matlabSeconds = toc(started);
fprintf('MATLAB Summary: %s\nMATLAB Time: %.3f s\n', summaryMatlab, matlabSeconds);

%% Python launches the independent llama.cpp reference
request = struct('model_folder', modelFolder, 'reference_work', referenceWork, ...
    'output_dir', outputDir, 'prompt', prompt, 'max_new_tokens', count, ...
    'matlab_root', matlabroot);
requestPath = fullfile(outputPath, 'request.json');
responsePath = fullfile(outputPath, 'response.json');
iWriteJson(requestPath, request);
disp('Running llama.cpp FP32 reference through Python...');
python = pyenv;
command = py.list({char(python.Executable), ...
    fullfile(repoRoot, 'tools', 'qwen3_5_llama_infer.py'), ...
    '--request', requestPath, '--response', responsePath});
process = py.subprocess.run(command, pyargs('capture_output', true, 'text', true));
assert(double(process.returncode) == 0, 'qwen3_5:ReferenceFailed', ...
    'Python/llama.cpp reference failed:\n%s\n%s', char(process.stdout), char(process.stderr));
reference = jsondecode(fileread(responsePath));
assert(reference.ok, 'qwen3_5:ReferenceFailed', 'Reference returned an error.');
fprintf('llama.cpp Summary: %s\nllama.cpp Time: %.3f s\n', ...
    reference.summary, reference.generation_seconds);

%% Compare the actual independently generated IDs and text
promptEqual = isequal(inputIds(:), int32(reference.prompt_token_ids(:)));
tokensEqual = isequal(matlabIds(:), int32(reference.generated_token_ids(:)));
textEqual = strcmp(summaryMatlab, string(reference.summary));
result = struct('PromptTokensEqual', promptEqual, 'GeneratedTokensEqual', tokensEqual, ...
    'TextEqual', textEqual, 'MatlabSummary', summaryMatlab, ...
    'ReferenceSummary', string(reference.summary), ...
    'MatlabGeneratedTokens', matlabIds, ...
    'ReferenceGeneratedTokens', int32(reference.generated_token_ids(:).'), ...
    'MatlabSeconds', matlabSeconds, 'ReferenceSeconds', reference.generation_seconds, ...
    'MaxNewTokens', count, 'GeneratedTokens', numel(matlabIds), ...
    'Pass', promptEqual && tokensEqual && textEqual);
iWriteJson(fullfile(outputPath, 'matlab_summary_comparison.json'), result);
fprintf('Prompt IDs Equal: %s\nGenerated IDs Equal: %s\nText Equal: %s\n', ...
    string(promptEqual), string(tokensEqual), string(textEqual));
assert(result.Pass, 'qwen3_5:SummaryMismatch', 'MATLAB and llama.cpp summaries differ.');
disp('QWEN3_5_SUMMARY_COMPARE_PASS');
end

function iWriteJson(filePath, value)
fid = fopen(filePath, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'qwen3_5:WriteFailed', 'Cannot write %s.', filePath);
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(value, 'PrettyPrint', true));
end
