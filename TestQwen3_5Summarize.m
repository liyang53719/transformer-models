%TestQwen3_5Summarize - Compare local MATLAB and PyTorch FP32 summaries
%   Run this script directly. The default loads existing local weights.
%   Model downloads require explicitly setting allowDownload to true.
%
%   See also TestQwen3_5Numerics, qwen3_5.generate

%% 1. Configuration
modelName = "Qwen/Qwen3.5-0.8B";
modelFolder = fullfile('work', 'models', 'Qwen3.5-0.8B-FULL');
allowDownload = false;
setupPython = true;
forceReload = false;
basePython = "python3";
pythonEnvironment = fullfile('work', 'deps', 'qwen3_5_python');
outputFolder = fullfile('work', 'tmp', 'qwen3_5_transformers', 'summarize');
inputText = "The process of photosynthesis is how plants convert light " + ...
    "energy into chemical energy. Chlorophyll absorbs sunlight and uses " + ...
    "it to convert carbon dioxide and water into glucose.";
promptTemplate = "Summarize this: %s Summary:";
useChatTemplate = false;
maxNewTokens = 30;

%% 2. Check local weights and prepare Python dependencies
repoRoot = fileparts(mfilename('fullpath'));
addpath(repoRoot);
outputPath = fullfile(repoRoot, outputFolder);
if ~isfolder(outputPath)
    mkdir(outputPath);
end
prepareResponse = fullfile(outputPath, 'preparation.json');
pythonArgs = {char(basePython), fullfile(repoRoot, 'tools', 'prepare_qwen3_5.py'), ...
    '--model', fullfile(repoRoot, modelFolder), '--model-name', char(modelName), ...
    '--python-env', fullfile(repoRoot, pythonEnvironment), ...
    '--work-dir', outputPath, '--setup-only', '--response', prepareResponse, ...
    '--matlab-root', matlabroot};
pythonArgs = iPrepareOptions(pythonArgs, allowDownload, setupPython);
disp('Checking local weights and the project Python environment...');
iRun(pythonArgs);
runtime = jsondecode(fileread(prepareResponse));
pythonExecutable = runtime.python_executable;
request = struct('model_folder', modelFolder, 'prompt', sprintf(promptTemplate, inputText), ...
    'use_chat_template', useChatTemplate, 'max_new_tokens', maxNewTokens, ...
    'matlab_root', matlabroot);
tokenization = iPython('tokenize', request, pythonExecutable, repoRoot, outputPath);
prompt = string(tokenization.prompt);
inputIds = int32(tokenization.token_ids(:).');

%% 3. Load or reuse the MATLAB model
if forceReload || ~exist('mdl', 'var') || ~isfield(mdl, 'SourceSignature') || ...
        ~strcmp(mdl.SourceSignature, runtime.model_signature)
    disp('Loading MATLAB FP32 model...');
    mdl = struct('Parameters', qwen3_5.load(runtime.model_folder), ...
        'ModelName', modelName, 'SourceSignature', runtime.model_signature);
else
    disp('Reusing the MATLAB model from the workspace.');
end
mdl.Tokenizer = struct( ...
    'encode', @(text) iEncode(text, request, pythonExecutable, repoRoot, outputPath), ...
    'decode', @(ids) iDecode(ids, request, pythonExecutable, repoRoot, outputPath));

%% 4. Generate the MATLAB summary
disp('Running MATLAB greedy generation...');
[matlabIds, t_matlab] = iGenerate(inputIds, mdl.Parameters, maxNewTokens);
decodeRequest = request;
decodeRequest.token_ids = double(matlabIds);
decoded = iPython('decode', decodeRequest, pythonExecutable, repoRoot, outputPath);
summary_matlab = string(decoded.summary);
fprintf('MATLAB Summary: %s\nMATLAB Time: %.3f s\n', summary_matlab, t_matlab);

%% 5. Generate the Transformers/PyTorch reference
disp('Running Transformers/PyTorch FP32 reference...');
reference = iPython('generate', request, pythonExecutable, repoRoot, outputPath);
summary_python = string(reference.summary);
pythonIds = int32(reference.generated_token_ids(:).');
t_python = reference.generation_seconds;
fprintf('Python Summary: %s\nPython Time: %.3f s\n', summary_python, t_python);

%% 6. Compare and retain results in the workspace
promptEqual = isequal(inputIds(:), int32(reference.prompt_token_ids(:)));
tokensEqual = isequal(matlabIds(:), pythonIds(:));
textEqual = strcmp(summary_matlab, summary_python);
comparison = struct('Pass', promptEqual && tokensEqual && textEqual, ...
    'PromptTokensEqual', promptEqual, 'GeneratedTokensEqual', tokensEqual, ...
    'TextEqual', textEqual, 'MatlabSummary', summary_matlab, ...
    'PythonSummary', summary_python, 'MatlabIds', matlabIds, 'PythonIds', pythonIds, ...
    'MatlabSeconds', t_matlab, 'PythonSeconds', t_python, 'AllowDownload', allowDownload);
iWriteJson(fullfile(outputPath, 'matlab_summary_comparison.json'), comparison);
fprintf('Prompt IDs Equal: %s\nGenerated IDs Equal: %s\nText Equal: %s\n', ...
    string(promptEqual), string(tokensEqual), string(textEqual));
assert(comparison.Pass, 'qwen3_5:SummaryMismatch', 'MATLAB and PyTorch summaries differ.');
disp('QWEN3_5_TRANSFORMERS_SUMMARY_PASS');

function [ids, elapsed] = iGenerate(input, parameters, count)
previousThreads = maxNumCompThreads(8);
threadCleanup = onCleanup(@() maxNumCompThreads(previousThreads));
started = tic;
ids = qwen3_5.generate(input, parameters, count);
elapsed = toc(started);
end

function commandArgs = iPrepareOptions(commandArgs, download, setup)
if download
    commandArgs{end+1} = '--allow-download';
end
if ~setup
    commandArgs{end+1} = '--no-setup-python';
end
end

function ids = iEncode(text, request, python, root, output)
request.prompt = char(text);
request.use_chat_template = false;
response = iPython('tokenize', request, python, root, output);
ids = int32(response.token_ids(:).');
end

function text = iDecode(ids, request, python, root, output)
request.token_ids = double(ids);
response = iPython('decode', request, python, root, output);
text = string(response.summary);
end

function response = iPython(action, request, python, root, output)
requestPath = fullfile(output, [action '_request.json']);
responsePath = fullfile(output, [action '_response.json']);
iWriteJson(requestPath, request);
iRun({char(python), fullfile(root, 'tools', 'qwen3_5_transformers.py'), ...
    '--action', action, '--request', requestPath, '--response', responsePath});
response = jsondecode(fileread(responsePath));
assert(response.ok, 'qwen3_5:PythonFailed', 'Python reference failed.');
end

function iRun(commandArgs)
quoted = cellfun(@iQuote, commandArgs, 'UniformOutput', false);
status = system(strjoin(quoted, ' '));
assert(status == 0, 'qwen3_5:PythonFailed', 'Python command failed; see the output above.');
end

function value = iQuote(value)
quote = char(39);
value = [quote strrep(char(value), quote, [quote '"' quote '"' quote]) quote];
end

function iWriteJson(filePath, value)
fid = fopen(filePath, 'w', 'n', 'UTF-8');
assert(fid >= 0, 'qwen3_5:WriteFailed', 'Cannot write %s.', filePath);
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(value, 'PrettyPrint', true));
end
