%% TestSummarizeTinyLlama
% This script demonstrates how to summarize text using the TinyLlama model
% in MATLAB, and compares the result against a Python reference.

%% 0. Setup Python
% Set environment variable to prevent MKL/OpenMP conflicts causing Segfaults
% when mixing MATLAB (MKL) and PyTorch (MKL/OpenMP).
setenv('KMP_DUPLICATE_LIB_OK', 'TRUE');
setenv('OMP_NUM_THREADS', '1'); % Restrict Python threads to avoid contention

% Configure MATLAB to use a compatible Python environment
possiblePaths = ["/usr/bin/python3", "/usr/bin/python", "python3", "python"];
pyPath = "";
for p = possiblePaths
    [s, out] = system("which " + p);
    if s == 0 && contains(strtrim(string(out)), "/usr/bin")
        pyPath = strtrim(string(out));
        break; 
    end
end
if pyPath ~= ""
    pe = pyenv;
    if pe.Status == "NotLoaded"
        try
             pyenv('Version', pyPath);
        catch ME
             disp("Warning: Could not set Python version: " + ME.message);
        end
    end
end

%% 1. Ensure Model Downloaded
% We use the new llama.download function to ensure files are local
disp("Checking Model Availability...");
% This will download to 'tinyllama_model' in current dir if missing
modelPath = llama.download("TinyLlama/TinyLlama-1.1B-Chat-v1.0");

%% 2. Setup Input Text
% You can modify the input text here.
inputText = "Deep Learning is a subset of machine learning, which is essentially a neural network with three or more layers. These neural networks attempt to simulate the behavior of the human brain—albeit far from matching its ability—allowing it to learn from large amounts of data. While a neural network with a single layer can still make approximate predictions, additional hidden layers can help to optimize and refine for accuracy.";

disp("=== Input Text ===");
disp(inputText);
disp(" ");

%% 3. Load MATLAB Model
% Ensure params are available for pure MATLAB inference
if ~exist('tinyllama_params.mat','file')
    warning("tinyllama_params.mat not found. Running prepare tool...");
    % Start python script to convert weights
    % Note: prepare_tinyllama.py needs either internet or the local model we just downloaded
    % Let's pass the local path to it if we could, but the script is hardcoded.
    % We assume the user runs tools/prepare_tinyllama.py manually or we try to run it.
    system("python tools/prepare_tinyllama.py"); 
end

disp("Loading MATLAB Model...");
if ~exist('mdl','var')
    % Only load once to save time
    mdl.Parameters = llama.load('tinyllama_params.mat');
    
    % Initialize Tokenizer using local path
    mdl.Tokenizer = llama.tokenizer.LlamaTokenizer(modelPath);
end

%% 4. Generate Summary (MATLAB)
disp("Running MATLAB Inference...");
tic;
% Use Greedy (TopK=1) for deterministic output, useful for comparison
summary_matlab = generateSummary_tinyllama(mdl, inputText, ...
    'TopK', 1, ...
    'MaxNewTokens', 50, ...
    'PromptTemplate', "Summarize the following text:\n%s\n\nSummary:");
t_matlab = toc;
disp("MATLAB Summary: " + summary_matlab);
disp("Time: " + t_matlab + "s");

%% 5. Generate Summary (Python Reference)
% Note: Running PyTorch inside MATLAB process can cause MKL/OpenMP conflicts (Segfaults).
% We will run the Python generation in a separate process via `system()` to imply safety.

disp(" ");
disp("Running Python Inference (Reference)...");
tic;

% Construct command to run python script externally
% We need a wrapper script because generateSummary_tinyllama_py is a MATLAB function
% that uses the in-process bridge.
% Instead, we will generate a temporary python script and run it.

pyScriptContent = ["import torch";
                   "from transformers import AutoModelForCausalLM, AutoTokenizer";
                   "import sys";
                   "model_path='" + modelPath + "'";
                   "input_text='Summarize the following text:\n" + inputText + "\n\nSummary:'";
                   "print(f'Loading model from {model_path}...')";
                   "tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)";

                   "model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)";
                   "input_ids = tokenizer(input_text, return_tensors='pt').input_ids";
                   "outputs = model.generate(input_ids, max_new_tokens=50, do_sample=False, top_k=1)";
                   "new_tokens = outputs[0][len(input_ids[0]):]";
                   "summary = tokenizer.decode(new_tokens, skip_special_tokens=True)";
                   "print('PYTHON_SUMMARY_START')";
                   "print(summary.strip())";
                   "print('PYTHON_SUMMARY_END')";
                   ];
                   
fid = fopen('temp_py_infer.py', 'w');
fprintf(fid, '%s\n', pyScriptContent);
fclose(fid);

[status, cmdout] = system(pyPath + " temp_py_infer.py");
summary_py = "";

if status == 0
    % Parse output
    lines = splitlines(string(cmdout));
    idxStart = find(lines == "PYTHON_SUMMARY_START", 1);
    idxEnd = find(lines == "PYTHON_SUMMARY_END", 1);
    if ~isempty(idxStart) && ~isempty(idxEnd)
        summary_py = join(lines(idxStart+1:idxEnd-1), newline);
    end
else
    disp("Python execution failed:");
    disp(cmdout);
    summary_py = "ERROR";
end

% Cleanup
delete('temp_py_infer.py');

t_py = toc;
disp("Python Summary: " + summary_py);
disp("Time: " + t_py + "s");

%% 6. Compare
disp(" ");
disp("=== Comparison ===");
if summary_matlab == summary_py
    disp("[SUCCESS] The summaries match exactly!");
else
    disp("[DIFFERENCE] The summaries differ.");
    disp("Length MATLAB: " + strlength(summary_matlab));
    disp("Length Python: " + strlength(summary_py));
end
