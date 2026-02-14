%% TestSummarizeQwen2
% This script demonstrates how to summarize text using the Qwen2-1.5B model
% in MATLAB, and compares the result against a Python reference.

%% 0. Setup Python
setenv('KMP_DUPLICATE_LIB_OK', 'TRUE');
setenv('OMP_NUM_THREADS', '1');

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

%% 1. Configuration
modelName = "Qwen/Qwen2.5-1.5B-Instruct"; 
% Note: Qwen2.5 is the latest, matching the prepare_qwen.py default.
% User originally asked for Qwen2 1.5B, but prepare_qwen defaults to 2.5-1.5B.
% We will stick to the repo used in prepare_qwen.py to avoid mismatch.

modelFolder = "qwen_model";
matParamsFile = "qwen_params.mat";
inputText = "Note: Qwen2.5 is the latest, matching the prepare_qwen.py default. User originally asked for Qwen2 1.5B, but prepare_qwen defaults to 2.5-1.5B.We will stick to the repo used in prepare_qwen.py to avoid mismatch."
% inputText = "The process of photosynthesis is how plants convert light energy into chemical energy. Chlorophyll absorbs sunlight and uses it to convert carbon dioxide and water into glucose.";

%% 2. Download Model (if needed)
if ~exist(modelFolder, 'dir')
    disp("Downloading Model (this may take time)...");
    qwen2.download(modelName, modelFolder);
end

%% 3. Prepare Weights (if needed)
if ~exist(matParamsFile, 'file')
    disp("Converting Weights to MATLAB format...");
    % Start python script to convert weights
    system("python tools/prepare_qwen.py"); 
end

disp("Loading MATLAB Model...");
if ~exist('mdl','var')
    mdl.Parameters = qwen2.load(matParamsFile);
    % Initialize Tokenizer using local path
    mdl.Tokenizer = qwen2.tokenizer.QwenTokenizer(modelFolder);
end

%% 4. Generate Summary (MATLAB)
disp("Running MATLAB Inference...");
tic;
% Use Greedy (TopK=1) for deterministic output
summary_matlab = generateSummary_Qwen2(mdl, inputText, ...
    'TopK', 1, ...
    'MaxNewTokens', 30, ...
    'PromptTemplate', "Summarize this: %s Summary:");
t_matlab = toc;
disp("MATLAB Summary: " + summary_matlab);
disp("Time: " + t_matlab + "s");

%% 5. Generate Summary (Python Reference)
disp("Running Python Inference for Comparison...");
% We construct a small python script on the fly to ensure same inputs
pyScript = [
"import torch",
"from transformers import AutoModelForCausalLM, AutoTokenizer",
"import time",
"model_id = './" + modelFolder + "'",
"text = '" + inputText + "'",
"prompt = f'Summarize this: {text} Summary:'",
"tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)",
"model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True)",
"inputs = tokenizer(prompt, return_tensors='pt')",
"t0 = time.time()",
"outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False, top_p=None, temperature=None, repetition_penalty=1.0)",
"dt = time.time() - t0",
"summary = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)",
"print(f'Python Summary: {summary}')",
"print(f'Time: {dt}s')",
];

fid = fopen('temp_qwen_infer.py', 'w');
fprintf(fid, '%s\n', pyScript);
fclose(fid);

system("python temp_qwen_infer.py");
delete('temp_qwen_infer.py');
