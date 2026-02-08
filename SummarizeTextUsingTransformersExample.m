%% Summarize Text Using Transformers
% This example shows how to summarize a piece of text using GPT-2.
% 
% Transformer networks such as GPT-2 can be used to summarize a piece of
% text. The trained GPT-2 transformer can generate text given an initial
% sequence of words as input. The model was trained on comments left on
% various web pages and internet forums.
% 
% Because lots of these comments themselves contain a summary indicated by
% the statement "TL;DR" (Too long, didn't read), you can use the
% transformer model to generate a summary by appending "TL;DR" to the input
% text. The |generateSummary| function takes the input text, automatically
% appends the string |"TL;DR"| and generates the summary.

%% Load Transformer Model
% Load the GPT-2 transformer model using the |gpt2| function.

mdl = gpt2;

%% Load Data
% Extract the help text for the |eigs| function.

% inputText = help('eigs')
inputText = "The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of minGPT that prioritizes teeth over education. Still under active development, but currently the file train.py reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: train.py is a ~300-line boilerplate training loop and model.py a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.";

%% Generate Summary
% Summarize the text using the |generateSummary| function.

rng('default')
tic;
summary = generateSummary(mdl,inputText)
toc

%% 3. Load MATLAB Model
disp("Loading MATLAB Model...");
modelPath = llama.download("TinyLlama/TinyLlama-1.1B-Chat-v1.0");
mdl_tinny_llm.Parameters = llama.load('tinyllama_params.mat');
mdl_tinny_llm.Tokenizer = llama.tokenizer.LlamaTokenizer(modelPath);

%% 4. Generate Summary (MATLAB)
disp("Running MATLAB Inference...");
tic;
% Use Greedy (TopK=1) for deterministic output, useful for comparison
summary_matlab = generateSummary_tinyllama(mdl_tinny_llm, inputText, ...
    'TopK', 1, ...
    'MaxNewTokens', 50, ...
    'PromptTemplate', "Summarize the following text:\n%s\n\nSummary:")
toc
