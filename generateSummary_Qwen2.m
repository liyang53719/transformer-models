function summary = generateSummary_Qwen2(mdl, text, varargin)
% GENERATESUMMARY_QWEN2   Generate summary using Qwen2 model
%
%   summary = generateSummary_Qwen2(mdl, text)
%   summary = generateSummary_Qwen2(mdl, text, 'MaxNewTokens', 50, 'Temperature', 1)
%
%   Input:
%       mdl     - Struct with .Parameters and .Tokenizer
%       text    - String input containing the prompt (or raw text if PromptTemplate used)
%       Name-Value Arguments:
%           MaxNewTokens    - Max tokens to generate
%           Temperature     - Sampling temperature
%           TopK            - Top-K sampling (default 1 for Greedy)
%           PromptTemplate  - Sprintf format string, e.g. "Summarize: %s"

    p = inputParser;
    addRequired(p, 'mdl', @(x) isstruct(x) && isfield(x, 'Parameters') && isfield(x, 'Tokenizer'));
    addRequired(p, 'text', @(x) isstring(x) || ischar(x));
    addParameter(p, 'MaxNewTokens', 50, @isnumeric);
    addParameter(p, 'Temperature', 1.0, @isnumeric);
    addParameter(p, 'TopK', 1, @isnumeric);
    addParameter(p, 'PromptTemplate', "%s", @(x) isstring(x) || ischar(x));
    parse(p, mdl, text, varargin{:});
    
    inputText = string(p.Results.text);
    maxNewTokens = p.Results.MaxNewTokens;
    temp = p.Results.Temperature;
    topK = p.Results.TopK;
    promptTemplate = string(p.Results.PromptTemplate);
    
    tokenizer = mdl.Tokenizer;
    params = mdl.Parameters;
    
    % Format Prompt
    % If inputText contains "%s", assume it's the template? No, explicit PromptTemplate is cleaner.
    if promptTemplate ~= "%s"
         prompt = sprintf(promptTemplate, inputText);
    else
         prompt = inputText;
    end
    
    % Tokenize
    inputIds = tokenizer.encode(prompt);
    inputIds = double(inputIds);
    % Ensure correct shape [1, Seq] or [Seq, 1]. qwen2.model handles it as vector.
    if size(inputIds, 1) > size(inputIds, 2)
        inputIds = inputIds'; 
    end
    
    % Generation Loop
    % 1. Prefill
    % Pass full prompt
    [logits, state] = qwen2.model(inputIds, params);
    
    % Sample first token
    nextId = sampleInput(logits, temp, topK); 
    % nextId is 0-based token ID
    generatedIds = nextId;
    
    % 2. Autoregressive
    for i = 1:maxNewTokens
        % Model expects 0-based ID input (logic: X+1 internal conversion)
        [logits, state] = qwen2.model(nextId, params, state);
        
        nextId = sampleInput(logits, temp, topK);
        
        % Stop Condition (EOS)
        if nextId == tokenizer.EosTokenId || nextId == tokenizer.PadTokenId || nextId == 151645
            break;
        end
        
        generatedIds = [generatedIds, nextId];
    end
    
    summary = tokenizer.decode(generatedIds);
    
end

function nextId = sampleInput(logits, temp, topK)
    % Logits: [Vocab, Seq, Batch]
    % Take last token logits
    lastLogits = logits(:, end, 1);
    
    % Apply Temperature
    lastLogits = lastLogits / temp;
    
    % Top K Sort
    [probs, idx] = sort(lastLogits, 'descend');
    
    % idx corresponds to 1-based index in Vocab
    % Token ID = Index - 1
    
    if topK == 1
        % Greedy
        nextId = idx(1) - 1;
    else
        % Deterministic top 1 choice if sampling not implemented
        nextId = idx(1) - 1;
    end
    
    % Ensure correct type
    nextId = double(nextId);
end
