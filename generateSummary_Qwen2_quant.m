function summary = generateSummary_Qwen2_quant(mdl, text, varargin)
% GENERATESUMMARY_QWEN2_QUANT   Generate summary using Quantized Qwen2 model
%
%   summary = generateSummary_Qwen2_quant(mdl, text)
%   summary = generateSummary_Qwen2_quant(mdl, text, 'MaxNewTokens', 50, 'Temperature', 1)
%
%   This function works with quantized GGUF models loaded via qwen2_quant.load_gguf
%
%   Input:
%       mdl     - Struct with .Parameters and .Tokenizer
%       text    - String input containing the prompt (or raw text if PromptTemplate used)
%       Name-Value Arguments:
%           MaxNewTokens    - Max tokens to generate (default: 50)
%           Temperature     - Sampling temperature (default: 1.0)
%           TopK            - Top-K sampling (default: 1 for Greedy)
%           PromptTemplate  - Sprintf format string, e.g. "Summarize: %s"

    p = inputParser;
    addRequired(p, 'mdl', @(x) isstruct(x) && isfield(x, 'Parameters'));
    addRequired(p, 'text', @(x) isstring(x) || ischar(x));
    addParameter(p, 'MaxNewTokens', 50, @isnumeric);
    addParameter(p, 'Temperature', 1.0, @isnumeric);
    addParameter(p, 'TopK', 1, @isnumeric);
    addParameter(p, 'PromptTemplate', "%s", @(x) isstring(x) || ischar(x));
    parse(p, mdl, text, varargin{:});

    persistent pCache tCache kCache
    
    inputText = string(p.Results.text);
    maxNewTokens = p.Results.MaxNewTokens;
    temp = p.Results.Temperature;
    topK = p.Results.TopK;
    promptTemplate = string(p.Results.PromptTemplate);
    params = mdl.Parameters;
    requestedRuntimeCfg = struct();
    if isfield(params, 'RuntimeConfig') && isstruct(params.RuntimeConfig)
        requestedRuntimeCfg = params.RuntimeConfig;
    end

    backendType = "";
    if isstruct(params) && isfield(params, 'BackendType')
        backendType = lower(string(params.BackendType));
    end
    isHFBranch = any(strcmp(backendType, ["hf_gptq","hf_awq","hf_gptq_matlab","hf_awq_matlab"]));

    tokenizer = [];
    if isfield(mdl, 'Tokenizer')
        tokenizer = mdl.Tokenizer;
    end

    if isHFBranch
        if promptTemplate ~= "%s"
            prompt = sprintf(promptTemplate, inputText);
        else
            prompt = inputText;
        end

        if endsWith(backendType, "_matlab")
            if ~isfield(params, 'MatParamsFile') || ~exist(params.MatParamsFile, 'file')
                missingPath = "";
                if isfield(params, 'MatParamsFile')
                    missingPath = string(params.MatParamsFile);
                end
                error('generateSummary_Qwen2_quant:MatParamsMissing', ...
                    'MATLAB-native HF branch requires MatParamsFile. Missing: %s', missingPath);
            end
            tokPath = 'qwen_model';
            if isfield(params, 'TokenizerPath')
                tokPath = params.TokenizerPath;
            end

            if isempty(kCache) || ~strcmp(kCache, params.MatParamsFile + "|" + string(tokPath))
                pCache = qwen2_quant.load_hf_quant_matlab(params.MatParamsFile);
                if contains(backendType, "gptq") || contains(backendType, "awq")
                    defaultRuntimeCfg = struct( ...
                        'LinearMode', 'gptq_int4_quant_sim', ...
                        'TracePrecision', false, ...
                        'TraceTensors', false, ...
                        'Int8WeightScaleMode', 'per_row', ...
                        'Int8ActivationScaleMode', 'per_col');
                    pCache.RuntimeConfig = mergeStruct(defaultRuntimeCfg, requestedRuntimeCfg);
                end
                tCache = qwen2.tokenizer.QwenTokenizer(tokPath);
                kCache = params.MatParamsFile + "|" + string(tokPath);
            end
            params = pCache;
            tokenizer = tCache;
        else
            summary = qwen2_quant.internal.hf_quant_generate(prompt, params, ...
                'MaxNewTokens', maxNewTokens, ...
                'Temperature', temp, ...
                'TopK', topK, ...
                'ForceReload', false);
            return;
        end
    end

    if isempty(tokenizer)
        error('generateSummary_Qwen2_quant:MissingTokenizer', ...
            'Tokenizer is required for this branch.');
    end

    forceQuantForward = false;
    if isfield(params, 'RuntimeConfig') && isfield(params.RuntimeConfig, 'LinearMode')
        mode = string(params.RuntimeConfig.LinearMode);
        forceQuantForward = strcmpi(mode, "int8_int32_sim") || ...
                            strcmpi(mode, "q8_0_block_sim") || ...
                            strcmpi(mode, "q4_0_block_sim") || ...
                            strcmpi(mode, "q4_k_block_sim") || ...
                            strcmpi(mode, "q4_k_m_block_sim") || ...
                            strcmpi(mode, "gptq_int4_matlab_sim") || ...
                            strcmpi(mode, "gptq_int4_quant_sim");
    end

    % In quantized forward modes, keep linear layer weights quantized but
    % pre-dequantize non-linear/static tensors once to avoid per-token overhead.
    if forceQuantForward && isfield(params, 'Weights')
        if isfield(params.Weights, 'embed_tokens') && ...
                isa(params.Weights.embed_tokens, 'qwen2_quant.internal.quantized_weight')
            params.Weights.embed_tokens = dlarray(single(params.Weights.embed_tokens.dequantize()));
        end
        if isfield(params.Weights, 'norm') && ...
                isa(params.Weights.norm, 'qwen2_quant.internal.quantized_weight')
            params.Weights.norm = dlarray(single(params.Weights.norm.dequantize()));
        end
        if isfield(params.Weights, 'lm_head') && ...
                isa(params.Weights.lm_head, 'qwen2_quant.internal.quantized_weight')
            params.Weights.lm_head = dlarray(single(params.Weights.lm_head.dequantize()));
        end
    end

    % Choose forward function based on weight storage:
    % - quantized_weight present -> qwen2_quant.model (on-the-fly dequant)
    % - already dequantized floats -> qwen2.model (same path as FP32, better fidelity)
    forwardFn = @qwen2_quant.model;
    if ~forceQuantForward && isfield(params, 'Weights') && isfield(params.Weights, 'embed_tokens') && ...
            ~isa(params.Weights.embed_tokens, 'qwen2_quant.internal.quantized_weight')
        forwardFn = @qwen2.model;
    end
    
    % Format Prompt
    if promptTemplate ~= "%s"
         prompt = sprintf(promptTemplate, inputText);
    else
         prompt = inputText;
    end
    
    % Tokenize
    inputIds = tokenizer.encode(prompt);
    inputIds = double(inputIds);
    % Ensure correct shape [1, Seq]
    if size(inputIds, 1) > size(inputIds, 2)
        inputIds = inputIds'; 
    end
    
    % Generation Loop
    % 1. Prefill
    [logits, state] = forwardFn(inputIds, params);
    
    % Sample first token
    nextId = sampleToken(logits, temp, topK); 
    generatedIds = nextId;
    
    % 2. Autoregressive generation
    for i = 1:maxNewTokens
        % Model expects 0-based ID input
        [logits, state] = forwardFn(nextId, params, state);
        
        nextId = sampleToken(logits, temp, topK);
        
        % Stop Condition (EOS)
        if nextId == tokenizer.EosTokenId || nextId == tokenizer.PadTokenId || nextId == 151645
            break;
        end
        
        generatedIds = [generatedIds, nextId];
    end
    
    % Decode generated tokens
    summary = tokenizer.decode(generatedIds);
    
end

function merged = mergeStruct(base, override)
    merged = base;
    if ~isstruct(override)
        return;
    end
    f = fieldnames(override);
    for i = 1:numel(f)
        merged.(f{i}) = override.(f{i});
    end
end

function nextId = sampleToken(logits, temp, ~)
    % Sample next token from logits
    % Logits: [Vocab, Seq, Batch]
    % Take last token logits
    lastLogits = logits(:, end, 1);
    
    % Handle quantized_weight if needed
    if isa(lastLogits, 'qwen2_quant.internal.quantized_weight')
        lastLogits = lastLogits.dequantize();
    end
    lastLogits = extractdata(lastLogits);
    lastLogits = double(lastLogits);
    
    % Apply Temperature
    lastLogits = lastLogits / temp;
    
    % Top K Sort
    [~, idx] = sort(lastLogits, 'descend');
    
    nextId = double(idx(1)) - 1;  % Convert to 0-based, ensure double
    
    % Validate: must be non-negative integer
    if isnan(nextId) || isinf(nextId) || nextId < 0
        error('Invalid nextId: %f', nextId);
    end
end
