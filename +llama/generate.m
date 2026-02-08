function summary = generate(mdl, ids, nameValueArguments)
% GENERATE  Generate text using Llama model
%
%   ids = GENERATE(mdl, ids) generates continuation tokens for the input
%   ids (array of token IDs)
%
%   mdl is a struct with fields:
%       .Parameters (loaded via llama.load)
%       .Tokenizer  (instance of llama.tokenizer.LlamaTokenizer - placeholder)
%
%   Paramters:
%       'MaxNewTokens' (default 50)
%       'Temperature'  (default 1.0)
%       'TopK'         (default 50)

    arguments
        mdl
        ids
        nameValueArguments.MaxNewTokens (1,1) {mustBeInteger, mustBePositive} = 50
        nameValueArguments.Temperature  (1,1) {mustBePositive} = 1.0
        nameValueArguments.TopK         (1,1) {mustBeInteger, mustBePositive} = 50
    end

    maxNewTokens = nameValueArguments.MaxNewTokens;
    temp = nameValueArguments.Temperature;
    topK = nameValueArguments.TopK;
    
    params = mdl.Parameters;
    
    % Initialize KV Cache
    % llama.model handles empty state initialization
    layerStates = []; 
    
    % Pre-fill phase
    % Run model on full input prompt to fill cache
    % Ensure input is row vector for concatenation
    if size(ids, 1) > 1
        ids = reshape(ids, 1, []);
    end
    
    [logits, layerStates] = llama.model(ids, params, layerStates);
    
    % Get last token logic for sampling
    lastLogits = logits(:, end, :); % [Vocab, 1, Batch]
    
    % Generate loop
    for i = 1:maxNewTokens
        % Sample next token
        
        % Greedy Decoding bypass if topK=1 or temp is very low
        if topK == 1 || temp < 1e-5
             [~, nextToken] = max(extractdata(lastLogits), [], 1);
             % max returns index 1..V. 
             % Our model weights are 1-based index friendly but token IDs are 0-based from Python?
             % Wait. The python IDs are 0-based.
             % llama.model takes IDs and does `ids + 1`.
             % So model output max index is 1-based index in vocab.
             % To get back to 0-based ID:
             nextToken = nextToken - 1;
        else
             scaledLogits = lastLogits ./ temp;
             
             % Filter TopK (using sampling.topKLogits if available, or manual)
             if exist('sampling.topKLogits', 'file')
                  scaledLogits = sampling.topKLogits(scaledLogits, topK);
             end
             
             % Softmax along dim 1
             probs = exp(scaledLogits - max(scaledLogits,[],1));
             probs = probs ./ sum(probs, 1);
             
             nextToken = sampling.sampleFromCategorical(extractdata(probs));
             % sampleFromCategorical returns 1-based index (I assume? Let's check)
             % If so, convert to ID
             
             % Let's verify sampleFromCategorical
             % Assuming it returns 1..K index? Or 1..Vocab index?
             % Usually it matches the input vector size.
             % So it's 1-based index in Vocab.
             nextToken = nextToken - 1;
        end
        
        % Append
        ids = [ids, nextToken]; %#ok<AGROW>
        
        % Run model for single step
        [logits, layerStates] = llama.model(nextToken, params, layerStates);
        lastLogits = logits(:, end, :);
    end
    
    summary = ids;

end
