function summary = generate(mdl, ids, nameValueArguments)
% GENERATE  Generate text using Qwen2 quantized model
%
%   summary = GENERATE(mdl, ids)
%   summary = GENERATE(mdl, ids, 'MaxNewTokens', 50, 'Temperature', 1.0)
%
%   This is adapted from qwen2.generate to work with quantized models.

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
    layerStates = []; 
    
    % Ensure input is row vector
    if size(ids, 1) > 1
        ids = reshape(ids, 1, []);
    end
    
    % Pre-fill phase
    [logits, layerStates] = qwen2_quant.model(ids, params, layerStates);
    
    % Get last token logits
    lastLogits = logits(:, end, :);
    
    % Generate loop
    for i = 1:maxNewTokens
        % Sample next token
        if topK == 1 || temp < 1e-5
             [~, nextToken] = max(extractdata(lastLogits), [], 1);
             nextToken = nextToken - 1;
        else
             scaledLogits = lastLogits ./ temp;
             
             % Softmax
             probs = exp(scaledLogits - max(scaledLogits,[],1));
             probs = probs ./ sum(probs, 1);
             
             nextToken = sampleFromCategorical(extractdata(probs));
             nextToken = nextToken - 1;
        end
        
        % Append to output
        ids = [ids, nextToken];
        
        % Check for EOS
        if mdl.Tokenizer.EosTokenId == nextToken
            break;
        end
        
        % Run model for single step
        [logits, layerStates] = qwen2_quant.model(nextToken, params, layerStates);
        lastLogits = logits(:, end, :);
    end
    
    summary = ids;
end

function idx = sampleFromCategorical(probs)
    % Sample from categorical distribution
    cumprobs = cumsum(probs(:));
    idx = find(cumprobs >= rand(), 1, 'first');
end
