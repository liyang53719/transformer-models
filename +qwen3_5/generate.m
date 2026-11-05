function [tokens, state] = generate(inputTokens, parameters, maxNewTokens)
%GENERATE - Generate Qwen3.5 tokens with greedy FP32 decoding
%   TOKENS = GENERATE(inputTokens,PARAMETERS,maxNewTokens) returns generated
%   zero-based IDs, excluding the prompt, and stops at the configured EOS.
%
%   [TOKENS,STATE] = GENERATE(...) also returns the decoder state after the
%   last input consumed. The final returned token is not yet in the cache.
%
%   See also qwen3_5.model, qwen3_5.load

validateattributes(maxNewTokens, {'numeric'}, {'scalar', 'integer', 'positive'});
[logits, state] = qwen3_5.model(inputTokens, parameters);
tokens = zeros(1, maxNewTokens, 'int32');
for i = 1:maxNewTokens
    [~, next] = max(logits(:, end));
    tokens(i) = int32(next - 1);
    if tokens(i) == parameters.Hyperparameters.EosTokenId
        tokens = tokens(1:i);
        return
    end
    if i < maxNewTokens
        [logits, state] = qwen3_5.model(tokens(i), parameters, state);
    end
end
end
