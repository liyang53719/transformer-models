function [summary, tokens] = generateSummary_Qwen3_5(mdl, text, varargin)
%generateSummary_Qwen3_5 - Generate text with the FP32 Qwen3.5 decoder
%   SUMMARY = generateSummary_Qwen3_5(MDL,TEXT) generates up to 50 tokens.
%   MDL contains Parameters and Tokenizer, as in the other model examples.
%
%   SUMMARY = generateSummary_Qwen3_5(...,MaxNewTokens=COUNT,PromptTemplate=FORMAT)
%   specifies the generation length and an optional sprintf prompt format.
%
%   [SUMMARY,TOKENS] = generateSummary_Qwen3_5(...) also returns token IDs.
%
%   See also qwen3_5.generate, qwen3_5.tokenizer.QwenTokenizer

p = inputParser;
addParameter(p, 'MaxNewTokens', 50);
addParameter(p, 'PromptTemplate', '%s');
addParameter(p, 'TopK', 1, @(x) isequal(x, 1));
parse(p, varargin{:});
prompt = sprintf(p.Results.PromptTemplate, char(text));
input = mdl.Tokenizer.encode(prompt);
tokens = qwen3_5.generate(input, mdl.Parameters, p.Results.MaxNewTokens);
summary = mdl.Tokenizer.decode(tokens);
end
