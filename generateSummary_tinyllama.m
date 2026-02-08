function summary = generateSummary_tinyllama(mdl, text, nameValueArguments)
% GENERATESUMMARY_TINYLLAMA Generate summary using TinyLlama model (MATLAB implementation)
%
%   summary = generateSummary_tinyllama(mdl, text)
%
%   mdl: struct containing .Parameters (and optionally .Tokenizer if we cached it)
%   text: string or char array
%
%   Optional Name-Value arguments:
%       'MaxNewTokens' (default 50)
%       'Temperature'  (default 1.0)
%       'TopK'         (default 50)
%       'PromptTemplate' (string, default "Summarize:\n%s\n\nSummary:")

arguments
    mdl
    text 
    nameValueArguments.MaxNewTokens (1,1) {mustBeInteger, mustBePositive} = 50
    nameValueArguments.Temperature (1,1) {mustBePositive} = 1.0
    nameValueArguments.TopK (1,1) {mustBeInteger, mustBePositive} = 1
    nameValueArguments.PromptTemplate string = "Summarize the following text:\n%s\n\nSummary:"
end

    % 1. Initialize Tokenizer (Use existing if in mdl, otherwise create)
    if isfield(mdl, 'Tokenizer') && ~isempty(mdl.Tokenizer)
        tok = mdl.Tokenizer;
    else
        % Default to the one we know
        tok = llama.tokenizer.LlamaTokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    end

    % 2. Prepare Prompt
    % TinyLlama Chat template is specific, but here we use a simple prompting strategy 
    % or the one provided by argument.
    if strlength(nameValueArguments.PromptTemplate) > 0
        fullPrompt = sprintf(nameValueArguments.PromptTemplate, text);
    else
        fullPrompt = text;
    end
    
    % 3. Encode
    input_ids = tok.encode(fullPrompt);
    
    % 4. Generate
    % Using the pure MATLAB implementation in +llama/generate.m
    generated_ids = llama.generate(mdl, input_ids, ...
        'MaxNewTokens', nameValueArguments.MaxNewTokens, ...
        'Temperature', nameValueArguments.Temperature, ...
        'TopK', nameValueArguments.TopK);
        
    % 5. Decode
    % We accept the full sequence, but usually we only want the *new* text for a "summary" function
    % +llama/generate returns the FULL sequence (prompt + new).
    
    new_tokens = generated_ids(length(input_ids)+1:end);
    
    summary = tok.decode(new_tokens);
    
    % Cleanup strings (optional)
    summary = strtrim(summary);

end
