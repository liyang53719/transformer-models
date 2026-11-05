function [ids, primary] = stopTokenIds(modelFolder)
%stopTokenIds - Read text and chat generation terminators
%   IDS = stopTokenIds(modelFolder) combines model EOS IDs with the chat
%   EOS and FIM terminators declared by local tokenizer metadata.
%
%   [IDS,PRIMARY] = stopTokenIds(...) also returns the tokenizer's main EOS.
%
%   See also qwen3_5.load, qwen3_5.generate

cfg = jsondecode(fileread(fullfile(modelFolder, 'config.json')));
if isfield(cfg, 'text_config')
    cfg = cfg.text_config;
end
ids = double(cfg.eos_token_id(:).');
primary = ids(1);
generationFile = fullfile(modelFolder, 'generation_config.json');
if isfile(generationFile)
    generation = jsondecode(fileread(generationFile));
    if isfield(generation, 'eos_token_id')
        ids = [ids double(generation.eos_token_id(:).')];
    end
end
tokenizerFile = fullfile(modelFolder, 'tokenizer_config.json');
if isfile(tokenizerFile)
    meta = jsondecode(fileread(tokenizerFile));
    if isfield(meta, 'eos_token') && isfield(meta, 'added_tokens_decoder')
        eos = meta.eos_token;
        if isstruct(eos)
            eos = eos.content;
        end
        names = fieldnames(meta.added_tokens_decoder);
        for i = 1:numel(names)
            token = meta.added_tokens_decoder.(names{i});
            if strcmp(token.content, eos) || strcmp(token.content, '<|fim_suffix|>')
                id = str2double(regexp(names{i}, '\d+$', 'match', 'once'));
                ids(end+1) = id; %#ok<AGROW>
                if strcmp(token.content, eos)
                    primary = id;
                end
            end
        end
    end
end
ids = unique(ids, 'stable');
validateattributes(ids, {'numeric'}, ...
    {'vector', 'finite', 'integer', '>=', 0, '<', cfg.vocab_size});
end
