classdef QwenTokenizer < handle
    %QwenTokenizer - Tokenize local Qwen3.5 text without loading Python weights
    %   OBJ = QwenTokenizer(modelFolder) opens tokenizer.json using the
    %   Python tokenizers package. Token IDs stay zero based.
    %
    %   QwenTokenizer functions:
    %       encode - Convert text to token IDs
    %       decode - Convert token IDs to text
    %
    %   QwenTokenizer properties:
    %       EosTokenId - End-of-sequence token ID
    %
    %   See also qwen3_5.load, qwen3_5.generate

    properties (SetAccess = private)
        EosTokenId  % End-of-sequence token ID
    end
    properties (Access = private)
        Backend
    end
    methods
        function obj = QwenTokenizer(modelFolder)
            module = py.importlib.import_module('tokenizers');
            obj.Backend = module.Tokenizer.from_file( ...
                char(fullfile(modelFolder, 'tokenizer.json')));
            [~, obj.EosTokenId] = qwen3_5.internal.stopTokenIds(modelFolder);
        end
        function tokens = encode(obj, text)
            %ENCODE - Convert text to zero-based token IDs
            %   TOKENS = ENCODE(OBJ,TEXT) encodes text without adding tokens.
            %
            %   See also decode, qwen3_5.model
            result = obj.Backend.encode(char(text), pyargs('add_special_tokens', false));
            tokens = int32(cellfun(@double, cell(result.ids)));
        end
        function text = decode(obj, tokens)
            %DECODE - Convert zero-based token IDs to text
            %   TEXT = DECODE(OBJ,TOKENS) preserves special token spellings.
            %
            %   See also encode, qwen3_5.generate
            ids = py.list(num2cell(int64(tokens(:).')));
            text = string(obj.Backend.decode(ids, pyargs('skip_special_tokens', false)));
        end
    end
end
