classdef QwenTokenizer < handle
    % QwenTokenizer   MATLAB wrapper for Hugging Face AutoTokenizer (Qwen)
    %
    %   Loads tokenizer via Python.
    
    properties
        PyTokenizer
    end
    
    properties (Dependent)
        EosTokenId
        PadTokenId
    end
    
    methods
        function val = get.EosTokenId(obj)
            if ~isempty(obj.PyTokenizer) && ~isempty(obj.PyTokenizer.eos_token_id)
                val = double(obj.PyTokenizer.eos_token_id);
            else
                val = 151643; % Default Qwen EOS
            end
        end
        
        function val = get.PadTokenId(obj)
            if ~isempty(obj.PyTokenizer) && ~isempty(obj.PyTokenizer.pad_token_id)
                val = double(obj.PyTokenizer.pad_token_id);
            else
                val = 151643; % Default Qwen Pad
            end
        end

        function obj = QwenTokenizer(modelPath)
            % Initialize Python Tokenizer
            try
                if count(py.sys.path, '') == 0
                    insert(py.sys.path, int32(0), '');
                end
                
                % Use builtin getattr to handle transformers LazyModule
                builtins = py.importlib.import_module('builtins');
                transformers = py.importlib.import_module('transformers');
                AutoTokenizer = builtins.getattr(transformers, 'AutoTokenizer');
                
                % Use fast tokenizer
                obj.PyTokenizer = AutoTokenizer.from_pretrained(modelPath, pyargs('trust_remote_code', true, 'local_files_only', true));
            catch
                % Fallback
                 builtins = py.importlib.import_module('builtins');
                 transformers = py.importlib.import_module('transformers');
                 AutoTokenizer = builtins.getattr(transformers, 'AutoTokenizer');
                 
                 obj.PyTokenizer = AutoTokenizer.from_pretrained(modelPath, pyargs('trust_remote_code', true));
            end
        end
        
        function ids = encode(obj, text)
            % Encode text to token IDs (1-based for MATLAB, but output is 0-based from Py)
            % We keep them as double.
            % Check if text is string/char
            textStr = string(text);
            
            % encode
            encoded = obj.PyTokenizer.encode(textStr);
            
            % Convert to double
            ids = double(encoded);
            
            % Ensure column vector
            ids = reshape(ids, [], 1);
        end
        
        function str = decode(obj, ids)
            % Decode token IDs to string
            % ids: vector of numbers (integers)
            
            % Ensure input is treated as integers for Python
            if isa(ids, 'dlarray')
                ids = extractdata(ids);
            end
            ids = int64(ids);

            % PyTokenizer.decode expects list of integers
            pylist = py.list();
            for i = 1:numel(ids)
                pylist.append(ids(i));
            end
            
            decoded = obj.PyTokenizer.decode(pylist, pyargs('skip_special_tokens', true));
            str = char(decoded);
        end
    end
end
