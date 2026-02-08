classdef LlamaTokenizer < handle
    % LLAMATOKENIZER   Wrapper for HuggingFace Tokenizer via Python Bridge
    %
    %   obj = llama.tokenizer.LlamaTokenizer(modelName)
    %
    %   Prerequisites:
    %       - Python environment configured in MATLAB (pyenv)
    %       - 'transformers' library installed in that environment
    
    properties
        PyTokenizer
        ModelName
    end
    
    methods
        function obj = LlamaTokenizer(modelName)
            if nargin < 1
                modelName = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
            end
            obj.ModelName = modelName;
            
            try
                % Attempt to import transformers
                tr = py.importlib.import_module('transformers');
                
                % Transformers uses LazyModule, so we must use getattr to trigger it
                AutoTokenizerClass = py.getattr(tr, 'AutoTokenizer');
                
                % Load tokenizer
                % We use local_files_only=true first to avoid connection issues if cached
                try
                    obj.PyTokenizer = AutoTokenizerClass.from_pretrained(modelName, local_files_only=true);
                catch
                    % Fallback to online (requires network/proxy)
                    obj.PyTokenizer = AutoTokenizerClass.from_pretrained(modelName);
                end
            catch ME
                warning("LlamaTokenizer:PythonError", "Failed to initialize Python Tokenizer. \nError: %s\nVerify 'transformers' is installed in current pyenv.", ME.message);
            end
        end
        
        function ids = encode(obj, text)
            % encode   Convert string to IP list
            % ids = obj.encode("Hello world")
            
            if isempty(obj.PyTokenizer)
                error("Tokenizer not initialized successfully.");
            end
            
            % Python: tokenizer.encode(text) -> List[int]
            % Ensure text is char or string
            if isstring(text)
                text = char(text);
            end
            
            pyOut = obj.PyTokenizer.encode(text);
            
            % Convert Python list to MATLAB double
            ids = double(pyOut);
        end
        
        function text = decode(obj, ids)
            % decode   Convert ID array to string
            % text = obj.decode(ids)
            
            if isempty(obj.PyTokenizer)
                error("Tokenizer not initialized successfully.");
            end
            
            % Convert MATLAB array to Python list of integers
            % casting to int64 ensures python treats them as ints, not floats
            pyIds = py.list(int64(ids));
            
            % Python: tokenizer.decode(ids, skip_special_tokens=True)
            % Note: Named arguments in MATLAB python interface
            % obj.PyTokenizer.decode(pyIds, pyargs('skip_special_tokens', true))?
            
            % AutoTokenizer.decode usually accepts lists
            try
                pyStr = obj.PyTokenizer.decode(pyIds, skip_special_tokens=true);
            catch
                % Fallback for older matlab/python bindings
                kw = pyargs('skip_special_tokens', true);
                pyStr = obj.PyTokenizer.decode(pyIds, kw);
            end
            
            text = string(pyStr);
        end
    end
end
