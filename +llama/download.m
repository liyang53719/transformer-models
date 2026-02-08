function modelPath = download(modelName, destination)
% DOWNLOAD Download TinyLlama model files for offline usage via Hugging Face Hub
%
%   modelPath = llama.download()
%       Downloads default TinyLlama model to 'tinyllama_model' in current folder.
%
%   modelPath = llama.download(modelName)
%       Downloads specified model repo ID.
%
%   modelPath = llama.download(modelName, destination)
%       Downloads to specific destination folder.

    arguments
        modelName (1,1) string = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        destination (1,1) string = fullfile(pwd, 'tinyllama_model')
    end

    if ~exist(destination, 'dir')
        fprintf("Downloading model '%s'...\n", modelName);
        fprintf("Destination: %s\n", destination);
        
        try
            % Ensure we have python loaded
            if count(py.sys.path, '') == 0
                insert(py.sys.path, int32(0), '');
            end
            
            % Use Python's huggingface_hub to download
            % equivalent to: from huggingface_hub import snapshot_download
            % hf_hub = py.importlib.import_module('huggingface_hub');
            % Note: snapshot_download is often a function inside the module, but sometimes needs specific import
            
            % Try getting the function directly
            % from huggingface_hub import snapshot_download
            % snapshot_download(...)
            
            % If simply importing module doesn't expose it (it should), we use getattr
            hf_hub = py.importlib.import_module('huggingface_hub');
            snapshot_download_func = py.getattr(hf_hub, 'snapshot_download');
            
            % Fix Proxy manually here if needed since this is the designated download script
            % Clearing the bad SOCKS proxy that seems to persist in environment
            os = py.importlib.import_module('os');
            bad_keys = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"];
            for k = bad_keys
                val = char(os.environ.get(k));
                if ~isempty(val) && contains(val, "socks")
                     fprintf("Clearing unsupported proxy var %s=%s\n", k, val);
                     os.environ.pop(k);
                end
            end
            
            % Call the function
            % Note: resume_download and local_dir_use_symlinks are deprecated/ignored in newer versions
            snapshot_download_func(repo_id=modelName, ...
                local_dir=destination); 
                
            fprintf("Download complete.\n");
        catch ME
            % Enhance error message
            msg = "Failed to download model via Python bridge.";
            if contains(ME.message, "ModuleNotFoundError")
                msg = msg + " Verify 'huggingface_hub' is installed in your Python environment.";
            elseif contains(ME.message, "ConnectionError") || contains(ME.message, "proxy")
                msg = msg + " Check your network/proxy connection.";
            end
            error("llama:download:Failed", "%s\nPython Error: %s", msg, ME.message);
        end
    else
        fprintf("Model already exists locally at: %s\n", destination);
    end
    
    modelPath = destination;
end
