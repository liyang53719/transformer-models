function modelPath = download(modelName, destination)
% DOWNLOAD Download Qwen2 model files for offline usage via Hugging Face Hub
%
%   modelPath = qwen2.download()
%       Downloads default Qwen2 model to 'qwen_model' in current folder.
%
%   modelPath = qwen2.download(modelName)
%       Downloads specified model repo ID.
%
%   modelPath = qwen2.download(modelName, destination)
%       Downloads to specific destination folder.

    arguments
        modelName (1,1) string = "Qwen/Qwen2.5-1.5B-Instruct" 
        destination (1,1) string = fullfile(pwd, 'qwen_model')
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
    
    % Automatically run the preparation/conversion script to generate .mat files
    % This is expected to be in 'tools/prepare_tinyllama.py' relative to project root
    scriptPath = fullfile("tools", "prepare_tinyllama.py");
    paramsFile = "tinyllama_params.mat";
    
    % Only run if script exists. 
    % Also check if params already exist? 
    % The user requested "automatically execute after download", implying we should likely ensure it runs or at least ensure the output exists.
    % We will run it if params.mat is MISSING or if we just performed a fresh download (though we don't track that easy boolean here, let's just check file existence)
    
    if exist(scriptPath, 'file')
        if ~exist(paramsFile, 'file') || ~exist(destination, 'dir') 
             % Check destination 'dir' is weird condition here, but basically if we just downloaded we might want to run.
             % Simplest logic: If params don't exist, run. 
             % What if user wants to update? 
             % Let's run it always if script exists, it is relatively fast if verified.
             
             fprintf("Executing conversion script '%s'...\n", scriptPath);
             
             % Determine Python Executable
             pe = pyenv;
             if pe.Status == "Loaded"
                 pyExe = pe.Executable;
             else
                 pyExe = "python"; % Default fallback
             end
             
             % Run script
             cmd = sprintf('"%s" "%s"', pyExe, scriptPath);
             [status, cmdout] = system(cmd);
             
             if status ~= 0
                 warning("llama:download:ConversionFailed", ...
                     "Model downloaded, but weight conversion failed.\nCommand: %s\nOutput:\n%s", cmd, cmdout);
             else
                 fprintf("Weight conversion complete.\n");
             end
        else
            fprintf("Weights already converted ('%s' exists).\n", paramsFile);
        end
    else
        warning("Preparation script '%s' not found. Please run parameter conversion manually.", scriptPath);
    end
    
    modelPath = destination;
end
