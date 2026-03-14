function parameters = load_hf_quant(modelNameOrPath, options)
% LOAD_HF_QUANT Build parameter struct for HuggingFace quantized Qwen2 branches.
%
%   params = qwen2_quant.load_hf_quant("Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4")
%   params = qwen2_quant.load_hf_quant("Qwen/Qwen2.5-1.5B-Instruct-AWQ")

    arguments
        modelNameOrPath (1,1) string
        options.Engine (1,1) string {mustBeMember(options.Engine,["python_ref","matlab_native"])} = "python_ref"
        options.LocalFilesOnly (1,1) logical = true
        options.AutoRetryOnline (1,1) logical = true
        options.HFEndpoint (1,1) string = "https://hf-mirror.com"
        options.TrustRemoteCode (1,1) logical = true
        options.UseGPU (1,1) logical = true
        options.MatParamsFile (1,1) string = ""
        options.TokenizerPath (1,1) string = "qwen_model"
    end

    lowerName = lower(modelNameOrPath);
    if contains(lowerName, 'gptq')
        backendType = "hf_gptq";
    elseif contains(lowerName, 'awq')
        backendType = "hf_awq";
    else
        error('qwen2_quant:load_hf_quant:UnknownBackend', ...
            'Cannot infer backend from model name/path: %s (expect GPTQ or AWQ).', modelNameOrPath);
    end

    parameters = struct();
    if options.Engine == "matlab_native"
        parameters.BackendType = char(backendType + "_matlab");
    else
        parameters.BackendType = char(backendType);
    end
    parameters.Engine = char(options.Engine);
    parameters.ModelName = char(modelNameOrPath);
    parameters.LocalFilesOnly = options.LocalFilesOnly;
    parameters.AutoRetryOnline = options.AutoRetryOnline;
    parameters.HFEndpoint = char(options.HFEndpoint);
    parameters.TrustRemoteCode = options.TrustRemoteCode;
    parameters.UseGPU = options.UseGPU;

    if options.Engine == "matlab_native"
        if strlength(options.MatParamsFile) == 0
            [~, modelLeaf] = fileparts(modelNameOrPath);
            defaultMat = fullfile(pwd, 'qwen_hf_quant_matlab', modelLeaf + "_params.mat");
            options.MatParamsFile = defaultMat;
        end
        if options.TokenizerPath == "qwen_model" && exist(modelNameOrPath, 'dir')
            options.TokenizerPath = modelNameOrPath;
        end
        parameters.MatParamsFile = char(options.MatParamsFile);
        parameters.TokenizerPath = char(options.TokenizerPath);
    end
end
