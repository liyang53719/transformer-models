function matFile = prepare_hf_quant_matlab(modelNameOrPath, outputMatFile, options)
% PREPARE_HF_QUANT_MATLAB Export HF quant model weights into MATLAB .mat for native inference.

    arguments
        modelNameOrPath (1,1) string
        outputMatFile (1,1) string = fullfile(pwd, 'qwen_hf_quant_matlab', 'qwen_hf_quant_params.mat')
        options.LocalFilesOnly (1,1) logical = true
        options.TrustRemoteCode (1,1) logical = true
        options.HFEndpoint (1,1) string = "https://hf-mirror.com"
    end

    projectRoot = fileparts(fileparts(mfilename('fullpath')));
    pyExe = fullfile(projectRoot, '.venv', 'bin', 'python');
    scriptPath = fullfile(projectRoot, 'tools', 'prepare_qwen_hf_quant.py');

    if ~exist(pyExe, 'file')
        error('qwen2_quant:prepare_hf_quant_matlab:PythonNotFound', 'Missing python executable: %s', pyExe);
    end
    if ~exist(scriptPath, 'file')
        error('qwen2_quant:prepare_hf_quant_matlab:ScriptNotFound', 'Missing script: %s', scriptPath);
    end

    outDir = fileparts(outputMatFile);
    if ~isempty(outDir) && ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    if strlength(options.HFEndpoint) > 0
        setenv('HF_ENDPOINT', char(options.HFEndpoint));
    end

    cmd = sprintf('"%s" "%s" --model "%s" --output "%s"', ...
        pyExe, scriptPath, char(modelNameOrPath), char(outputMatFile));
    if options.LocalFilesOnly
        cmd = cmd + " --local-files-only";
    end
    if options.TrustRemoteCode
        cmd = cmd + " --trust-remote-code";
    end

    [status, out] = system(cmd);
    if status ~= 0 || ~exist(outputMatFile, 'file')
        error('qwen2_quant:prepare_hf_quant_matlab:Failed', ...
            'Export failed. Output:\n%s', out);
    end

    matFile = outputMatFile;
end
