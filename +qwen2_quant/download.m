function modelPath = download(branch, destination, options)
% DOWNLOAD Download quantized model branches for +qwen2_quant.
%
%   qwen2_quant.download("gptq")
%   qwen2_quant.download("awq")

    arguments
        branch (1,1) string {mustBeMember(branch, ["gptq", "awq"])} = "gptq"
        destination (1,1) string = fullfile(pwd, 'qwen_hf_quant')
        options.UseMirror (1,1) logical = true
        options.HFEndpoint (1,1) string = "https://hf-mirror.com"
    end

    if ~exist(destination, 'dir')
        mkdir(destination);
    end

    i_ensure_project_pyenv();

    switch lower(branch)
        case "gptq"
            repoId = "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4";
            localDir = fullfile(destination, 'Qwen2.5-1.5B-Instruct-GPTQ-Int4');
        case "awq"
            repoId = "Qwen/Qwen2.5-1.5B-Instruct-AWQ";
            localDir = fullfile(destination, 'Qwen2.5-1.5B-Instruct-AWQ');
    end

    if exist(localDir, 'dir')
        modelPath = string(localDir);
        fprintf('Model already exists locally: %s\n', localDir);
        return;
    end

    fprintf('Downloading %s model: %s\n', upper(branch), repoId);
    fprintf('Destination: %s\n', localDir);

    try
        if count(py.sys.path, '') == 0
            insert(py.sys.path, int32(0), '');
        end

        os = py.importlib.import_module('os');
        if options.UseMirror && strlength(options.HFEndpoint) > 0
            os.putenv('HF_ENDPOINT', char(options.HFEndpoint));
            fprintf('Using HF endpoint: %s\n', char(options.HFEndpoint));
        else
            os.putenv('HF_ENDPOINT', '');
        end

        hf_hub = py.importlib.import_module('huggingface_hub');
        snapshot_download = py.getattr(hf_hub, 'snapshot_download');
        snapshot_download(repo_id=repoId, local_dir=string(localDir));

        modelPath = string(localDir);
        fprintf('Download complete: %s\n', modelPath);
    catch ME
        error('qwen2_quant:download:Failed', ...
            'Failed to download %s (%s). Python Error: %s', upper(branch), repoId, ME.message);
    end
end

function i_ensure_project_pyenv()
    thisFile = mfilename('fullpath');
    projectRoot = fileparts(fileparts(thisFile));
    venvPython = fullfile(projectRoot, '.venv', 'bin', 'python');

    if ~exist(venvPython, 'file')
        return;
    end

    pe = pyenv;
    if pe.Status == "NotLoaded"
        pyenv('Version', venvPython);
    elseif ~strcmp(pe.Executable, venvPython)
        warning('qwen2_quant:download:PythonMismatch', ...
            'MATLAB is using %s, expected %s for quant download path.', pe.Executable, venvPython);
    end
end
