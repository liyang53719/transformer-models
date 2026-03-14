%% TestQwen2AWQQuantSummarize
% Compare Python-reference AWQ and MATLAB-native AWQ summarize outputs.

promptText = "The process of photosynthesis is how plants convert light energy into chemical energy. Chlorophyll absorbs sunlight and uses it to convert carbon dioxide and water into glucose.";
promptTemplate = "Summarize this: %s Summary:";
maxNewTokens = 30;
topK = 1;

awqPath = fullfile('qwen_hf_quant', 'Qwen2.5-1.5B-Instruct-AWQ');
if ~exist(awqPath, 'dir')
    error('Missing AWQ model folder: %s', awqPath);
end

matOut = fullfile('qwen_hf_quant_matlab', 'Qwen2.5-1.5B-Instruct-AWQ_params.mat');
needExport = ~exist(matOut, 'file');
if ~needExport
    vars = whos('-file', matOut);
    names = string({vars.name});
    hasLinear = any(names == "layer_0_self_attn_q_proj") || any(names == "layer_0_self_attn_q_proj_qweight");
    hasQuantType = any(names == "layer_0_self_attn_q_proj_quant_type");
    needExport = ~(hasLinear && hasQuantType);
end
if needExport
    fprintf('Exporting MATLAB params from AWQ model...\n');
    qwen2_quant.prepare_hf_quant_matlab(awqPath, matOut, ...
        'LocalFilesOnly', true, 'TrustRemoteCode', true, 'HFEndpoint', "https://hf-mirror.com");
end

fprintf('\n=== Python reference branch (AWQ) ===\n');
t1 = tic;
outPy = runPythonReferenceAWQ(awqPath, promptText, promptTemplate, maxNewTokens);
tPy = toc(t1);
fprintf('Python-ref Time: %.3fs\n', tPy);
fprintf('Python-ref Out : %s\n', outPy);

fprintf('\n=== MATLAB-native branch (AWQ) ===\n');
mdlM = struct();
mdlM.Parameters = qwen2_quant.load_hf_quant(awqPath, ...
    'Engine', "matlab_native", ...
    'MatParamsFile', matOut, ...
    'TokenizerPath', awqPath, ...
    'LocalFilesOnly', true, 'AutoRetryOnline', false);
enablePackedFullChain = false;
mdlM.Parameters.RuntimeConfig = struct('LinearMode', 'gptq_int4_quant_sim', ...
    'EnablePackedFullChain', enablePackedFullChain);
fprintf('LinearMode: %s\n', string(mdlM.Parameters.RuntimeConfig.LinearMode));
fprintf('EnablePackedFullChain: %s\n', string(enablePackedFullChain));
t2 = tic;
outM = string(generateSummary_Qwen2_quant(mdlM, promptText, ...
    'TopK', topK, 'MaxNewTokens', maxNewTokens, 'PromptTemplate', promptTemplate));
tM = toc(t2);
fprintf('MATLAB-native Time: %.3fs\n', tM);
fprintf('MATLAB-native Out : %s\n', outM);

eq = strcmp(outPy, outM);
fprintf('\nExact Equal (Python-ref vs MATLAB-native): %s\n', string(eq));
if ~eq
    warning('Parity check failed: outputs are not exactly equal.');
end

function out = runPythonReferenceAWQ(modelPath, text, promptTmpl, maxNewTokens)
    prompt = sprintf(promptTmpl, text);
    pyFile = [tempname, '.py'];
    outFile = [tempname, '.json'];
    cleanupObj = onCleanup(@()cleanupTemp(pyFile, outFile)); %#ok<NASGU>

    promptEsc = strrep(prompt, '\\', '\\\\');
    promptEsc = strrep(promptEsc, '"', '\\"');
    modelEsc = strrep(modelPath, '\\', '/');
    outEsc = strrep(outFile, '\\', '/');

    pyLines = {
        'import json'
        'import torch'
        'from transformers import AutoTokenizer'
        'from awq import AutoAWQForCausalLM'
        sprintf('model_id = r"%s"', modelEsc)
        sprintf('prompt = "%s"', promptEsc)
        sprintf('out_file = r"%s"', outEsc)
        'tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)'
        'mdl = AutoAWQForCausalLM.from_quantized(model_id, fuse_layers=False)'
        'device = next(mdl.model.parameters()).device'
        'ids = tok.encode(prompt, add_special_tokens=False)'
        'input_ids = torch.tensor([ids], dtype=torch.long, device=device)'
        'with torch.no_grad():'
        '    out = mdl.model(input_ids=input_ids, use_cache=True, return_dict=True)'
        'past = out.past_key_values'
        'next_id = int(torch.argmax(out.logits[0, -1, :]).item())'
        'generated = [next_id]'
        sprintf('max_new_tokens = %d', maxNewTokens)
        'eos_id = tok.eos_token_id if tok.eos_token_id is not None else -1'
        'pad_id = tok.pad_token_id if tok.pad_token_id is not None else -1'
        'for _ in range(max_new_tokens):'
        '    step_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)'
        '    with torch.no_grad():'
        '        out = mdl.model(input_ids=step_ids, past_key_values=past, use_cache=True, return_dict=True)'
        '    past = out.past_key_values'
        '    next_id = int(torch.argmax(out.logits[0, -1, :]).item())'
        '    if next_id in (eos_id, pad_id, 151645):'
        '        break'
        '    generated.append(next_id)'
        'summary = tok.decode(generated, skip_special_tokens=True)'
        'with open(out_file, "w", encoding="utf-8") as f:'
        '    json.dump({"summary": summary}, f, ensure_ascii=False)'
    };

    fid = fopen(pyFile, 'w');
    assert(fid > 0, 'Failed to create temp python file.');
    fprintf(fid, '%s\n', pyLines{:});
    fclose(fid);

    pyExe = fullfile(pwd, '.venv', 'bin', 'python');
    if ~exist(pyExe, 'file')
        pyExe = 'python';
    end

    cmd = sprintf('"%s" "%s"', pyExe, pyFile);
    [status, cmdOut] = system(cmd);
    if status ~= 0 || ~exist(outFile, 'file')
        error('Python AWQ reference failed:\n%s', cmdOut);
    end

    resp = jsondecode(fileread(outFile));
    out = string(resp.summary);
end

function cleanupTemp(pyFile, outFile)
    if exist(pyFile, 'file'), delete(pyFile); end
    if exist(outFile, 'file'), delete(outFile); end
end
