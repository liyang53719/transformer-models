function parameters = load_hf_quant_matlab(filepath)
% LOAD_HF_QUANT_MATLAB Load GPTQ-exported MAT params into qwen2_quant parameter struct.

    raw = load(filepath);

    parameters = struct();
    parameters.Hyperparameters = struct();
    parameters.Weights = struct();

    hpFields = {'NumLayers', 'HiddenSize', 'NumHeads', 'NumKVHeads', 'HeadDim', 'VocabSize', 'RopeTheta'};
    for i = 1:numel(hpFields)
        if isfield(raw, hpFields{i})
            parameters.Hyperparameters.(hpFields{i}) = double(raw.(hpFields{i}));
        end
    end
    if ~isfield(parameters.Hyperparameters, 'RopeTheta')
        parameters.Hyperparameters.RopeTheta = 1000000.0;
    end

    if isfield(raw, 'embed_tokens')
        parameters.Weights.embed_tokens = dlarray(single(raw.embed_tokens));
    end
    if isfield(raw, 'norm')
        parameters.Weights.norm = dlarray(single(asColumn(raw.norm)));
    end
    if isfield(raw, 'lm_head_quant_type') || isfield(raw, 'lm_head_qweight')
        parameters.Weights.lm_head = buildGptqLinear(raw, 'lm_head');
    elseif isfield(raw, 'lm_head')
        parameters.Weights.lm_head = dlarray(single(raw.lm_head));
    end

    numLayers = parameters.Hyperparameters.NumLayers;
    for layerIdx = 0:numLayers-1
        layerName = sprintf('h%d', layerIdx);
        pfx = sprintf('layer_%d_', layerIdx);

        layerWeights = struct();

        layerWeights.self_attn_q_proj = buildLinear(raw, [pfx 'self_attn_q_proj']);
        layerWeights.self_attn_k_proj = buildLinear(raw, [pfx 'self_attn_k_proj']);
        layerWeights.self_attn_v_proj = buildLinear(raw, [pfx 'self_attn_v_proj']);
        layerWeights.self_attn_o_proj = buildLinear(raw, [pfx 'self_attn_o_proj']);

        if isfield(raw, [pfx 'self_attn_q_bias']) && ~linearHasInternalBias(layerWeights.self_attn_q_proj)
            layerWeights.self_attn_q_bias = dlarray(single(asColumn(raw.([pfx 'self_attn_q_bias']))));
        end
        if isfield(raw, [pfx 'self_attn_k_bias']) && ~linearHasInternalBias(layerWeights.self_attn_k_proj)
            layerWeights.self_attn_k_bias = dlarray(single(asColumn(raw.([pfx 'self_attn_k_bias']))));
        end
        if isfield(raw, [pfx 'self_attn_v_bias']) && ~linearHasInternalBias(layerWeights.self_attn_v_proj)
            layerWeights.self_attn_v_bias = dlarray(single(asColumn(raw.([pfx 'self_attn_v_bias']))));
        end
        if isfield(raw, [pfx 'self_attn_o_bias']) && ~linearHasInternalBias(layerWeights.self_attn_o_proj)
            layerWeights.self_attn_o_bias = dlarray(single(asColumn(raw.([pfx 'self_attn_o_bias']))));
        end

        layerWeights.mlp_gate_proj = buildLinear(raw, [pfx 'mlp_gate_proj']);
        layerWeights.mlp_up_proj = buildLinear(raw, [pfx 'mlp_up_proj']);
        layerWeights.mlp_down_proj = buildLinear(raw, [pfx 'mlp_down_proj']);

        layerWeights.input_layernorm = dlarray(single(asColumn(raw.([pfx 'input_layernorm']))));
        layerWeights.post_attention_layernorm = dlarray(single(asColumn(raw.([pfx 'post_attention_layernorm']))));

        parameters.Weights.(layerName) = layerWeights;
    end
end

function w = buildLinear(raw, prefix)
    if isfield(raw, [prefix '_qweight'])
        w = buildGptqLinear(raw, prefix);
        return;
    end

    if ~isfield(raw, prefix)
        error('qwen2_quant:load_hf_quant_matlab:MissingLinear', ...
            'Missing linear weight for %s', prefix);
    end

    w = dlarray(single(raw.(prefix)));
end

function w = buildGptqLinear(raw, prefix)
    required = {'qweight', 'qzeros', 'scales', 'bits', 'group_size', 'in_features', 'out_features'};
    for i = 1:numel(required)
        fn = [prefix '_' required{i}];
        if ~isfield(raw, fn)
            error('qwen2_quant:load_hf_quant_matlab:MissingGptqField', ...
                'Missing GPTQ field: %s', fn);
        end
    end

    w = struct();
    quantType = 'gptq_int4';
    if isfield(raw, [prefix '_quant_type'])
        qt = raw.([prefix '_quant_type']);
        if iscell(qt)
            quantType = string(qt{1});
        else
            quantType = string(qt);
        end
    end
    w.QuantType = upper(char(quantType));
    w.qweight = int32(raw.([prefix '_qweight']));
    w.qzeros = int32(raw.([prefix '_qzeros']));
    w.scales = single(raw.([prefix '_scales']));
    w.bits = double(raw.([prefix '_bits']));
    w.group_size = double(raw.([prefix '_group_size']));
    w.in_features = double(raw.([prefix '_in_features']));
    w.out_features = double(raw.([prefix '_out_features']));
    if isfield(raw, [prefix '_g_idx'])
        w.g_idx = int32(raw.([prefix '_g_idx']));
    else
        w.g_idx = int32(floor((0:w.in_features-1) ./ w.group_size));
    end
    if isfield(raw, [prefix '_bias'])
        w.bias = single(asColumn(raw.([prefix '_bias'])));
    end
end

function y = asColumn(x)
    y = x;
    if isvector(y) && size(y, 2) > 1
        y = y(:);
    end
end

function tf = linearHasInternalBias(w)
    tf = isstruct(w) && isfield(w, 'bias') && ~isempty(w.bias);
end
