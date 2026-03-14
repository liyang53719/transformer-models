function [h, present, h_packed] = block(h, past, weights, hyperParameters, freqs_cis, cfg)
% block   Transformer block for Qwen2 with Quantization Support
%
%   [h, present] = block(h, past, weights, hyperParameters, freqs_cis)
%
%   This is a quantization-aware version that uses quantized_matmul
%   for weight matrix operations.

    import qwen2_quant.layer.*
    import transformer.layer.rmsNormalization
    if nargin < 6 || ~isstruct(cfg)
        error('qwen2_quant:block:MissingConfig', ...
            'cfg is required and must be passed from top-level RuntimeConfig.');
    end
    
    useQuantSim = strcmpi(string(cfg.LinearMode), "gptq_int4_quant_sim");
    usePackedFullChain = useQuantSim && getCfgBool(cfg, 'EnablePackedFullChain', false);
    h = dequantizePackedActivation(h);
    resid = h;
    h_packed = [];
    
    % 1. Input Norm
    if isa(weights.input_layernorm, 'qwen2_quant.internal.quantized_weight')
        norm_weight = weights.input_layernorm.dequantize();
    else
        norm_weight = extractdata(weights.input_layernorm);
    end
    h = rmsNormalization(h, norm_weight, 1e-6);
    h_norm_packed = [];
    if usePackedFullChain
        h_norm_packed = packAffineInt8Activation(h);
    end
    if cfg.TracePrecision
        qwen2_quant.internal.precision_trace('log', 'block.input_norm.output', h);
    end
    
    % 2. Attention
    attnWeights.q_proj = weights.self_attn_q_proj;
    attnWeights.k_proj = weights.self_attn_k_proj;
    attnWeights.v_proj = weights.self_attn_v_proj;
    attnWeights.o_proj = weights.self_attn_o_proj;

    % Support for Qwen bias
    if isfield(weights, 'self_attn_q_bias'), attnWeights.q_bias = weights.self_attn_q_bias; end
    if isfield(weights, 'self_attn_k_bias'), attnWeights.k_bias = weights.self_attn_k_bias; end
    if isfield(weights, 'self_attn_v_bias'), attnWeights.v_bias = weights.self_attn_v_bias; end
    if isfield(weights, 'self_attn_o_bias'), attnWeights.o_bias = weights.self_attn_o_bias; end
    
    attnInput = h;
    qProjQuantType = getQuantType(attnWeights.q_proj);
    usePreAttnPack = useQuantSim && (qProjQuantType == "GPTQ_INT4" || qProjQuantType == "AWQ_INT4");
    if usePreAttnPack
        if isa(h, 'dlarray')
            h_data = single(extractdata(h));
        else
            h_data = single(h);
        end
        if qProjQuantType == "GPTQ_INT4"
            attnInput = packGroupedAffineInt8Activation(h_data, hyperParameters.HeadDim);
        elseif ~isempty(h_norm_packed)
            attnInput = h_norm_packed;
        else
            attnInput = packAffineInt8Activation(h_data);
        end
    elseif usePackedFullChain && ~isempty(h_norm_packed)
        attnInput = h_norm_packed;
    end

    [h_attn, present, h_attn_packed] = quantized_attentionGQA(attnInput, past, attnWeights, freqs_cis, hyperParameters, cfg);
    h_attn = dequantizePackedOrFloat(h_attn_packed, h_attn);
    
    h = resid + h_attn;
    if usePackedFullChain
        h = dequantizePackedActivation(packAffineInt8Activation(h));
    end
    if cfg.TracePrecision
        qwen2_quant.internal.precision_trace('log', 'block.attn_residual.output', h);
    end
    
    % 3. Post Attention Norm
    resid = h;
    if isa(weights.post_attention_layernorm, 'qwen2_quant.internal.quantized_weight')
        norm_weight = weights.post_attention_layernorm.dequantize();
    else
        norm_weight = extractdata(weights.post_attention_layernorm);
    end
    h = rmsNormalization(h, norm_weight, 1e-6);
    h_post_norm_packed = [];
    if usePackedFullChain
        h_post_norm_packed = packAffineInt8Activation(h);
    end
    if cfg.TracePrecision
        qwen2_quant.internal.precision_trace('log', 'block.post_attn_norm.output', h);
    end
    
    % 4. MLP
    ffnWeights.gate_proj = weights.mlp_gate_proj;
    ffnWeights.up_proj   = weights.mlp_up_proj;
    ffnWeights.down_proj = weights.mlp_down_proj;
    
    if usePackedFullChain && ~isempty(h_post_norm_packed)
        [h_ffn, h_ffn_packed] = gatedMLP(h_post_norm_packed, ffnWeights, cfg);
    else
        [h_ffn, h_ffn_packed] = gatedMLP(h, ffnWeights, cfg);
    end
    h_ffn = dequantizePackedOrFloat(h_ffn_packed, h_ffn);
    
    h = resid + h_ffn;
    if usePackedFullChain
        h_packed = packAffineInt8Activation(h);
    end
    if cfg.TracePrecision
        qwen2_quant.internal.precision_trace('log', 'block.ffn_residual.output', h);
    end

end

function x = dequantizePackedActivation(X)
    if isstruct(X) && isfield(X, 'Q') && isfield(X, 'Scale') && isfield(X, 'Bias')
        q = single(X.Q);
        s = single(X.Scale);
        b = single(X.Bias);
        [r, c] = size(q);

        if isscalar(s)
            x2 = q .* s + b;
        elseif isrow(s) && numel(s) == c
            x2 = q .* s + b;
        elseif size(s, 2) == c && size(s, 1) == r
            x2 = q .* s + b;
        elseif size(s, 2) == c && mod(r, size(s, 1)) == 0
            groupSize = r / size(s, 1);
            x2 = zeros(r, c, 'single');
            for g = 1:size(s, 1)
                rows = (g-1)*groupSize + (1:groupSize);
                x2(rows, :) = q(rows, :) .* s(g, :) + b(g, :);
            end
        else
            error('qwen2_quant:block:ShapeMismatch', 'Unsupported packed activation shape.');
        end

        if isfield(X, 'OriginalSize') && numel(X.OriginalSize) == 3
            x = reshape(x2, X.OriginalSize);
        else
            x = x2;
        end
    else
        x = X;
    end
end

function x = dequantizePackedOrFloat(packed, fallback)
    if isstruct(packed) && isfield(packed, 'Q') && isfield(packed, 'Scale') && isfield(packed, 'Bias')
        x = dequantizePackedActivation(packed);
    else
        x = single(fallback);
    end
end

function quantType = getQuantType(W)
    if isstruct(W) && isfield(W, 'QuantType')
        quantType = upper(string(W.QuantType));
    else
        quantType = "";
    end
end

function packed = packGroupedAffineInt8Activation(x, groupSize)
% x: [hiddenSize, seqLen, batchSize] single
    [hiddenSize, seqLen, batchSize] = size(x);
    n_cols = seqLen * batchSize;
    x2 = reshape(single(x), hiddenSize, n_cols);

    if groupSize <= 0 || mod(hiddenSize, groupSize) ~= 0
        error('qwen2_quant:block:InvalidGroupSize', ...
            'HeadDim group size must divide hidden size. hidden=%d, group=%d', hiddenSize, groupSize);
    end

    numGroups = hiddenSize / groupSize;
    q2 = zeros(hiddenSize, n_cols, 'int8');
    s2 = zeros(numGroups, n_cols, 'single');
    b2 = zeros(numGroups, n_cols, 'single');

    for g = 1:numGroups
        rows = (g-1)*groupSize + (1:groupSize);
        chunk = x2(rows, :);

        cmin = min(chunk, [], 1);
        cmax = max(chunk, [], 1);
        span = cmax - cmin;

        scale = single(span / 254.0);
        tiny = span < eps('single');
        scale(tiny) = single(1);
        bias = single((cmax + cmin) / 2.0);

        q_chunk = int8(max(min(round((chunk - bias) ./ scale), 127), -127));

        q2(rows, :) = q_chunk;
        s2(g, :) = scale;
        b2(g, :) = bias;
    end

    packed = struct();
    packed.Q = q2;
    packed.Scale = s2;
    packed.Bias = b2;
    packed.OriginalSize = [hiddenSize, seqLen, batchSize];
end

function packed = packAffineInt8Activation(x)
% x: [hiddenSize, seqLen, batchSize] single
    [hiddenSize, seqLen, batchSize] = size(x);
    n_cols = seqLen * batchSize;
    x2 = reshape(single(x), hiddenSize, n_cols);

    cmin = min(x2, [], 1);
    cmax = max(x2, [], 1);
    span = cmax - cmin;

    scale = single(span / 254.0);
    tiny = span < eps('single');
    scale(tiny) = single(1);
    bias = single((cmax + cmin) / 2.0);

    q = int8(max(min(round((x2 - bias) ./ scale), 127), -127));

    packed = struct();
    packed.Q = q;
    packed.Scale = scale;
    packed.Bias = bias;
    packed.OriginalSize = [hiddenSize, seqLen, batchSize];
end

function tf = getCfgBool(cfg, fieldName, defaultVal)
    tf = defaultVal;
    if isfield(cfg, fieldName)
        tf = logical(cfg.(fieldName));
    end
end
