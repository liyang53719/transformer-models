function [X_out, past_key_value, X_out_packed] = quantized_attentionGQA(X, past_key_value, weights, freqs_cis, hyperParameters, cfg)
% quantized_attentionGQA   Grouped Query Attention with quantized linear projections

    import qwen2_quant.layer.quantized_matmul

    if nargin < 6 || ~isstruct(cfg)
        error('quantized_attentionGQA:MissingConfig', ...
            'cfg is required and must be passed from top-level RuntimeConfig.');
    end
    useQuantSim = strcmpi(string(cfg.LinearMode), "gptq_int4_quant_sim");
    usePackedFullChain = useQuantSim && getCfgBool(cfg, 'EnablePackedFullChain', false);
    useQuantSimOutputPack = useQuantSim && isInt4PackedWeight(weights.o_proj);

    isPackedActivation = isstruct(X) && isfield(X, 'Q') && isfield(X, 'Scale') && isfield(X, 'Bias');
    if isPackedActivation
        if ~isfield(X, 'OriginalSize') || numel(X.OriginalSize) ~= 3
            error('quantized_attentionGQA:InvalidPackedActivation', ...
                'Packed activation must include OriginalSize=[hidden,seq,batch].');
        end
        hiddenSize = X.OriginalSize(1);
        seqLen = X.OriginalSize(2);
        batchSize = X.OriginalSize(3);
        X2_packed = struct('Q', int8(X.Q), 'Scale', single(X.Scale), 'Bias', single(X.Bias));
    else
        [hiddenSize, seqLen, batchSize] = size(X);
        X2 = reshape(X, hiddenSize, []);
    end

    numHeads = hyperParameters.NumHeads;
    numKVHeads = hyperParameters.NumKVHeads;
    headDim = hyperParameters.HeadDim;

    if isPackedActivation
        [xq, xq_packed] = quantized_matmul(weights.q_proj, X2_packed, cfg);
        [xk, xk_packed] = quantized_matmul(weights.k_proj, X2_packed, cfg);
        [xv, xv_packed] = quantized_matmul(weights.v_proj, X2_packed, cfg);
    else
        [xq, xq_packed] = quantized_matmul(weights.q_proj, X2, cfg);
        [xk, xk_packed] = quantized_matmul(weights.k_proj, X2, cfg);
        [xv, xv_packed] = quantized_matmul(weights.v_proj, X2, cfg);
    end

    if useQuantSim
        xq = dequantizePackedOrFloat(xq_packed, xq);
        xk = dequantizePackedOrFloat(xk_packed, xk);
        xv = dequantizePackedOrFloat(xv_packed, xv);
    end
    xq = reshape(xq, numHeads*headDim, seqLen, batchSize);
    xk = reshape(xk, numKVHeads*headDim, seqLen, batchSize);
    xv = reshape(xv, numKVHeads*headDim, seqLen, batchSize);

    if isfield(weights, 'q_bias'), xq = xq + weights.q_bias; end
    if isfield(weights, 'k_bias'), xk = xk + weights.k_bias; end
    if isfield(weights, 'v_bias'), xv = xv + weights.v_bias; end

    xq = reshape(xq, [headDim, numHeads, seqLen, batchSize]);
    xk = reshape(xk, [headDim, numKVHeads, seqLen, batchSize]);
    xv = reshape(xv, [headDim, numKVHeads, seqLen, batchSize]);

    [xq, xk] = transformer.layer.RoPE(xq, xk, freqs_cis);

    if usePackedFullChain
        xq2 = reshape(xq, headDim*numHeads, []);
        xk2 = reshape(xk, headDim*numKVHeads, []);
        xv2 = reshape(xv, headDim*numKVHeads, []);
        xq2 = dequantizePackedOrFloat(quantizeAffineInt8ByHeadGroup2D(xq2, headDim), xq2);
        xk2 = dequantizePackedOrFloat(quantizeAffineInt8ByHeadGroup2D(xk2, headDim), xk2);
        xv2 = dequantizePackedOrFloat(quantizeAffineInt8ByHeadGroup2D(xv2, headDim), xv2);
        xq = reshape(xq2, [headDim, numHeads, seqLen, batchSize]);
        xk = reshape(xk2, [headDim, numKVHeads, seqLen, batchSize]);
        xv = reshape(xv2, [headDim, numKVHeads, seqLen, batchSize]);
    end

    if ~isempty(past_key_value)
        keys = cat(3, past_key_value.keys, xk);
        values = cat(3, past_key_value.values, xv);
    else
        keys = xk;
        values = xv;
    end

    past_key_value.keys = keys;
    past_key_value.values = values;

    n_rep = numHeads / numKVHeads;
    if n_rep > 1
        keys = repelem(keys, 1, n_rep);
        values = repelem(values, 1, n_rep);
    end

    xq_for_mul = reshape(permute(xq, [3, 1, 2, 4]), seqLen, headDim, []);
    keys_for_mul = reshape(permute(keys, [1, 3, 2, 4]), headDim, [], numHeads*batchSize);
    scores = pagemtimes(xq_for_mul, keys_for_mul) / sqrt(headDim);

    scores = transformer.layer.Mask(scores, -1e4);
    attn_weights = transformer.layer.Softmax(scores, 2);

    if usePackedFullChain
        attn_shape = size(attn_weights);
        attn_weights2 = reshape(attn_weights, attn_shape(1), []);
        attn_weights2 = dequantizePackedOrFloat(quantizeAffineInt8PerCol2D(attn_weights2), attn_weights2);
        attn_weights = reshape(attn_weights2, attn_shape);
    end

    values_for_mul = reshape(permute(values, [3, 1, 2, 4]), size(values,3), headDim, []);
    attn_output = pagemtimes(attn_weights, values_for_mul);
    attn_output = reshape(attn_output, seqLen, headDim, numHeads, batchSize);
    attn_output = permute(attn_output, [2, 3, 1, 4]);
    attn_output_cat = reshape(attn_output, headDim*numHeads, seqLen*batchSize);

    if useQuantSimOutputPack
        o_in = packGroupedAffineInt8Activation2D(attn_output_cat, headDim);
        [X_out, X_out_packed] = quantized_matmul(weights.o_proj, o_in, cfg);
    else
        [X_out, X_out_packed] = quantized_matmul(weights.o_proj, attn_output_cat, cfg);
    end
    X_out = reshape(X_out, hiddenSize, seqLen, batchSize);
    if isfield(weights, 'o_bias'), X_out = X_out + weights.o_bias; end
end

function tf = isInt4PackedWeight(W)
    tf = isstruct(W) && isfield(W, 'QuantType') && ...
        (strcmpi(string(W.QuantType), "GPTQ_INT4") || strcmpi(string(W.QuantType), "AWQ_INT4"));
end

function packed = packGroupedAffineInt8Activation2D(x2, groupSize)
% x2: [hiddenSize, n_cols]
    [hiddenSize, n_cols] = size(x2);
    x2 = single(x2);

    if groupSize <= 0 || mod(hiddenSize, groupSize) ~= 0
        error('quantized_attentionGQA:InvalidGroupSize', ...
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

    packed = struct('Q', q2, 'Scale', s2, 'Bias', b2);
end

function x = dequantizePackedOrFloat(packed, fallback)
    if isstruct(packed) && isfield(packed, 'Q') && isfield(packed, 'Scale') && isfield(packed, 'Bias')
        x = dequantizePacked2D(packed);
    else
        x = single(fallback);
    end
end

function x2 = dequantizePacked2D(packed)
    q = single(packed.Q);
    s = single(packed.Scale);
    b = single(packed.Bias);
    [r, c] = size(q);

    if isscalar(s)
        x2 = q .* s + single(b);
        return;
    end

    if isrow(s) && numel(s) == c
        x2 = q .* s + b;
        return;
    end

    if size(s, 2) ~= c
        error('quantized_attentionGQA:ShapeMismatch', ...
            'Packed scale columns mismatch: expected %d, got %d.', c, size(s, 2));
    end

    if size(s, 1) == r
        x2 = q .* s + b;
        return;
    end

    if mod(r, size(s, 1)) ~= 0
        error('quantized_attentionGQA:ShapeMismatch', ...
            'Packed grouped rows mismatch: rows=%d, groups=%d.', r, size(s, 1));
    end

    groupSize = r / size(s, 1);
    x2 = zeros(r, c, 'single');
    for g = 1:size(s, 1)
        rows = (g-1)*groupSize + (1:groupSize);
        x2(rows, :) = q(rows, :) .* s(g, :) + b(g, :);
    end
end

function packed = quantizeAffineInt8ByHeadGroup2D(x2, groupSize)
    [hiddenSize, n_cols] = size(x2);
    x2 = single(x2);
    if groupSize <= 0 || mod(hiddenSize, groupSize) ~= 0
        error('quantized_attentionGQA:InvalidGroupSize', ...
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

    packed = struct('Q', q2, 'Scale', s2, 'Bias', b2);
end

function packed = quantizeAffineInt8PerCol2D(x2)
    x2 = single(x2);
    cmin = min(x2, [], 1);
    cmax = max(x2, [], 1);
    span = cmax - cmin;

    scale = single(span / 254.0);
    tiny = span < eps('single');
    scale(tiny) = single(1);
    bias = single((cmax + cmin) / 2.0);

    q = int8(max(min(round((x2 - bias) ./ scale), 127), -127));
    packed = struct('Q', q, 'Scale', scale, 'Bias', bias);
end

function tf = getCfgBool(cfg, fieldName, defaultVal)
    tf = defaultVal;
    if isfield(cfg, fieldName)
        tf = logical(cfg.(fieldName));
    end
end
