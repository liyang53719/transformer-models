function [logits, layerStates, debugInfo] = model(X, parameters, layerStates, options)
% model   Qwen2 Model Forward Pass with Quantization Support
%
%   [logits, layerStates] = model(X, parameters, layerStates)
%
%   This is a quantization-aware version of qwen2.model that supports
%   quantized weights via runtime dequantization.
%
%   Inputs:
%       X           - Input tokens [1, seqLen*batch] or [1, seqLen, batch]
%       parameters  - Struct containing .Weights and .Hyperparameters
%       layerStates - Struct array of length NumLayers (optional, for KV cache)
%
%   Outputs:
%       logits      - [vocabSize, seqLen, batch]
%       layerStates - Updated KV cache

    if nargin < 3
        layerStates = [];
    end

    if nargin < 4
        options = struct();
    end

    import qwen2_quant.layer.*
    import transformer.layer.precomputeFreqsCis
    import transformer.layer.rmsNormalization

    runtimeCfg = defaultRuntimeConfig();
    if isfield(options, 'RuntimeConfig') && isstruct(options.RuntimeConfig)
        runtimeCfg = mergeStruct(runtimeCfg, options.RuntimeConfig);
    elseif isfield(parameters, 'RuntimeConfig') && isstruct(parameters.RuntimeConfig)
        runtimeCfg = mergeStruct(runtimeCfg, parameters.RuntimeConfig);
    end
    qwen2_quant.internal.precision_trace('reset');
    traceTensors = isfield(runtimeCfg, 'TraceTensors') && logical(runtimeCfg.TraceTensors);

    tensorTrace = struct();
    if traceTensors
        tensorTrace.blocks = cell(0, 1);
    end
    
    % Unpack Parameters
    if isfield(parameters, 'Hyperparameters')
        hp = parameters.Hyperparameters;
        weights = parameters.Weights;
    else
        error('model:InvalidParameters', 'Parameters must have Hyperparameters field');
    end
    
    numLayers = hp.NumLayers;
    headDim = hp.HeadDim;
    ropeTheta = hp.RopeTheta;
    
    % 1. Embeddings
    if ismatrix(X) && size(X,1) == 1
         [~, fullLen] = size(X);
         batchSize = 1; 
         seqLen = fullLen;
    else
         [~, seqLen, batchSize] = size(X);
         X = reshape(X, 1, []);
    end
    
    % Token IDs from tokenizer are 0-based, convert to 1-based for MATLAB indexing
    idx = double(X) + 1;
    
    % Access embed_tokens (handle quantized case)
    if isa(weights.embed_tokens, 'qwen2_quant.internal.quantized_weight')
        embed_weights = weights.embed_tokens.dequantize();
    else
        embed_weights = extractdata(weights.embed_tokens);
    end
    
    Z = embed_weights(:, idx);
    Z = reshape(Z, [], seqLen, batchSize); 
    Z = single(Z);
    if traceTensors
        tensorTrace.embed = Z;
    end
    if runtimeCfg.TracePrecision
        qwen2_quant.internal.precision_trace('log', 'embed.output', Z);
    end
    
    % 2. RoPE frequencies
    startPos = 1;
    if exist('layerStates', 'var') && ~isempty(layerStates)
        if isstruct(layerStates) 
             if isfield(layerStates, 'keys')
                startPos = size(layerStates(1).keys, 3) + 1;
             end
        elseif iscell(layerStates) && ~isempty(layerStates{1})
             startPos = size(layerStates{1}.keys, 3) + 1;
        end
    end
    
    maxSeqLen = startPos + seqLen + 128;
    freqs_cis = precomputeFreqsCis(headDim, maxSeqLen, ropeTheta);
    freqs_cis = complex(single(real(freqs_cis)), single(imag(freqs_cis)));
    
    currentFreqs = freqs_cis(:, startPos:startPos+seqLen-1);
    useQuantSim = strcmpi(string(runtimeCfg.LinearMode), "gptq_int4_quant_sim");
    usePackedFullChain = useQuantSim && getCfgBool(runtimeCfg, 'EnablePackedFullChain', false);

    % Init layerStates if empty
    if isempty(layerStates)
        layerStates = cell(numLayers, 1);
    end
    
    % 3. Layers Loop
    if usePackedFullChain
        Z_work = packAffineInt8Activation3D(Z);
    else
        Z_work = Z;
    end

    for i = 1:numLayers
        layerName = sprintf('h%d', i-1);
        
        if isfield(weights, layerName)
             layerWeights = weights.(layerName);
        else 
             error('model:MissingLayer', 'Missing layer weights: %s', layerName);
        end

        state = layerStates{i};
        
        [Z, newState, Z_packed] = qwen2_quant.layer.block(Z_work, state, layerWeights, hp, currentFreqs, runtimeCfg);
        if usePackedFullChain && ~isempty(Z_packed)
            Z_work = Z_packed;
        else
            Z_work = Z;
        end
        if traceTensors
            tensorTrace.blocks{i, 1} = Z;
        end
        if runtimeCfg.TracePrecision
            qwen2_quant.internal.precision_trace('log', sprintf('block%d.output', i-1), Z);
        end
        
        layerStates{i} = newState;
    end
    
    % 4. Final Norm
    if usePackedFullChain
        Z = dequantizePackedActivation3D(Z_work);
    else
        Z = Z_work;
    end

    if isa(weights.norm, 'qwen2_quant.internal.quantized_weight')
        norm_weight = weights.norm.dequantize();
    else
        norm_weight = extractdata(weights.norm);
    end
    Z = rmsNormalization(Z, norm_weight, 1e-6);
    Z_norm_packed = [];
    if usePackedFullChain
        Z_norm_packed = packAffineInt8Activation3D(Z);
    end
    if traceTensors
        tensorTrace.final_norm = Z;
    end
    if runtimeCfg.TracePrecision
        qwen2_quant.internal.precision_trace('log', 'final_norm.output', Z);
    end
    
    % 5. Output Head
    if usePackedFullChain && ~isempty(Z_norm_packed)
        Z_flat = struct('Q', int8(Z_norm_packed.Q), 'Scale', single(Z_norm_packed.Scale), 'Bias', single(Z_norm_packed.Bias));
    else
        Z_flat = reshape(Z, size(Z,1), []);
    end
    Z_flat_float = dequantizePacked2DOrPass(Z_flat);
    
    if isa(weights.lm_head, 'qwen2_quant.internal.quantized_weight')
        lm_head_weight = weights.lm_head.dequantize();
        logits = lm_head_weight * Z_flat_float;
    elseif isstruct(weights.lm_head)
        [logits, ~] = qwen2_quant.layer.quantized_matmul(weights.lm_head, Z_flat, runtimeCfg);
    else
        logits = weights.lm_head * Z_flat_float;
    end
    
    vocabSize = size(logits, 1);
    logits = reshape(logits, vocabSize, seqLen, batchSize);
    if traceTensors
        tensorTrace.logits = logits;
    end

    if runtimeCfg.TracePrecision
        qwen2_quant.internal.precision_trace('log', 'logits.output', logits);
    end

    debugInfo = struct();
    debugInfo.RuntimeConfig = runtimeCfg;
    debugInfo.PrecisionTrace = qwen2_quant.internal.precision_trace('get');
    if traceTensors
        debugInfo.TensorTrace = tensorTrace;
    end

end

function packed = packAffineInt8Activation3D(x)
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
    packed = struct('Q', q, 'Scale', scale, 'Bias', bias, 'OriginalSize', [hiddenSize, seqLen, batchSize]);
end

function x = dequantizePackedActivation3D(packed)
    q = single(packed.Q);
    s = single(packed.Scale);
    b = single(packed.Bias);
    x2 = q .* s + b;
    x = reshape(x2, packed.OriginalSize);
end

function x2 = dequantizePacked2DOrPass(x)
    if isstruct(x) && isfield(x, 'Q') && isfield(x, 'Scale') && isfield(x, 'Bias')
        x2 = single(x.Q) .* single(x.Scale) + single(x.Bias);
    else
        x2 = x;
    end
end

function cfg = defaultRuntimeConfig()
    cfg = struct();
    cfg.LinearMode = 'float';
    cfg.TracePrecision = false;
    cfg.TraceTensors = false;
    cfg.Int8WeightScaleMode = 'per_row';
    cfg.Int8ActivationScaleMode = 'per_col';
    cfg.EnablePackedFullChain = false;
end

function tf = getCfgBool(cfg, fieldName, defaultVal)
    tf = defaultVal;
    if isfield(cfg, fieldName)
        tf = logical(cfg.(fieldName));
    end
end

function merged = mergeStruct(base, override)
    merged = base;
    fields = fieldnames(override);
    for i = 1:numel(fields)
        merged.(fields{i}) = override.(fields{i});
    end
end
