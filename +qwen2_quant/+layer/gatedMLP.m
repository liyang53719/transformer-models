function [Y, Y_packed] = gatedMLP(X, weights, cfg)
% gatedMLP   Gated MLP (SwiGLU) with Quantization Support
%
%   Y = gatedMLP(X, weights)
%
%   Uses quantized_matmul for weight operations

    import qwen2_quant.layer.*
    if nargin < 3 || ~isstruct(cfg)
        error('qwen2_quant:gatedMLP:MissingConfig', ...
            'cfg is required and must be passed from top-level RuntimeConfig.');
    end
    
    if isPackedActivation(X)
        if ~isfield(X, 'OriginalSize') || numel(X.OriginalSize) ~= 3
            error('qwen2_quant:gatedMLP:InvalidPackedActivation', ...
                'Packed activation must include OriginalSize=[hidden,seq,batch].');
        end
        hiddenSize = X.OriginalSize(1);
        seqLen = X.OriginalSize(2);
        batchSize = X.OriginalSize(3);
        X_flat = [];
    else
        [hiddenSize, seqLen, batchSize] = size(X);
        X_flat = reshape(X, hiddenSize, []);
    end
    Y_packed = [];

    useQuantSim = strcmpi(string(cfg.LinearMode), "gptq_int4_quant_sim");
    usePackedFullChain = useQuantSim && getCfgBool(cfg, 'EnablePackedFullChain', false);
    usePackedInput = useQuantSim && isInt4PackedWeight(weights.gate_proj) && ...
        isInt4PackedWeight(weights.up_proj);

    X_for_proj = X_flat;
    if isPackedActivation(X)
        X_for_proj = struct('Q', int8(X.Q), 'Scale', single(X.Scale), 'Bias', single(X.Bias));
    elseif usePackedInput
        X_for_proj = packAffineInt8Activation2D(X_flat);
    end
    
    % Gate and Up projections
    [gate, gate_packed] = quantized_matmul(weights.gate_proj, X_for_proj, cfg);
    [up, up_packed] = quantized_matmul(weights.up_proj, X_for_proj, cfg);
    if useQuantSim
        gate = dequantizePackedOrFloat(gate_packed, gate);
        up = dequantizePackedOrFloat(up_packed, up);
    end
    if cfg.TracePrecision
        qwen2_quant.internal.precision_trace('log', 'mlp.gate_proj.output', gate);
        qwen2_quant.internal.precision_trace('log', 'mlp.up_proj.output', up);
    end
    
    % SwiGLU activation
    activated = swish(gate) .* up;
    activated_packed = [];
    if usePackedFullChain
        activated_packed = packAffineInt8Activation2D(activated);
        activated = dequantizePackedOrFloat(activated_packed, activated);
    end
    if cfg.TracePrecision
        qwen2_quant.internal.precision_trace('log', 'mlp.swiglu.output', activated);
    end
    
    % Down projection
    if usePackedFullChain && isInt4PackedWeight(weights.down_proj)
        [Y, Y_packed] = quantized_matmul(weights.down_proj, activated_packed, cfg);
    else
        [Y, Y_packed] = quantized_matmul(weights.down_proj, activated, cfg);
    end
    if cfg.TracePrecision
        qwen2_quant.internal.precision_trace('log', 'mlp.down_proj.output', Y);
    end
    
    Y = reshape(Y, [], seqLen, batchSize);
end

function tf = isPackedActivation(X)
    tf = isstruct(X) && isfield(X, 'Q') && isfield(X, 'Scale') && isfield(X, 'Bias');
end

function tf = isInt4PackedWeight(W)
    tf = isstruct(W) && isfield(W, 'QuantType') && ...
        (strcmpi(string(W.QuantType), "GPTQ_INT4") || strcmpi(string(W.QuantType), "AWQ_INT4"));
end

function packed = packAffineInt8Activation2D(x2)
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

function x2 = dequantizePackedOrFloat(packed, fallback)
    if isstruct(packed) && isfield(packed, 'Q') && isfield(packed, 'Scale') && isfield(packed, 'Bias')
        q = single(packed.Q);
        s = single(packed.Scale);
        b = single(packed.Bias);
        x2 = q .* s + b;
    else
        x2 = single(fallback);
    end
end

function y = swish(x)
    % Swish activation: x * sigmoid(x)
    y = x .* (1 ./ (1 + exp(-x)));
end

function tf = getCfgBool(cfg, fieldName, defaultVal)
    tf = defaultVal;
    if isfield(cfg, fieldName)
        tf = logical(cfg.(fieldName));
    end
end
