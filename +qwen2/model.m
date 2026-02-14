function [logits, layerStates] = model(X, parameters, layerStates)
% model   Qwen2 Model Forward Pass
%
%   [logits, layerStates] = model(X, parameters, layerStates)
%
%   Inputs:
%       X           - Input tokens [1, seqLen*batch] or [1, seqLen, batch]
%       parameters  - Struct containing .Weights and .Hyperparameters
%       layerStates - Struct array of length NumLayers (optional, for KV cache)
%                     Each element struct with .keys and .values
%
%   Outputs:
%       logits      - [vocabSize, seqLen, batch]
%       layerStates - Updated KV cache

    if nargin < 3
        layerStates = [];
    end

    import transformer.layer.*
    
    % Unpack Parameters
    if isfield(parameters, 'Hyperparameters')
        hp = parameters.Hyperparameters;
        weights = parameters.Weights;
    else
        % Compatible with old flat struct
        hp.NumLayers = double(parameters.NumLayers);
        hp.HeadDim = double(parameters.HeadDim);
        hp.NumHeads = double(parameters.NumHeads);
        hp.NumKVHeads = double(parameters.NumKVHeads);
        if isfield(parameters, 'RopeTheta')
             hp.RopeTheta = double(parameters.RopeTheta);
        else
             hp.RopeTheta = 1000000.0;
        end
        weights = parameters;
        % In old format, layer weights are mixed in 'weights'.
        % We need to handle both formats or migrate.
        % The logic below assumes the NEW hierarchical format from qwen2.load.
        % If 'weights' contains 'layer_0_...', convert on fly?
        % Let's assume user uses qwen2.load.
    end
    
    numLayers = hp.NumLayers;
    headDim = hp.HeadDim;
    numHeads = hp.NumHeads;
    numKVHeads = hp.NumKVHeads;
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
    
    idx = double(X) + 1; 
    
    % Access weights.embed_tokens
    Z = weights.embed_tokens(:, idx);
    
    % Debug Shape
    % disp("Model Debug: Z_embed size=" + mat2str(size(Z)) + " seqLen=" + seqLen);
    
    Z = reshape(Z, [], seqLen, batchSize); 
    Z = single(Z);
    
    % disp("Model Debug: Z_reshaped size=" + mat2str(size(Z)));

    % 2. RoPE
    % Compute frequencies based on absolute position in sequence
    % Offset by cache length
    startPos = 1;
    if exist('layerStates', 'var') && ~isempty(layerStates)
        % Assuming layerStates format matches: .keys
        if isstruct(layerStates) 
             if isfield(layerStates, 'keys')
                startPos = size(layerStates(1).keys, 3) + 1;
             end
        elseif iscell(layerStates) && ~isempty(layerStates{1}) % GPT-2 style cell array
             startPos = size(layerStates{1}.keys, 3) + 1;
        end
    end
    
    maxSeqLen = startPos + seqLen + 128; % Dynamic buffer
    freqs_cis = precomputeFreqsCis(headDim, maxSeqLen, ropeTheta);
    freqs_cis = complex(single(real(freqs_cis)), single(imag(freqs_cis)));
    
    currentFreqs = freqs_cis(:, startPos:startPos+seqLen-1);

    % Init layerStates if empty
    if isempty(layerStates)
        layerStates = cell(numLayers, 1);
    end
    
    % 3. Layers Loop
    for i = 1:numLayers
        layerName = sprintf('h%d', i-1);
        
        % Support old flat format for backward compatibility w/ test script
        if isfield(weights, layerName)
             layerWeights = weights.(layerName);
        else 
             % Fallback for flat struct (TestTinyLlamaExample uses this currently)
             % This is messy. Let's force proper loading in the future.
             % But for now, construct layerWeights on the fly from flat
             pfx = "layer_" + (i-1) + "_";
             layerWeights.input_layernorm = weights.(pfx+"input_layernorm");
             layerWeights.post_attention_layernorm = weights.(pfx+"post_attention_layernorm");
             layerWeights.self_attn_q_proj = weights.(pfx+"self_attn_q_proj");
             layerWeights.self_attn_k_proj = weights.(pfx+"self_attn_k_proj");
             layerWeights.self_attn_v_proj = weights.(pfx+"self_attn_v_proj");
             layerWeights.self_attn_o_proj = weights.(pfx+"self_attn_o_proj");
             if isfield(weights, pfx+"self_attn_q_bias"), layerWeights.self_attn_q_bias = weights.(pfx+"self_attn_q_bias"); end
             if isfield(weights, pfx+"self_attn_k_bias"), layerWeights.self_attn_k_bias = weights.(pfx+"self_attn_k_bias"); end
             if isfield(weights, pfx+"self_attn_v_bias"), layerWeights.self_attn_v_bias = weights.(pfx+"self_attn_v_bias"); end
             if isfield(weights, pfx+"self_attn_o_bias"), layerWeights.self_attn_o_bias = weights.(pfx+"self_attn_o_bias"); end
             layerWeights.mlp_gate_proj = weights.(pfx+"mlp_gate_proj");
             layerWeights.mlp_up_proj = weights.(pfx+"mlp_up_proj");
             layerWeights.mlp_down_proj = weights.(pfx+"mlp_down_proj");
        end

        state = layerStates{i};
        
        [Z, newState] = qwen2.layer.block(Z, state, layerWeights, hp, currentFreqs);
        
        layerStates{i} = newState;
    end
    
    % 4. Final Norm
    Z = rmsNormalization(Z, weights.norm, 1e-6);
    
    % 5. Output Head
    Z_flat = reshape(Z, size(Z,1), []);
    logits = weights.lm_head * Z_flat;
    
    vocabSize = size(weights.lm_head, 1);
    logits = reshape(logits, vocabSize, seqLen, batchSize);

end
