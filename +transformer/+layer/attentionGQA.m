function [X_out, past_key_value, attn_weights] = attentionGQA(X, past_key_value, weights, freqs_cis, hyperParameters)
% attentionGQA   Grouped Query Attention (GQA) with RoPE Support
%
%   Implements GQA as used in Llama 2/3 and Qwen.
%   Assumes weights are struct with fields: q_proj, k_proj, v_proj, o_proj.
%
%   Inputs:
%       X              - [hiddenSize, seqLen, batchSize]
%       past_key_value - Struct or Array with cached keys/values. 
%                        If struct: .keys [headDim, kvHeads, maxSeq, batch]
%                                   .values [headDim, kvHeads, maxSeq, batch]
%                        (Layout depends on your cache strategy, here we assume standard)
%       weights        - Struct containing projection weights.
%       freqs_cis      - [headDim/2, seqLen] Complex exponentials for RoPE.
%       hyperParameters- Struct with fields:
%                        .NumHeads (Query Heads)
%                        .NumKVHeads (Key/Value Heads)
%                        .HeadDim
%
%   Outputs:
%       X_out          - [hiddenSize, seqLen, batchSize]
%       past_key_value - Updated cache
%       attn_weights   - [numHeads, seqLen, cacheLen, batchSize] (Attention Scores)

    [hiddenSize, seqLen, batchSize] = size(X);
    numHeads = hyperParameters.NumHeads;
    numKVHeads = hyperParameters.NumKVHeads;
    headDim = hyperParameters.HeadDim;
    
    % 1. Projections
    % Weights in MATLAB are often [InputDim, OutputDim] in this repo's dense implementation
    % But let's check multiLayerPerceptron or attention.m to be consistent.
    % In this repo: linear layer is Z = weights * X + bias
    % So weights are [OutputDim, InputDim].
    
    xq = weights.q_proj * X; % [numHeads*headDim, seqLen, batch]
    xk = weights.k_proj * X; % [numKVHeads*headDim, seqLen, batch]
    xv = weights.v_proj * X; % [numKVHeads*headDim, seqLen, batch]
    
    % Reshape for heads
    % xq: [headDim, numHeads, seqLen, batch]
    xq = reshape(xq, [headDim, numHeads, seqLen, batchSize]);
    xk = reshape(xk, [headDim, numKVHeads, seqLen, batchSize]);
    xv = reshape(xv, [headDim, numKVHeads, seqLen, batchSize]);
    
    % 2. Apply RoPE (Rotary Embeddings)
    % Convert to complex for easy rotation
    % Assumes headDim is even.
    % Llama / Hugging Face uses "pair via split halves" strategy.
    % Pairs are (x[i], x[i + dim/2]).
    
    half = headDim / 2;
    xq_real = xq(1:half, :, :, :);
    xq_imag = xq(half+1:end, :, :, :);
    xq_c = complex(xq_real, xq_imag);
    
    xk_real = xk(1:half, :, :, :);
    xk_imag = xk(half+1:end, :, :, :);
    xk_c = complex(xk_real, xk_imag);
    
    % Reshape freqs_cis to broadcast: [headDim/2, 1, seqLen, 1]
    % Note: user must pass correct slice of freqs_cis corresponding to current seq positions
    f_cis = reshape(freqs_cis, [half, 1, seqLen, 1]);
    
    xq_rot = xq_c .* f_cis;
    xk_rot = xk_c .* f_cis;
    
    % Convert back to real and concatenate halves (Inverse of split)
    % x_out = [real, imag]
    xq(1:half, :, :, :) = real(xq_rot);
    xq(half+1:end, :, :, :) = imag(xq_rot);
    
    xk(1:half, :, :, :) = real(xk_rot);
    xk(half+1:end, :, :, :) = imag(xk_rot);
    
    % 3. KV Cache Management
    % Concatenate with past
    if ~isempty(past_key_value)
        % Assuming past_key_value is a struct with keys/values arrays
        % Concatenate along sequence dimension (dim 3)
        keys = cat(3, past_key_value.keys, xk);
        values = cat(3, past_key_value.values, xv);
    else
        keys = xk;
        values = xv;
    end
    
    % Update Cache
    past_key_value.keys = keys;
    past_key_value.values = values;
    
    % 4. Grouped Query Attention Matching
    % We need to repeat keys/values if numKVHeads < numHeads
    % GQA: Each KV head is shared by (numHeads / numKVHeads) Query heads.
    n_rep = numHeads / numKVHeads;
    if n_rep > 1
        % Repeat keys and values
        % keys: [headDim, numKVHeads, cacheLen, batch]
        % Target: [headDim, numHeads, cacheLen, batch]
        % We can use repelem on dim 2
        keys = repelem(keys, 1, n_rep);
        values = repelem(values, 1, n_rep);
    end
    
    % Now keys/values match numHeads dimensions
    % xq: [headDim, numHeads, seqLen, batch]
    % keys: [headDim, numHeads, cacheLen, batch]
    
    % 5. Scaled Dot-Product Attention
    % Scores = (xq' * keys) / sqrt(headDim)
    % Since we have multi-dim arrays, manual pagemtimes or permute is needed.
    
    % Permute to [headDim, cacheLen, numHeads, batch] for easier page mult?
    % Or better: [numHeads, batch, seqLen, headDim] x [numHeads, batch, headDim, cacheLen]
    
    % Let's permute to: [numHeads, headDim, seqLen, batch] -> This is messy in MATLAB linear algebra
    % MATLAB's pagemtimes operates on the first two dims.
    % Let's keep first two dims as vectors to multiply.
    
    % Query: [headDim, (numHeads*batch*seqLen)] ? No mixing seqLen and batch is bad for softmax.
    
    % Let's treat (NumHeads * Batch) as the "Page" dimension.
    % xq: [headDim, seqLen, numHeads, batch]
    xq_p = permute(xq, [1, 3, 4, 2]); % [headDim, numHeads, batch, seqLen] -> Wait.
    
    % Standard strategy:
    % Compute scores for each head and batch.
    
    % Transpose Q to: [headDim, seqLen, numHeads, batch]
    % K to: [headDim, cacheLen, numHeads, batch]
    
    % Since seqLen is usually 1 during generation, let's optimize for decoding.
    % If seqLen > 1 (prefill), we do full matrix mult.
    
    keys_p = reshape(keys, headDim, [], numHeads*batchSize);      % [headDim, cacheLen, N*B] -> No, creates mixing
    % We need [headDim, cacheLen, numHeads*batchSize]
    % Keys original: [headDim, numHeads, cacheLen, batch]
    keys_perm = permute(keys, [1, 3, 2, 4]); % [headDim, cacheLen, numHeads, batch]
    keys_2d = reshape(keys_perm, headDim, [], numHeads*batchSize); % [headDim, cacheLen, H*B] ?? NO.
    % Reshape combines adjacent dimensions. 
    % We want (2,4) to combine. 
    % Correct permute for keys: [headDim, cacheLen, numHeads, batch]
    keys_flat = reshape(keys_perm, headDim, size(keys,3), []); % [headDim, cacheLen, H*B]
    
    xq_perm = permute(xq, [1, 3, 2, 4]); % [headDim, numHeads, seqLen, batch] -> [headDim, seqLen, H, B]??
    % Wait, permute idx 2 (numHeads) and 3 (seqLen).
    % xq: [headDim, numHeads, seqLen, batch]
    xq_perm = permute(xq, [3, 1, 2, 4]); % [seqLen, headDim, numHeads, batch]
    % We want Q * K'. 
    % Q: [seqLen, headDim]
    % K: [cacheLen, headDim] (Transposed from above)
    
    % Actually MATLAB pagemtimes(X, Y) does X*Y on pages.
    % Let's prepare:
    % Q: [seqLen, headDim, H*B]
    % K: [headDim, cacheLen, H*B]
    % Result: [seqLen, cacheLen, H*B]
    
    xq_for_mul = reshape(permute(xq, [3, 1, 2, 4]), seqLen, headDim, []); 
    keys_for_mul = reshape(permute(keys, [1, 3, 2, 4]), headDim, [], numHeads*batchSize);
    
    scores = pagemtimes(xq_for_mul, keys_for_mul); % [seqLen, cacheLen, H*B]
    scores = scores / sqrt(headDim);
    
    % Masking (Causal) should be applied here if seqLen > 1
    % In this simplified implementation, we assume we are in "prefill" (no past)
    % or "generating" (seqLen=1).
    % If seqLen > 1 and no past, we need triangular mask.
    % If seqLen == 1, we attend to everything in cache (causal by definition if cache is past).
    
    [sL, cL, ~] = size(scores);
    if sL > 1
        % Create Causal Mask (Lower Triangular)
        % Assumes queries align with end of keys if using past? 
        % For standard prefill (sL == cL):
        if sL == cL
             mask = tril(true(sL, cL));
             % Apply -inf
             % Using a large negative number preserving precision type
             neg_inf = -1e4; % Sufficient for softmax
             
             % Expand mask to [sL, cL, 1] to broadcast
             % scores(~mask) = neg_inf; % Logical indexing might be slow on batches?
             % Additive mask is better
             maskVal = zeros(sL, cL, 'like', scores);
             maskVal(~mask) = neg_inf;
             
             scores = scores + maskVal;
        end
        % TODO: Handle general case where sL > 1 and past exists (rare in simple loops)
    end
    
    % Softmax
    % Implement manually for safety/compatibility
    % attn_weights = softmax(scores, 2);
    max_scores = max(scores, [], 2);
    exps = exp(scores - max_scores);
    attn_weights = exps ./ sum(exps, 2);
    
    % 6. Value weighted sum
    % values: [headDim, numHeads, cacheLen, batch]
    % values_for_mul: [cacheLen, headDim, H*B] (Need to perform V * Weights')
    % Or Weights * V.
    % Weights: [seqLen, cacheLen]
    % V: [cacheLen, headDim]
    % -> [seqLen, headDim]
    
    values_perm = permute(values, [3, 1, 2, 4]); % [cacheLen, headDim, H, B]
    values_for_mul = reshape(values_perm,size(values,3), headDim, []);
    
    attn_output = pagemtimes(attn_weights, values_for_mul); % [seqLen, headDim, H*B]
    
    % 7. Reshape and Project Out
    % [seqLen, headDim, numHeads, batch]
    attn_output = reshape(attn_output, seqLen, headDim, numHeads, batchSize);
    
    % Permute back to [headDim, numHeads, seqLen, batch] for concatenation
    % Target: [headDim*numHeads, seqLen, batch]
    attn_output = permute(attn_output, [2, 3, 1, 4]); % [headDim, numHeads, seqLen, batch]
    
    attn_output_cat = reshape(attn_output, headDim*numHeads, seqLen, batchSize);
    
    X_out = weights.o_proj * attn_output_cat;
    
    % Prepare attention weights output
    % [seqLen, cacheLen, H*B] -> [numHeads, seqLen, cacheLen, batchSize]
    tmp_weights = reshape(attn_weights, seqLen, size(attn_weights, 2), numHeads, batchSize);
    attn_weights = permute(tmp_weights, [3, 1, 2, 4]);

end
