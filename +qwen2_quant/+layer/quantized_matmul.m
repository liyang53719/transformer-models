function [Y, Y_packed] = quantized_matmul(W, X, cfg)
% quantized_matmul   Matrix multiplication with optional dequantization
%
%   Y = quantized_matmul(W, X)
%
%   Performs Y = W * X, where W can be:
%     - quantized_weight object (dequantized on-the-fly)
%     - dlarray or numeric array (used directly)
%
%   Inputs:
%       W - Weight matrix (possibly quantized)
%       X - Input activation matrix
%
%   Output:
%       Y - Result of W * X

    if nargin < 3 || ~isstruct(cfg)
        error('quantized_matmul:MissingConfig', ...
            'cfg is required and must be a struct. Pass RuntimeConfig from top-level.');
    end
    Y_packed = [];

    if isa(X, 'dlarray')
        X_data = single(extractdata(X));
    elseif isnumeric(X)
        X_data = single(X);
    else
        X_data = [];
    end

    switch lower(string(cfg.LinearMode))
        case "int8_int32_sim"
            if isa(W, 'qwen2_quant.internal.quantized_weight')
                W_data = single(W.dequantize());
            elseif isa(W, 'dlarray')
                W_data = single(extractdata(W));
            elseif isnumeric(W)
                W_data = single(W);
            else
                error('quantized_matmul:UnsupportedType', ...
                    'Unsupported weight type: %s', class(W));
            end
            
            [W_q, W_scale] = quantizeWeightInt8(W_data, cfg);
            [X_q, X_scale] = quantizeActivationInt8(X_data, cfg);

            % Always use strict int32 accumulation as per user request
            acc_int = int8Int8MatmulAccInt32(W_q, X_q);
            
            Y_data = single(acc_int) .* (W_scale * X_scale);

            if isa(X, 'dlarray')
                Y = dlarray(Y_data);
            else
                Y = Y_data;
            end

            if cfg.TracePrecision
                qwen2_quant.internal.precision_trace('log', 'linear.int8_weight', W_q);
                qwen2_quant.internal.precision_trace('log', 'linear.int8_activation', X_q);
                qwen2_quant.internal.precision_trace('log', 'linear.int32_accumulator', acc_int);
                qwen2_quant.internal.precision_trace('log', 'linear.output', Y);
            end

        case "q8_0_block_sim"
            % Block-wise Quantization Simulation
            % Directly uses Q8_0 components without dequantizing to full float first.
            
            if isa(W, 'qwen2_quant.internal.quantized_weight') && strcmp(W.QuantType, 'Q8_0')
                % Ideal path: Extract raw components
                [W_q, W_scales] = get_q8_0_components_cached(W);
                % W_q: [32, num_blocks], W_scales: [1, num_blocks]
            else
                % Fallback (slow/incorrect for simulation intent if not Q8_0):
                % If user forces this mode but weights aren't Q8_0, we can't do block-sim efficiently.
                % For now, error out or fallback.
                error('quantized_matmul:InvalidWeight', ...
                    'q8_0_block_sim mode requires Q8_0 quantized weights.');
            end
            
            % Quantize Activation to int8 (per-tensor or per-col doesn't matter much for the int8 part, 
            % but we need consistent scaling.
            % However, standard Q8_0 dot product: sum( (w_i * s_b) * x_i )
            % = sum( w_i * x_i * s_b )
            % If we quantize X: x_i ~= x_q_i * s_x
            % = sum( w_i * (x_q_i * s_x) * s_b )
            % = s_x * sum( w_i * x_q_i * s_b )
            % This requires accumulators to be float if s_b changes, OR we multiply s_b inside memory.
            % But the user asked for "int8 calculation".
            % Typically: Scale is applied AFTER accumulation for row-wise.
            % For Block-wise, Scale changes every 32 items.
            % So we must accumulate in chunks of 32.
            
            % Implementation using block-wise kernel with float activations.
            % This keeps GGUF integer weights/scales and avoids full-matrix dequantization.
            if isempty(X_data)
                error('quantized_matmul:UnsupportedActivationType', ...
                    'q8_0_block_sim requires numeric or dlarray activation input.');
            end
            Y_data = q8_0_block_matmul(W_q, W_scales, X_data, W);
            
            if isa(X, 'dlarray')
                Y = dlarray(Y_data);
            else
                Y = Y_data;
            end

        case "q4_0_block_sim"
            if isa(W, 'qwen2_quant.internal.quantized_weight') && strcmp(W.QuantType, 'Q4_0')
                [W_q, W_d] = get_q4_0_components_cached(W);
            else
                error('quantized_matmul:InvalidWeight', ...
                    'q4_0_block_sim mode requires Q4_0 quantized weights.');
            end

            if isempty(X_data)
                error('quantized_matmul:UnsupportedActivationType', ...
                    'q4_0_block_sim requires numeric or dlarray activation input.');
            end
            Y_data = q4_0_block_matmul(W_q, W_d, X_data, W);

            if isa(X, 'dlarray')
                Y = dlarray(Y_data);
            else
                Y = Y_data;
            end

        case "q4_k_block_sim"
            if isa(W, 'qwen2_quant.internal.quantized_weight') && strcmp(W.QuantType, 'Q4_K')
                comp = get_q4_k_components_cached(W);
                if isempty(X_data)
                    error('quantized_matmul:UnsupportedActivationType', ...
                        'q4_k_block_sim requires numeric or dlarray activation input.');
                end
                Y_data = q4_k_block_matmul(comp, X_data, W);
            elseif isa(W, 'qwen2_quant.internal.quantized_weight') && strcmp(W.QuantType, 'Q6_K')
                comp = get_q6_k_components_cached(W);
                if isempty(X_data)
                    error('quantized_matmul:UnsupportedActivationType', ...
                        'q4_k_block_sim requires numeric or dlarray activation input.');
                end
                Y_data = q6_k_block_matmul(comp, X_data, W);
            else
                error('quantized_matmul:InvalidWeight', ...
                    'q4_k_block_sim mode requires Q4_K/Q6_K quantized weights.');
            end

            if isa(X, 'dlarray')
                Y = dlarray(Y_data);
            else
                Y = Y_data;
            end

        case "q4_k_m_block_sim"
            if isa(W, 'qwen2_quant.internal.quantized_weight') && strcmp(W.QuantType, 'Q4_K')
                comp = get_q4_k_components_cached(W);
                if isempty(X_data)
                    error('quantized_matmul:UnsupportedActivationType', ...
                        'q4_k_m_block_sim requires numeric or dlarray activation input.');
                end
                Y_data = q4_k_block_matmul(comp, X_data, W);
            elseif isa(W, 'qwen2_quant.internal.quantized_weight') && strcmp(W.QuantType, 'Q6_K')
                comp = get_q6_k_components_cached(W);
                if isempty(X_data)
                    error('quantized_matmul:UnsupportedActivationType', ...
                        'q4_k_m_block_sim requires numeric or dlarray activation input.');
                end
                Y_data = q6_k_block_matmul(comp, X_data, W);
            elseif isa(W, 'qwen2_quant.internal.quantized_weight') && strcmp(W.QuantType, 'Q8_0')
                [W_q, W_scales] = get_q8_0_components_cached(W);
                if isempty(X_data)
                    error('quantized_matmul:UnsupportedActivationType', ...
                        'q4_k_m_block_sim requires numeric or dlarray activation input.');
                end
                Y_data = q8_0_block_matmul(W_q, W_scales, X_data, W);
            else
                error('quantized_matmul:InvalidWeight', ...
                    'q4_k_m_block_sim mode requires Q4_K/Q6_K/Q8_0 quantized weights.');
            end
            
            if isa(X, 'dlarray')
                Y = dlarray(Y_data);
            else
                Y = Y_data;
            end

        case "gptq_int4_matlab_sim"
            if isGptqPackedWeight(W)
                Y_data = gptq_int4_matmul(W, X_data);
            elseif isa(W, 'qwen2_quant.internal.quantized_weight')
                W_data = single(W.dequantize());
                Y_data = W_data * X_data;
            elseif isa(W, 'dlarray')
                Y_data = single(extractdata(W)) * X_data;
            elseif isnumeric(W)
                Y_data = single(W) * X_data;
            else
                error('quantized_matmul:UnsupportedType', ...
                    'Unsupported weight type: %s', class(W));
            end

            if isa(X, 'dlarray')
                Y = dlarray(Y_data);
            else
                Y = Y_data;
            end

        case "gptq_int4_quant_sim"
            if isGptqPackedWeight(W)
                X_pack = ensurePackedActivationForQuantSim(X, X_data, cfg);
                [Y_data, Y_packed] = gptq_int4_quant_matmul(W, X_pack, cfg);
            elseif isa(W, 'qwen2_quant.internal.quantized_weight')
                W_data = single(W.dequantize());
                Y_data = W_data * X_data;
            elseif isa(W, 'dlarray')
                Y_data = single(extractdata(W)) * X_data;
            elseif isnumeric(W)
                Y_data = single(W) * X_data;
            else
                error('quantized_matmul:UnsupportedType', ...
                    'Unsupported weight type: %s', class(W));
            end

            if isa(X, 'dlarray')
                Y = dlarray(Y_data);
            else
                Y = Y_data;
            end

        otherwise
            if isa(W, 'qwen2_quant.internal.quantized_weight')
                W_data = single(W.dequantize());
            elseif isa(W, 'dlarray')
                W_data = single(extractdata(W));
            elseif isnumeric(W)
                W_data = single(W);
            else
                error('quantized_matmul:UnsupportedType', ...
                    'Unsupported weight type: %s', class(W));
            end

            Y_data = W_data * X_data;
            if isa(X, 'dlarray')
                Y = dlarray(Y_data);
            else
                Y = Y_data;
            end

            if cfg.TracePrecision
                qwen2_quant.internal.precision_trace('log', 'linear.weight_float', W_data);
                qwen2_quant.internal.precision_trace('log', 'linear.activation_float', X_data);
                qwen2_quant.internal.precision_trace('log', 'linear.output', Y);
            end
    end
end

function X_pack = ensurePackedActivationForQuantSim(X, X_data, cfg)
    isPacked = isstruct(X) && isfield(X, 'Q') && isfield(X, 'Scale') && isfield(X, 'Bias');
    if isPacked
        X_pack = X;
        return;
    end

    if isempty(X_data)
        error('quantized_matmul:UnsupportedActivationType', ...
            'gptq_int4_quant_sim requires numeric, dlarray, or packed activation input.');
    end

    [X_q, X_scale, X_bias] = quantizeActivationInt8Affine(X_data, cfg);
    X_pack = struct('Q', X_q, 'Scale', X_scale, 'Bias', X_bias);
end

function [q, scale] = quantizeWeightInt8(x, cfg)
    mode = 'per_row';
    if isfield(cfg, 'Int8WeightScaleMode')
        mode = lower(string(cfg.Int8WeightScaleMode));
    end

    switch mode
        case "per_tensor"
            [q, scale] = quantizeSymmetricInt8PerTensor(x);
        otherwise
            [q, scale] = quantizeSymmetricInt8PerRow(x);
    end
end

function [q, scale] = quantizeActivationInt8(x, cfg)
    mode = 'per_col';
    if isfield(cfg, 'Int8ActivationScaleMode')
        mode = lower(string(cfg.Int8ActivationScaleMode));
    end

    switch mode
        case "per_tensor"
            [q, scale] = quantizeSymmetricInt8PerTensor(x);
        otherwise
            [q, scale] = quantizeSymmetricInt8PerCol(x);
    end
end

function [q, scale] = quantizeSymmetricInt8PerTensor(x)
    maxAbs = max(abs(x(:)));
    if maxAbs < eps('single')
        scale = single(1);
        q = int8(zeros(size(x), 'single'));
        return;
    end
    scale = single(maxAbs / 127.0);
    q = int8(max(min(round(x ./ scale), 127), -127));
end

function [q, scale] = quantizeSymmetricInt8PerRow(x)
    maxAbs = max(abs(x), [], 2);
    scale = max(maxAbs / 127.0, eps('single'));
    q = int8(max(min(round(x ./ scale), 127), -127));
end

function [q, scale] = quantizeSymmetricInt8PerCol(x)
    maxAbs = max(abs(x), [], 1);
    scale = max(maxAbs / 127.0, eps('single'));
    q = int8(max(min(round(x ./ scale), 127), -127));
end

function acc = int8Int8MatmulAccInt32(W_q, X_q)
    [m, k_w] = size(W_q);
    [k_x, n] = size(X_q);

    if k_w ~= k_x
        error('quantized_matmul:ShapeMismatch', ...
            'Inner dimensions must agree: size(W,2)=%d, size(X,1)=%d', k_w, k_x);
    end

    acc = zeros(m, n, 'int32');

    for kk = 1:k_w
        w_col = int32(W_q(:, kk));
        x_row = int32(X_q(kk, :));
        acc = acc + (w_col .* x_row);
    end
end

function Y_val = q8_0_block_matmul(W_q_blocks, W_scales, X_data, W)
% q8_0_block_matmul Block-wise matrix multiplication (float activations)
%
% Inputs:
%   W_q_blocks: [32, num_blocks] int8, raw qvals from Q8_0
%   W_scales:   [1, num_blocks] single
%   X_data:     [K, N] single
%   W:          quantized_weight object (contains Dims, NeedsTranspose)
%
% Output:
%   Y_val:      [M, N] single

    original_dims = W.Dims;

    % Check for Transpose logic
    if W.NeedsTranspose
        M = original_dims(2); % 256 (Output Dim)
        K = original_dims(1); % 1536 (Input Dim)
        blocks_per_row = ceil(K / 32); 
    else
        M = original_dims(1);
        K = original_dims(2);
        blocks_per_row = ceil(K / 32);
    end

    X_data = single(X_data);
    X_data = single(X_data);
    [~, N_act] = size(X_data);

    expected_blocks = M * blocks_per_row;
    W_q_subset = W_q_blocks(:, 1 : expected_blocks);
    W_s_subset = W_scales(1 : expected_blocks);

    W_q_grid = reshape(W_q_subset, 32, blocks_per_row, M);
    W_s_grid = reshape(W_s_subset, blocks_per_row, M); % [blocks_per_row, M]
    
    Y_val = zeros(M, N_act, 'single');

    for b = 1:blocks_per_row
        s_col = W_s_grid(b, :).'; 
        w_chunk = permute(W_q_grid(:, b, :), [3, 1, 2]); 

        idx_start = (b-1)*32 + 1;
        idx_end = b*32;
        x_chunk = X_data(idx_start:idx_end, :);

        partial = single(w_chunk) * x_chunk;
        Y_val = Y_val + (partial .* s_col);
    end
end

function Y_val = q4_0_block_matmul(W_q_blocks, W_deltas, X_data, W)
% q4_0_block_matmul Block-wise matrix multiplication for Q4_0

    original_dims = W.Dims;
    if W.NeedsTranspose
        M = original_dims(2);
        K = original_dims(1);
    else
        M = original_dims(1);
        K = original_dims(2);
    end

    [~, N_act] = size(X_data);
    blocks_per_row = ceil(K / 32);
    expected_blocks = M * blocks_per_row;

    W_q_subset = W_q_blocks(:, 1:expected_blocks);
    W_d_subset = W_deltas(1:expected_blocks);

    W_q_grid = reshape(W_q_subset, 32, blocks_per_row, M);
    W_d_grid = reshape(W_d_subset, blocks_per_row, M);

    Y_val = zeros(M, N_act, 'single');

    for b = 1:blocks_per_row
        d_col = W_d_grid(b, :).';
        q_chunk = permute(W_q_grid(:, b, :), [3, 1, 2]);

        idx_start = (b-1)*32 + 1;
        idx_end = min(b*32, K);
        valid_len = idx_end - idx_start + 1;

        x_chunk = X_data(idx_start:idx_end, :);
        partial = (single(q_chunk(:, 1:valid_len)) * x_chunk) .* d_col;
        Y_val = Y_val + partial;
    end
end

function Y_val = q4_k_block_matmul(comp, X_data, W)
% q4_k_block_matmul Block-wise matrix multiplication for Q4_K

    original_dims = W.Dims;
    if W.NeedsTranspose
        M = original_dims(2);
        K = original_dims(1);
    else
        M = original_dims(1);
        K = original_dims(2);
    end

    X_data = single(X_data);
    [~, N_act] = size(X_data);
    blocks_per_row = ceil(K / 256);
    expected_blocks = M * blocks_per_row;

    d = single(comp.d(1:expected_blocks));
    dmin = single(comp.dmin(1:expected_blocks));
    sc = single(comp.scales(:, 1:expected_blocks));
    mn = single(comp.mins(:, 1:expected_blocks));
    qs = comp.qs(:, 1:expected_blocks);

    Y_val = zeros(M, N_act, 'single');

    for b = 1:blocks_per_row
        idx_blocks = (0:M-1) * blocks_per_row + b;

        d_b = reshape(d(idx_blocks), M, 1);
        dmin_b = reshape(dmin(idx_blocks), M, 1);
        sc_b = sc(:, idx_blocks); % [8, M]
        mn_b = mn(:, idx_blocks); % [8, M]
        qs_b = qs(:, idx_blocks);         % [128, M]

        w_chunk = zeros(M, 256, 'single');

        for seg = 0:3
            q32 = qs_b(seg*32 + (1:32), :); % [32, M]
            ql = single(bitand(q32, uint8(15))).';
            qh = single(bitshift(q32, -4)).';

            is1 = seg*2 + 1;
            is2 = seg*2 + 2;

            d1 = d_b .* reshape(sc_b(is1, :).', M, 1);
            m1 = dmin_b .* reshape(mn_b(is1, :).', M, 1);
            d2 = d_b .* reshape(sc_b(is2, :).', M, 1);
            m2 = dmin_b .* reshape(mn_b(is2, :).', M, 1);

            base = seg * 64;
            w_chunk(:, base + (1:32)) = d1 .* ql - m1;
            w_chunk(:, base + 32 + (1:32)) = d2 .* qh - m2;
        end

        idx_start = (b-1)*256 + 1;
        idx_end = min(b*256, K);
        valid_len = idx_end - idx_start + 1;

        x_chunk = X_data(idx_start:idx_end, :);
        partial = w_chunk(:, 1:valid_len) * single(x_chunk);
        Y_val = Y_val + partial;
    end
end

function Y_val = q6_k_block_matmul(comp, X_data, W)
% q6_k_block_matmul Block-wise matrix multiplication for Q6_K

    original_dims = W.Dims;
    if W.NeedsTranspose
        M = original_dims(2);
        K = original_dims(1);
    else
        M = original_dims(1);
        K = original_dims(2);
    end

    X_data = single(X_data);
    [~, N_act] = size(X_data);
    blocks_per_row = ceil(K / 256);
    expected_blocks = M * blocks_per_row;

    ql = comp.ql(:, 1:expected_blocks);
    qh = comp.qh(:, 1:expected_blocks);
    sc = single(comp.scales(:, 1:expected_blocks));
    d = single(comp.d(1:expected_blocks));

    Y_val = zeros(M, N_act, 'single');

    for b = 1:blocks_per_row
        idx_blocks = (0:M-1) * blocks_per_row + b;

        ql_b = ql(:, idx_blocks); % [128, M]
        qh_b = qh(:, idx_blocks); % [64, M]
        sc_b = sc(:, idx_blocks); % [16, M]
        d_b = reshape(d(idx_blocks), M, 1);

        w_chunk = zeros(M, 256, 'single');

        outBase = 0;
        ql_ptr = 1;
        qh_ptr = 1;
        sc_ptr = 1;

        for n = 1:2
            ql64 = ql_b(ql_ptr:ql_ptr+63, :);
            qh32 = qh_b(qh_ptr:qh_ptr+31, :);

            for l = 0:31
                qh_v = qh32(l+1, :);

                q1 = single(bitand(ql64(l+1, :), uint8(15))) + single(bitshift(bitand(qh_v, uint8(3)), 4)) - 32;
                q2 = single(bitand(ql64(l+33, :), uint8(15))) + single(bitshift(bitand(bitshift(qh_v, -2), uint8(3)), 4)) - 32;
                q3 = single(bitshift(ql64(l+1, :), -4)) + single(bitshift(bitand(bitshift(qh_v, -4), uint8(3)), 4)) - 32;
                q4 = single(bitshift(ql64(l+33, :), -4)) + single(bitshift(bitand(bitshift(qh_v, -6), uint8(3)), 4)) - 32;

                is = sc_ptr + floor(l / 16);

                w_chunk(:, outBase + l + 1)      = d_b .* reshape(sc_b(is + 0, :).', M, 1) .* reshape(q1.', M, 1);
                w_chunk(:, outBase + l + 1 + 32) = d_b .* reshape(sc_b(is + 2, :).', M, 1) .* reshape(q2.', M, 1);
                w_chunk(:, outBase + l + 1 + 64) = d_b .* reshape(sc_b(is + 4, :).', M, 1) .* reshape(q3.', M, 1);
                w_chunk(:, outBase + l + 1 + 96) = d_b .* reshape(sc_b(is + 6, :).', M, 1) .* reshape(q4.', M, 1);
            end

            outBase = outBase + 128;
            ql_ptr = ql_ptr + 64;
            qh_ptr = qh_ptr + 32;
            sc_ptr = sc_ptr + 8;
        end

        idx_start = (b-1)*256 + 1;
        idx_end = min(b*256, K);
        valid_len = idx_end - idx_start + 1;

        x_chunk = X_data(idx_start:idx_end, :);
        partial = w_chunk(:, 1:valid_len) * single(x_chunk);
        Y_val = Y_val + partial;
    end
end

function tf = isGptqPackedWeight(W)
    tf = isstruct(W) && isfield(W, 'QuantType') && ...
    (strcmpi(string(W.QuantType), "GPTQ_INT4") || strcmpi(string(W.QuantType), "AWQ_INT4")) && ...
        isfield(W, 'qweight') && isfield(W, 'qzeros') && ...
        isfield(W, 'scales') && isfield(W, 'g_idx');
end

function Y_val = gptq_int4_matmul(W, X_data)
% GPTQ int4 matmul using packed tensors and group-wise scales/zeros.
% Semantics match: dequant = scales[g_idx] .* (qweight - qzeros[g_idx]).

    in_features = double(W.in_features);
    out_features = double(W.out_features);
    [k_x, ~] = size(X_data);
    if k_x ~= in_features
        error('quantized_matmul:ShapeMismatch', ...
            'GPTQ input mismatch: expected %d rows, got %d.', in_features, k_x);
    end

    quantType = upper(string(W.QuantType));
    useAwqOrder = strcmpi(quantType, "AWQ_INT4");

    qweight_u4 = unpack_qweight_int4(W.qweight, in_features, out_features, useAwqOrder);
    qzeros_u4 = unpack_qzeros_int4(W.qzeros, out_features, useAwqOrder);
    scales = single(W.scales);

    if size(scales, 2) ~= out_features
        error('quantized_matmul:ShapeMismatch', ...
            'GPTQ scales second dimension mismatch: expected %d, got %d.', out_features, size(scales, 2));
    end
    if size(qzeros_u4, 1) ~= size(scales, 1)
        error('quantized_matmul:ShapeMismatch', ...
            'GPTQ qzeros/scales group mismatch: %d vs %d.', size(qzeros_u4, 1), size(scales, 1));
    end

    g_idx = double(W.g_idx(:)) + 1;
    if numel(g_idx) ~= in_features
        error('quantized_matmul:ShapeMismatch', ...
            'GPTQ g_idx length mismatch: expected %d, got %d.', in_features, numel(g_idx));
    end

    w_dequant = (single(qweight_u4) - single(qzeros_u4(g_idx, :))) .* scales(g_idx, :);
    Y_val = w_dequant.' * X_data;

    if isfield(W, 'bias') && ~isempty(W.bias)
        Y_val = Y_val + single(W.bias);
    end
end

function [Y_val, Y_packed] = gptq_int4_quant_matmul(W, X_in, cfg)
% GPTQ/AWQ int4 weight with int8 activation simulation.
% Integer core accumulation, float requantization by scales and activation bias.

    in_features = double(W.in_features);
    out_features = double(W.out_features);

    X_q = X_in.Q;
    X_scale = X_in.Scale;
    X_bias = X_in.Bias;
    [k_x, n_act] = size(X_q);

    if k_x ~= in_features
        error('quantized_matmul:ShapeMismatch', ...
            'GPTQ input mismatch: expected %d rows, got %d.', in_features, k_x);
    end

    quantType = upper(string(W.QuantType));
    useAwqOrder = strcmpi(quantType, "AWQ_INT4");

    qweight_u4 = unpack_qweight_int4(W.qweight, in_features, out_features, useAwqOrder);
    qzeros_u4 = unpack_qzeros_int4(W.qzeros, out_features, useAwqOrder);
    scales = single(W.scales);

    if size(scales, 2) ~= out_features
        error('quantized_matmul:ShapeMismatch', ...
            'GPTQ scales second dimension mismatch: expected %d, got %d.', out_features, size(scales, 2));
    end
    if size(qzeros_u4, 1) ~= size(scales, 1)
        error('quantized_matmul:ShapeMismatch', ...
            'GPTQ qzeros/scales group mismatch: %d vs %d.', size(qzeros_u4, 1), size(scales, 1));
    end

    g_idx = double(W.g_idx(:)) + 1;
    if numel(g_idx) ~= in_features
        error('quantized_matmul:ShapeMismatch', ...
            'GPTQ g_idx length mismatch: expected %d, got %d.', in_features, numel(g_idx));
    end

    Y_val = zeros(out_features, n_act, 'single');
    num_groups = size(scales, 1);
    for g = 1:num_groups
        idx = find(g_idx == g);
        if isempty(idx)
            continue;
        end

        acc_g = zeros(out_features, n_act, 'int32');
        sum_w_g = zeros(out_features, 1, 'int32');
        z_row = int16(qzeros_u4(g, :));
        group_acc = zeros(out_features, n_act, 'single');
        for t = 1:numel(idx)
            r = idx(t);
            w_row = int16(qweight_u4(r, :)) - z_row;
            x_row = int16(X_q(r, :));
            prod_int = int32(w_row(:)) .* int32(x_row);
            acc_g = acc_g + prod_int;
            sum_w_g = sum_w_g + int32(w_row(:));

            s_row = activationParamRow(X_scale, r, n_act, in_features, 'scale');
            b_row = activationParamRow(X_bias, r, n_act, in_features, 'bias');
            group_acc = group_acc + single(prod_int) .* s_row + single(w_row(:)) * b_row;
        end

        Y_val = Y_val + scales(g, :).' .* group_acc;
    end

    if isfield(W, 'bias') && ~isempty(W.bias)
        Y_val = Y_val + single(W.bias);
    end

    [Y_q, Y_scale, Y_bias] = quantizeActivationInt8Affine(Y_val, cfg);
    Y_packed = struct('Q', Y_q, 'Scale', Y_scale, 'Bias', Y_bias, 'OriginalSize', size(Y_val));
end

function rowVal = activationParamRow(param, rowIdx, n_cols, in_features, paramName)
    if isscalar(param)
        rowVal = single(param);
        return;
    end

    [r, c] = size(param);
    if c ~= n_cols
        error('quantized_matmul:ShapeMismatch', ...
            'Activation %s columns mismatch: expected %d, got %d.', paramName, n_cols, c);
    end

    if r == 1
        rowVal = single(param);
    elseif r == in_features
        rowVal = single(param(rowIdx, :));
    elseif mod(in_features, r) == 0
        groupSize = in_features / r;
        g = floor((rowIdx - 1) / groupSize) + 1;
        rowVal = single(param(g, :));
    else
        error('quantized_matmul:ShapeMismatch', ...
            'Activation %s rows mismatch: expected 1, %d, or group rows that divide %d; got %d.', ...
            paramName, in_features, in_features, r);
    end
end

function [q, scale, bias] = quantizeActivationInt8Affine(x, cfg)
    mode = 'per_col';
    if isfield(cfg, 'Int8ActivationScaleMode')
        mode = lower(string(cfg.Int8ActivationScaleMode));
    end

    switch mode
        case "per_tensor"
            [q, scale, bias] = quantizeAffineInt8PerTensor(x);
        otherwise
            [q, scale, bias] = quantizeAffineInt8PerCol(x);
    end
end

function [q, scale, bias] = quantizeAffineInt8PerTensor(x)
    x_min = min(x(:));
    x_max = max(x(:));
    if abs(x_max - x_min) < eps('single')
        scale = single(1);
        bias = single(x_min);
        q = int8(zeros(size(x), 'single'));
        return;
    end

    scale = single((x_max - x_min) / 254.0);
    bias = single((x_max + x_min) / 2.0);
    q = int8(max(min(round((x - bias) ./ scale), 127), -127));
end

function [q, scale, bias] = quantizeAffineInt8PerCol(x)
    x_min = min(x, [], 1);
    x_max = max(x, [], 1);

    span = x_max - x_min;
    scale = single(span / 254.0);
    tiny = span < eps('single');
    scale(tiny) = single(1);

    bias = single((x_max + x_min) / 2.0);
    q = int8(max(min(round((x - bias) ./ scale), 127), -127));
end

function comp = get_gptq_components_cached(W)
    persistent gptqCompCache
    if isempty(gptqCompCache)
        gptqCompCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    end

    key = build_gptq_cache_key(W);
    if isKey(gptqCompCache, key)
        comp = gptqCompCache(key);
        return;
    end

    bits = double(W.bits);
    if bits ~= 4
        error('quantized_matmul:UnsupportedBits', 'Only GPTQ int4 is supported, got bits=%d.', bits);
    end

    in_features = double(W.in_features);
    out_features = double(W.out_features);

    quantType = upper(string(W.QuantType));
    useAwqOrder = strcmpi(quantType, "AWQ_INT4");

    qweight_u4 = unpack_qweight_int4(W.qweight, in_features, out_features, useAwqOrder);
    qzeros_u4 = unpack_qzeros_int4(W.qzeros, out_features, useAwqOrder);
    scales = single(W.scales);

    if size(scales, 2) ~= out_features
        error('quantized_matmul:ShapeMismatch', ...
            'GPTQ scales second dimension mismatch: expected %d, got %d.', out_features, size(scales, 2));
    end
    if size(qzeros_u4, 1) ~= size(scales, 1)
        error('quantized_matmul:ShapeMismatch', ...
            'GPTQ qzeros/scales group mismatch: %d vs %d.', size(qzeros_u4, 1), size(scales, 1));
    end

    g_idx = double(W.g_idx(:)) + 1; % convert 0-based to MATLAB 1-based
    if numel(g_idx) ~= in_features
        error('quantized_matmul:ShapeMismatch', ...
            'GPTQ g_idx length mismatch: expected %d, got %d.', in_features, numel(g_idx));
    end

    num_groups = size(scales, 1);
    group_rows = cell(num_groups, 1);
    for g = 1:num_groups
        group_rows{g} = find(g_idx == g);
    end

    comp = struct();
    comp.qweight_u4 = qweight_u4;
    comp.qzeros_u4 = qzeros_u4;
    comp.scales = scales;
    comp.group_rows = group_rows;

    gptqCompCache(key) = comp;
end

function key = build_gptq_cache_key(W)
    qshape = size(W.qweight);
    zshape = size(W.qzeros);
    sshape = size(W.scales);
    sampleQ = min(numel(W.qweight), 64);
    sampleZ = min(numel(W.qzeros), 64);
    sumQ = sum(uint64(typecast(int32(W.qweight(1:sampleQ)), 'uint32')) .* uint64((1:sampleQ)'));
    sumZ = sum(uint64(typecast(int32(W.qzeros(1:sampleZ)), 'uint32')) .* uint64((1:sampleZ)'));

    key = sprintf('GPTQ_%dx%d_%dx%d_%dx%d_b%d_g%d_q%u_z%u', ...
        qshape(1), qshape(2), zshape(1), zshape(2), sshape(1), sshape(2), ...
        double(W.bits), double(W.group_size), sumQ, sumZ);
end

function q_u4 = unpack_qweight_int4(qweight_packed, in_features, out_features, useAwqOrder)
% qweight_packed supports two layouts:
%   - [in_features/8, out_features] (GPTQ style)
%   - [out_features, in_features/8] (AWQ style)
%   - [in_features, out_features/8] (AWQ alt packing)
%   - [out_features/8, in_features] (AWQ alt packing transposed)
    packed = int32(qweight_packed);

    if isequal(size(packed), [in_features/8, out_features])
        q_u4 = zeros(in_features, out_features, 'uint8');
        for p = 0:7
            rows = (p + 1):8:in_features;
            srcNib = map_unpack_nibble(p, useAwqOrder);
            q_u4(rows, :) = uint8(bitand(bitshift(packed, -4*srcNib), int32(15)));
        end
        return;
    end

    % Ambiguous square case for AWQ (in_features == out_features):
    % [out_features, in_features/8] and [in_features, out_features/8] have identical shape.
    % Prefer [in_features, out_features/8] for AWQ exports produced by this project.
    if useAwqOrder
        if isequal(size(packed), [in_features, out_features/8])
            q_u4 = zeros(in_features, out_features, 'uint8');
            for p = 0:7
                cols = (p + 1):8:out_features;
                srcNib = map_unpack_nibble(p, useAwqOrder);
                q_u4(:, cols) = uint8(bitand(bitshift(packed, -4*srcNib), int32(15)));
            end
            return;
        end

        if isequal(size(packed), [out_features, in_features/8])
            q_u4_t = zeros(out_features, in_features, 'uint8');
            for p = 0:7
                cols = (p + 1):8:in_features;
                srcNib = map_unpack_nibble(p, useAwqOrder);
                q_u4_t(:, cols) = uint8(bitand(bitshift(packed, -4*srcNib), int32(15)));
            end
            q_u4 = q_u4_t.';
            return;
        end
    else
        if isequal(size(packed), [out_features, in_features/8])
            q_u4_t = zeros(out_features, in_features, 'uint8');
            for p = 0:7
                cols = (p + 1):8:in_features;
                srcNib = map_unpack_nibble(p, useAwqOrder);
                q_u4_t(:, cols) = uint8(bitand(bitshift(packed, -4*srcNib), int32(15)));
            end
            q_u4 = q_u4_t.';
            return;
        end

        if isequal(size(packed), [in_features, out_features/8])
            q_u4 = zeros(in_features, out_features, 'uint8');
            for p = 0:7
                cols = (p + 1):8:out_features;
                srcNib = map_unpack_nibble(p, useAwqOrder);
                q_u4(:, cols) = uint8(bitand(bitshift(packed, -4*srcNib), int32(15)));
            end
            return;
        end
    end

    if isequal(size(packed), [out_features/8, in_features])
        packed_t = packed.';
        q_u4 = zeros(in_features, out_features, 'uint8');
        for p = 0:7
            cols = (p + 1):8:out_features;
            srcNib = map_unpack_nibble(p, useAwqOrder);
            q_u4(:, cols) = uint8(bitand(bitshift(packed_t, -4*srcNib), int32(15)));
        end
        return;
    end

    error('quantized_matmul:UnsupportedLayout', ...
        'Unsupported qweight packed layout: got [%d,%d], expected one of [%d,%d], [%d,%d], [%d,%d], [%d,%d].', ...
        size(packed,1), size(packed,2), ...
        in_features/8, out_features, ...
        out_features, in_features/8, ...
        in_features, out_features/8, ...
        out_features/8, in_features);
end

function z_u4 = unpack_qzeros_int4(qzeros_packed, out_features, useAwqOrder)
% qzeros_packed: [num_groups, out_features/8] int32
    num_groups = size(qzeros_packed, 1);
    z_u4 = zeros(num_groups, out_features, 'uint8');
    packed = int32(qzeros_packed);
    for p = 0:7
        cols = (p + 1):8:out_features;
        srcNib = map_unpack_nibble(p, useAwqOrder);
        z_u4(:, cols) = uint8(bitand(bitshift(packed, -4*srcNib), int32(15)));
    end
end

function srcNib = map_unpack_nibble(dstPos, useAwqOrder)
% dstPos is 0-based logical position inside each 8-value pack.
    if ~useAwqOrder
        srcNib = dstPos;
        return;
    end
    % AWQ pack order_map=[0,2,4,6,1,3,5,7]: source nib index by destination pos.
    invMap = [0, 4, 1, 5, 2, 6, 3, 7];
    srcNib = invMap(dstPos + 1);
end

function [q, s] = get_q8_0_components_cached(W)
    persistent q8CompCache
    if isempty(q8CompCache)
        q8CompCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    end

    if isprop(W, 'CacheKey') && ~isempty(W.CacheKey) && uint64(W.CacheKey) ~= 0
        key = ['Q8_0_CK_' char(string(uint64(W.CacheKey)))];
    else
        data_u8 = uint8(W.Data(:));
        n = numel(data_u8);
        sampleN = min(n, 256);
        sample = data_u8(1:sampleN);
        checksum = sum(uint64(sample) .* uint64((1:sampleN)'));
        key = sprintf('Q8_0_SIG_%s_%d_%u', mat2str(double(W.Dims)), n, checksum);
    end

    if isKey(q8CompCache, key)
        comp = q8CompCache(key);
        q = comp.q;
        s = comp.s;
    else
        [q, s] = W.get_q8_0_components();
        q8CompCache(key) = struct('q', q, 's', s);
    end
end

function [q, d] = get_q4_0_components_cached(W)
    persistent q4CompCache
    if isempty(q4CompCache)
        q4CompCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    end

    if isprop(W, 'CacheKey') && ~isempty(W.CacheKey) && uint64(W.CacheKey) ~= 0
        key = ['Q4_0_CK_' char(string(uint64(W.CacheKey)))];
    else
        data_u8 = uint8(W.Data(:));
        n = numel(data_u8);
        sampleN = min(n, 256);
        sample = data_u8(1:sampleN);
        checksum = sum(uint64(sample) .* uint64((1:sampleN)'));
        key = sprintf('Q4_0_SIG_%s_%d_%u', mat2str(double(W.Dims)), n, checksum);
    end

    if isKey(q4CompCache, key)
        comp = q4CompCache(key);
        q = comp.q;
        d = comp.d;
    else
        [q, d] = W.get_q4_0_components();
        q4CompCache(key) = struct('q', q, 'd', d);
    end
end

function comp = get_q4_k_components_cached(W)
    persistent q4kCompCache
    if isempty(q4kCompCache)
        q4kCompCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    end

    key = build_weight_cache_key(W, 'Q4_K');
    if isKey(q4kCompCache, key)
        comp = q4kCompCache(key);
    else
        comp = W.get_q4_k_components();
        q4kCompCache(key) = comp;
    end
end

function comp = get_q6_k_components_cached(W)
    persistent q6kCompCache
    if isempty(q6kCompCache)
        q6kCompCache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    end

    key = build_weight_cache_key(W, 'Q6_K');
    if isKey(q6kCompCache, key)
        comp = q6kCompCache(key);
    else
        comp = W.get_q6_k_components();
        q6kCompCache(key) = comp;
    end
end

function key = build_weight_cache_key(W, prefix)
    if isprop(W, 'CacheKey') && ~isempty(W.CacheKey) && uint64(W.CacheKey) ~= 0
        key = [prefix '_CK_' char(string(uint64(W.CacheKey)))];
    else
        data_u8 = uint8(W.Data(:));
        n = numel(data_u8);
        sampleN = min(n, 256);
        sample = data_u8(1:sampleN);
        checksum = sum(uint64(sample) .* uint64((1:sampleN)'));
        key = sprintf('%s_SIG_%s_%d_%u', prefix, mat2str(double(W.Dims)), n, checksum);
    end
end
