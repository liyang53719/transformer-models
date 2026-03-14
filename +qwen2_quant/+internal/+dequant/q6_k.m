function weights = q6_k(raw_data, dims)
% q6_k   Dequantize GGML Q6_K format to float32
%
%   weights = q6_k(raw_data, dims)
%
%   Reference: llama.cpp dequantize_row_q6_K

    block_size = 256;
    bytes_per_block = 210; % 128(ql) + 64(qh) + 16(scales) + 2(d)

    num_elements = prod(dims);
    num_blocks = ceil(num_elements / block_size);

    raw_data = raw_data(1 : num_blocks * bytes_per_block);
    raw_matrix = reshape(raw_data, bytes_per_block, num_blocks);

    ql_all = raw_matrix(1:128, :);
    qh_all = raw_matrix(129:192, :);
    scales_bytes = raw_matrix(193:208, :);
    scales_all = reshape(typecast(scales_bytes(:), 'int8'), 16, []);
    d_all = decodeHalfVector(raw_matrix(209:210, :));

    out = zeros(block_size, num_blocks, 'single');

    for ib = 1:num_blocks
        ql = ql_all(:, ib);
        qh = qh_all(:, ib);
        sc = single(scales_all(:, ib));
        d = d_all(ib);

        block = zeros(block_size, 1, 'single');

        ql_ptr = 1;
        qh_ptr = 1;
        sc_ptr = 1;
        out_ptr = 1;

        for n = 1:2 % each iteration outputs 128 values
            ql64 = ql(ql_ptr:ql_ptr+63);
            qh32 = qh(qh_ptr:qh_ptr+31);
            sc8 = sc(sc_ptr:sc_ptr+7);

            for l = 0:31
                qh_v = qh32(l+1);

                q1 = int16(bitand(ql64(l+1), uint8(15))) + bitshift(int16(bitand(bitshift(qh_v, 0), uint8(3))), 4) - 32;
                q2 = int16(bitand(ql64(l+33), uint8(15))) + bitshift(int16(bitand(bitshift(qh_v, -2), uint8(3))), 4) - 32;
                q3 = int16(bitshift(ql64(l+1), -4)) + bitshift(int16(bitand(bitshift(qh_v, -4), uint8(3))), 4) - 32;
                q4 = int16(bitshift(ql64(l+33), -4)) + bitshift(int16(bitand(bitshift(qh_v, -6), uint8(3))), 4) - 32;

                is = floor(l / 16);
                block(out_ptr + l) = d * sc8(is + 1) * single(q1);
                block(out_ptr + l + 32) = d * sc8(is + 3) * single(q2);
                block(out_ptr + l + 64) = d * sc8(is + 5) * single(q3);
                block(out_ptr + l + 96) = d * sc8(is + 7) * single(q4);
            end

            out_ptr = out_ptr + 128;
            ql_ptr = ql_ptr + 64;
            qh_ptr = qh_ptr + 32;
            sc_ptr = sc_ptr + 8;
        end

        out(:, ib) = block;
    end

    weights = out(:);
    weights = weights(1:num_elements);
    weights = reshape(weights, dims);
end

function f32 = decodeHalfVector(twoBytesByN)
    h16 = typecast(twoBytesByN(:), 'uint16');

    signBits = bitshift(h16, -15);
    expBits = bitand(bitshift(h16, -10), uint16(31));
    manBits = bitand(h16, uint16(1023));

    f32 = zeros(size(h16), 'single');

    idxNormal = (expBits > 0) & (expBits < 31);
    f32(idxNormal) = (1.0 + single(manBits(idxNormal)) / 1024.0) .* (2.^(single(expBits(idxNormal)) - 15));

    idxSub = (expBits == 0) & (manBits ~= 0);
    f32(idxSub) = single(manBits(idxSub)) / 1024.0 * 2^(-14);

    idxInf = (expBits == 31) & (manBits == 0);
    f32(idxInf) = inf;
    idxNaN = (expBits == 31) & (manBits ~= 0);
    f32(idxNaN) = nan;

    f32(signBits == 1) = -f32(signBits == 1);
    f32 = reshape(f32, 1, []);
end
