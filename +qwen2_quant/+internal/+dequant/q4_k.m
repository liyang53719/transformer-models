function weights = q4_k(raw_data, dims)
% q4_k   Dequantize GGML Q4_K format to float32
%
%   weights = q4_k(raw_data, dims)
%
%   Reference: llama.cpp dequantize_row_q4_K

    block_size = 256;
    bytes_per_block = 144; % 2(d) + 2(dmin) + 12(scales) + 128(qs)

    num_elements = prod(dims);
    num_blocks = ceil(num_elements / block_size);

    raw_data = raw_data(1 : num_blocks * bytes_per_block);
    raw_matrix = reshape(raw_data, bytes_per_block, num_blocks);

    d = decodeHalfVector(raw_matrix(1:2, :));
    dmin = decodeHalfVector(raw_matrix(3:4, :));

    scales_raw = raw_matrix(5:16, :);      % [12, nb] uint8
    qbytes = raw_matrix(17:144, :);        % [128, nb] uint8

    out = zeros(block_size, num_blocks, 'single');

    for ib = 1:num_blocks
        [sc, mn] = unpackK4ScaleMin(scales_raw(:, ib)); % each [8,1]
        q = qbytes(:, ib);

        is = 1;
        qoff = 1;
        for seg = 0:3
            q32 = q(qoff:qoff+31);
            qoff = qoff + 32;

            lo = single(bitand(q32, uint8(15)));
            hi = single(bitshift(q32, -4));

            d1 = d(ib) * single(sc(is));
            m1 = dmin(ib) * single(mn(is));
            d2 = d(ib) * single(sc(is+1));
            m2 = dmin(ib) * single(mn(is+1));

            base = seg * 64;
            out(base + (1:32), ib) = d1 .* lo - m1;
            out(base + 32 + (1:32), ib) = d2 .* hi - m2;

            is = is + 2;
        end
    end

    weights = out(:);
    weights = weights(1:num_elements);
    weights = reshape(weights, dims);
end

function [d, m] = unpackK4ScaleMin(q)
% q is [12,1] uint8
% Equivalent to llama.cpp get_scale_min_k4
    d = zeros(8, 1, 'uint8');
    m = zeros(8, 1, 'uint8');

    for j = 0:7
        if j < 4
            d(j+1) = bitand(q(j+1), uint8(63));
            m(j+1) = bitand(q(j+5), uint8(63));
        else
            d(j+1) = bitor(bitand(q(j+5), uint8(15)), bitshift(bitand(q(j-3), uint8(192)), -2));
            m(j+1) = bitor(bitshift(q(j+5), -4), bitshift(bitand(q(j+1), uint8(192)), -2));
        end
    end
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
