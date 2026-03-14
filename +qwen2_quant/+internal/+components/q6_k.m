function comp = q6_k(raw_data, dims)
% q6_k   Extract Q6_K components for block simulation
%
%   comp = q6_k(raw_data, dims)
%   comp fields:
%     ql      [128, num_blocks] uint8
%     qh      [64, num_blocks] uint8
%     scales  [16, num_blocks] int8
%     d       [1, num_blocks] single

    block_size = 256;
    bytes_per_block = 210;

    num_elements = prod(dims);
    num_blocks = ceil(num_elements / block_size);

    raw_data = raw_data(1 : num_blocks * bytes_per_block);
    raw_matrix = reshape(raw_data, bytes_per_block, num_blocks);

    comp = struct();
    comp.ql = raw_matrix(1:128, :);
    comp.qh = raw_matrix(129:192, :);

    scales_bytes = raw_matrix(193:208, :);
    comp.scales = reshape(typecast(scales_bytes(:), 'int8'), 16, []);

    comp.d = decodeHalfRow(raw_matrix(209:210, :));
end

function f32 = decodeHalfRow(twoBytesByN)
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
