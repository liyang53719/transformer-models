function comp = q4_k(raw_data, dims)
% q4_k   Extract Q4_K components for block simulation
%
%   comp = q4_k(raw_data, dims)
%   comp fields:
%     d       [1, num_blocks] single
%     dmin    [1, num_blocks] single
%     scales  [8, num_blocks] uint8
%     mins    [8, num_blocks] uint8
%     qs      [128, num_blocks] uint8

    block_size = 256;
    bytes_per_block = 144;

    num_elements = prod(dims);
    num_blocks = ceil(num_elements / block_size);

    raw_data = raw_data(1 : num_blocks * bytes_per_block);
    raw_matrix = reshape(raw_data, bytes_per_block, num_blocks);

    comp = struct();
    comp.d = decodeHalfRow(raw_matrix(1:2, :));
    comp.dmin = decodeHalfRow(raw_matrix(3:4, :));
    comp.qs = raw_matrix(17:144, :);

    packed = raw_matrix(5:16, :);
    scales = zeros(8, num_blocks, 'uint8');
    mins = zeros(8, num_blocks, 'uint8');

    for j = 0:7
        if j < 4
            scales(j+1, :) = bitand(packed(j+1, :), uint8(63));
            mins(j+1, :) = bitand(packed(j+5, :), uint8(63));
        else
            scales(j+1, :) = bitor(bitand(packed(j+5, :), uint8(15)), bitshift(bitand(packed(j-3, :), uint8(192)), -2));
            mins(j+1, :) = bitor(bitshift(packed(j+5, :), -4), bitshift(bitand(packed(j+1, :), uint8(192)), -2));
        end
    end

    comp.scales = scales;
    comp.mins = mins;
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
