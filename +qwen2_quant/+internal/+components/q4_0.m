function [qvals, deltas] = q4_0(raw_data, dims)
% q4_0   Extract Q4_0 components
%
%   [qvals, deltas] = q4_0(raw_data, dims)
%   qvals : [32, num_blocks] int8 in range [-8, 7]
%   deltas: [1, num_blocks] single

    block_size = 32;
    bytes_per_block = 18;

    num_elements = prod(dims);
    num_blocks = ceil(num_elements / block_size);

    raw_data = raw_data(1 : num_blocks * bytes_per_block);
    raw_matrix = reshape(raw_data, bytes_per_block, num_blocks);

    delta_bytes = raw_matrix(1:2, :);
    deltas = decodeHalfRow(delta_bytes);

    packed = raw_matrix(3:18, :);
    qlow = int8(bitand(packed, uint8(15))) - int8(8);
    qhigh = int8(bitshift(packed, -4)) - int8(8);

    qvals = zeros(32, num_blocks, 'int8');
    qvals(1:16, :) = qlow;
    qvals(17:32, :) = qhigh;
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
