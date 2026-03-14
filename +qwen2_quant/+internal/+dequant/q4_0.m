function weights = q4_0(raw_data, dims)
% q4_0   Dequantize Q4_0 format to float32 (Vectorized)
%
%   weights = q4_0(raw_data, dims)

    block_size = 32;
    bytes_per_block = 18; % 2 (float16) + 16 (packed 4-bit)
    
    num_elements = prod(dims);
    num_blocks = ceil(num_elements / block_size);
    
    % Ensure raw_data is exactly num_blocks * bytes_per_block
    raw_data = raw_data(1 : num_blocks * bytes_per_block);
    
    % Reshape into [bytes_per_block, num_blocks]
    raw_matrix = reshape(raw_data, bytes_per_block, num_blocks);
    
    % Extract delta bytes (first 2 bytes)
    delta_bytes = raw_matrix(1:2, :);
    
    % Convert delta bytes to uint16
    h16 = typecast(delta_bytes(:), 'uint16');
    
    % Convert uint16 to single (Vectorized)
    sign = bitshift(h16, -15);
    exponent = bitand(bitshift(h16, -10), uint16(31));
    mantissa = bitand(h16, uint16(1023));
    
    f32 = zeros(size(h16), 'single');
    
    % Normal numbers
    idx_normal = (exponent > 0) & (exponent < 31);
    f32(idx_normal) = (1.0 + single(mantissa(idx_normal)) / 1024.0) .* (2.^(single(exponent(idx_normal)) - 15));
    
    % Subnormal
    idx_sub = (exponent == 0) & (mantissa ~= 0);
    f32(idx_sub) = single(mantissa(idx_sub)) / 1024.0 * 2^(-14);
    
    % Inf/NaN
    idx_inf = (exponent == 31) & (mantissa == 0);
    f32(idx_inf) = inf;
    idx_nan = (exponent == 31) & (mantissa ~= 0);
    f32(idx_nan) = nan;
    
    % Apply sign
    f32(sign == 1) = -f32(sign == 1);
    
    delta = f32'; % [1, num_blocks]
    
    % Extract packed 4-bit values (bytes 3 to 18)
    packed_bytes = raw_matrix(3:18, :); % [16, num_blocks]
    
    % Unpack 4-bit values
    % Low 4 bits
    qvals_low = single(bitand(packed_bytes, uint8(15))) - 8.0;
    % High 4 bits
    qvals_high = single(bitshift(packed_bytes, -4)) - 8.0;
    
    % Interleave low and high
    % In llama.cpp, q4_0 is stored as:
    % qs[i] = low, qs[i+16] = high
    % Wait, let's check llama.cpp q4_0 format.
    % In llama.cpp:
    % for (int i = 0; i < 16; ++i) {
    %     float v0 = (x[i].qs[j] & 0x0F) - 8;
    %     float v1 = (x[i].qs[j] >> 4)   - 8;
    %     y[i*32 + j + 0 ] = v0 * d;
    %     y[i*32 + j + 16] = v1 * d;
    % }
    % So low is 0..15, high is 16..31.
    
    qvals = zeros(32, num_blocks, 'single');
    qvals(1:16, :) = qvals_low;
    qvals(17:32, :) = qvals_high;
    
    % Multiply
    weights = qvals .* delta; % [32, num_blocks] .* [1, num_blocks]
    
    % Flatten and truncate
    weights = weights(:);
    weights = weights(1:num_elements);
    
    % Reshape to original dims
    weights = reshape(weights, dims);
end
