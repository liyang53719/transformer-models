function [qvals, scales] = q8_0(raw_data, dims)
% q8_0   Extract Q8_0 components (Vectorized)
%
%   [qvals, scales] = q8_0(raw_data, dims)
%   qvals: [32, num_blocks] int8
%   scales: [1, num_blocks] single

    block_size = 32;
    bytes_per_block = 34; % 2 (float16) + 32 (int8)
    
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
    
    % Inf/NaN (Should not happen in valid weights generally, but good to handle)
    idx_inf = (exponent == 31) & (mantissa == 0);
    f32(idx_inf) = inf;
    idx_nan = (exponent == 31) & (mantissa ~= 0);
    f32(idx_nan) = nan;
    
    % Apply sign
    f32(sign == 1) = -f32(sign == 1);
    
    scales = f32'; % [1, num_blocks]
    
    % Extract qvals (bytes 3 to 34)
    qvals_bytes = raw_matrix(3:34, :);
    qvals = typecast(qvals_bytes(:), 'int8');
    qvals = reshape(qvals, 32, num_blocks);
end
