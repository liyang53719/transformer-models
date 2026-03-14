function weights = mxfp4(raw_data, dims)
% mxfp4   Dequantize MXFP4 (Microsoft Microscaling Format) to float32
%
%   weights = mxfp4(raw_data, dims)
%
%   MXFP4 Format (E2M1):
%     - Block size: 32 elements
%     - Per block: 1x uint8 shared_exponent + 16 bytes (32x 4-bit E2M1)
%     - E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
%
%   Inputs:
%       raw_data - uint8 array containing quantized data
%       dims     - Original dimensions [rows, cols]
%
%   Output:
%       weights  - Dequantized float32 array

    block_size = 32;
    bytes_per_block = 17; % 1 (shared exp) + 16 (packed 4-bit)
    
    num_elements = prod(dims);
    num_blocks = ceil(num_elements / block_size);
    
    weights = zeros(num_elements, 1, 'single');
    
    offset = 1;
    for b = 1:num_blocks
        block_start = (b-1) * bytes_per_block + 1;
        
        if block_start > length(raw_data)
            break;
        end
        
        % Read shared exponent
        shared_exp = double(raw_data(block_start));
        
        % Read and unpack E2M1 values
        packed_start = block_start + 1;
        packed_end = min(block_start + 16, length(raw_data));
        packed = uint8(raw_data(packed_start : packed_end));
        
        for i = 1:min(32, num_elements - offset + 1)
            byte_idx = ceil(i/2);
            if byte_idx > length(packed)
                break;
            end
            
            if mod(i, 2) == 1
                bits = bitand(packed(byte_idx), uint8(15)); % Low 4 bits
            else
                bits = bitshift(packed(byte_idx), -4);      % High 4 bits
            end
            
            % Parse E2M1: [S|EE|M]
            sign_bit = bitand(bits, uint8(8)) > 0;  % Highest bit
            exp_bits = double(bitshift(bitand(bits, uint8(6)), -1)); % Middle 2 bits
            mant_bit = double(bitand(bits, uint8(1)));      % Lowest bit
            
            % Special value handling
            if exp_bits == 0 && mant_bit == 0
                val = 0.0; % Zero
            else
                % Standard float: (-1)^S * (1.M) * 2^(E - bias - shared_exp)
                exp_val = exp_bits - shared_exp - 1;
                mantissa = 1.0 + mant_bit * 0.5;
                val = mantissa * (2.0 ^ exp_val);
                if sign_bit
                    val = -val;
                end
            end
            
            weights(offset + i - 1) = single(val);
        end
        
        offset = offset + block_size;
    end
    
    weights = reshape(weights, dims);
end
