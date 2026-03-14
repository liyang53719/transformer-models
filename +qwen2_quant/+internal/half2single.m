function f32 = half2single(bytes)
% half2single   Convert float16 bytes to single precision
%
%   f32 = half2single(bytes)
%
%   Inputs:
%       bytes - 2-element uint8 array representing float16 in little-endian
%
%   Output:
%       f32 - Single precision float

    % Combine bytes (little-endian)
    h16 = uint16(bytes(1)) + bitshift(uint16(bytes(2)), 8);
    
    % Extract components
    sign = bitshift(h16, -15);
    exponent = bitand(bitshift(h16, -10), uint16(31));
    mantissa = bitand(h16, uint16(1023));
    
    % Convert to float32
    if exponent == 0
        if mantissa == 0
            % Zero
            f32 = 0;
        else
            % Subnormal
            f32 = single(mantissa) / 1024.0 * 2^(-14);
        end
    elseif exponent == 31
        % Inf or NaN
        if mantissa == 0
            f32 = inf;
        else
            f32 = nan;
        end
    else
        % Normal number
        f32 = (1.0 + single(mantissa) / 1024.0) * 2^(double(exponent) - 15);
    end
    
    % Apply sign
    if sign == 1
        f32 = -f32;
    end
    
    f32 = single(f32);
end
