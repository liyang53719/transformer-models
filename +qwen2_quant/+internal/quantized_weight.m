classdef quantized_weight
% quantized_weight   Container for quantized weight tensors
%
%   This class wraps quantized weight data and provides on-demand
%   dequantization for inference.
%
%   Properties:
%       QuantType   - 'Q8_0', 'Q4_0', 'Q4_K', 'Q6_K', 'MXFP4', or 'NONE'
%       Data        - Raw quantized data (uint8 array) or float array
%       Dims        - Original dimensions [rows, cols]
%       BlockSize   - Quantization block size (default 32)
    
    properties
        QuantType     % Quantization type string
        Data          % Raw quantized data or float data
        Dims          % Original dimensions
        BlockSize     % Block size for quantization
        NeedsTranspose = false  % Whether to transpose after dequantization
        CacheKey uint64 = uint64(0) % Stable per-instance key for runtime caches
    end
    
    methods
        function obj = quantized_weight(type, data, dims)
            % Constructor
            %   obj = quantized_weight(type, data, dims)
            %
            %   Inputs:
            %       type - 'Q8_0', 'Q4_0', 'Q4_K', 'Q6_K', 'MXFP4', or 'NONE'
            %       data - uint8 array (quantized) or numeric array (float)
            %       dims - Original dimensions of the weight tensor
            
            arguments
                type (1,:) char
                data
                dims (1,:) {mustBeNumeric}
            end
            
            obj.QuantType = type;
            obj.Data = data;
            obj.Dims = dims;
            obj.BlockSize = 32; % Default block size
            persistent NEXT_CACHE_KEY
            if isempty(NEXT_CACHE_KEY)
                NEXT_CACHE_KEY = uint64(1);
            end
            obj.CacheKey = NEXT_CACHE_KEY;
            NEXT_CACHE_KEY = NEXT_CACHE_KEY + uint64(1);
        end
        
        function W = dequantize(obj)
            % Dequantize the weight tensor to float32
            %   W = dequantize(obj)
            
            switch obj.QuantType
                case 'Q8_0'
                    W = qwen2_quant.internal.dequant.q8_0(obj.Data, obj.Dims);
                case 'Q4_0'
                    W = qwen2_quant.internal.dequant.q4_0(obj.Data, obj.Dims);
                case 'Q4_K'
                    W = qwen2_quant.internal.dequant.q4_k(obj.Data, obj.Dims);
                case 'Q6_K'
                    W = qwen2_quant.internal.dequant.q6_k(obj.Data, obj.Dims);
                case 'MXFP4'
                    W = qwen2_quant.internal.dequant.mxfp4(obj.Data, obj.Dims);
                case 'NONE'
                    W = obj.Data; % Already float
                otherwise
                    error('quantized_weight:UnsupportedType', ...
                        'Unsupported quantization type: %s', obj.QuantType);
            end
            
            % Transpose if needed for MATLAB convention
            if obj.NeedsTranspose && ndims(W) == 2
                W = W';
            end
        end
        
        function [q, s] = get_q8_0_components(obj)
            % Extract Q8_0 components (int8 weights and scale factors)
            %   [q, s] = get_q8_0_components(obj)
            %
            %   Outputs:
            %       q - [32, num_blocks] int8 matrix
            %       s - [1, num_blocks] single vector (scales)
            
            if ~strcmp(obj.QuantType, 'Q8_0')
                error('quantized_weight:InvalidType', 'Components only available for Q8_0');
            end
            
            [q, s] = qwen2_quant.internal.components.q8_0(obj.Data, obj.Dims);
        end

        function [q, d] = get_q4_0_components(obj)
            % Extract Q4_0 components
            %   q: [32, num_blocks] int8 in range [-8, 7]
            %   d: [1, num_blocks] single scales
            if ~strcmp(obj.QuantType, 'Q4_0')
                error('quantized_weight:InvalidType', 'Components only available for Q4_0');
            end

            [q, d] = qwen2_quant.internal.components.q4_0(obj.Data, obj.Dims);
        end

        function comp = get_q4_k_components(obj)
            % Extract Q4_K components
            %   comp contains parsed super-block fields for block simulation.
            if ~strcmp(obj.QuantType, 'Q4_K')
                error('quantized_weight:InvalidType', 'Components only available for Q4_K');
            end

            comp = qwen2_quant.internal.components.q4_k(obj.Data, obj.Dims);
        end

        function comp = get_q6_k_components(obj)
            % Extract Q6_K components
            if ~strcmp(obj.QuantType, 'Q6_K')
                error('quantized_weight:InvalidType', 'Components only available for Q6_K');
            end

            comp = qwen2_quant.internal.components.q6_k(obj.Data, obj.Dims);
        end
        
        function tf = isQuantized(obj)
            % Check if the weight is quantized
            %   tf = isQuantized(obj)
            
            tf = ~strcmp(obj.QuantType, 'NONE');
        end
        
        function sz = getMemorySize(obj)
            % Get memory size in bytes
            %   sz = getMemorySize(obj)
            
            info = whos('obj');
            sz = info.bytes;
        end
    end
end
