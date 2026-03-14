classdef gguf_reader < handle
% gguf_reader   GGUF binary file reader for quantized models
%
%   This class reads GGUF (GGML Unified Format) files containing
%   quantized LLM weights.
%
%   Example:
%       reader = qwen2_quant.internal.gguf_reader('model.gguf');
%       tensor = reader.readTensor(reader.TensorInfo(1));
%       delete(reader);
    
    properties
        FilePath
        FileID
        Header
        Metadata
        TensorInfo
        DataStartOffset  % Offset where tensor data begins
        Verbose (1,1) logical = true
    end
    
    methods
        function obj = gguf_reader(filepath, verbose)
            % Constructor - opens and parses GGUF file header
            %   obj = gguf_reader(filepath)

            if nargin < 2
                verbose = true;
            end
            
            obj.FilePath = filepath;
            obj.Verbose = verbose;
            obj.FileID = fopen(filepath, 'r', 'l'); % Little-endian
            
            if obj.FileID == -1
                error('gguf_reader:FileNotFound', ...
                    'Cannot open GGUF file: %s', filepath);
            end
            
            obj.readHeader();
            obj.readMetadata();
            obj.readTensorInfo();
        end
        
        function readHeader(obj)
            % Read GGUF file header
            
            % Magic number: 0x46554747 ("GGUF")
            magic = fread(obj.FileID, 1, 'uint32');
            if magic ~= hex2dec('46554747')
                error('gguf_reader:InvalidMagic', ...
                    'Invalid GGUF magic number: 0x%08X', magic);
            end
            
            obj.Header.version = fread(obj.FileID, 1, 'uint32');
            obj.Header.tensor_count = fread(obj.FileID, 1, 'uint64');
            obj.Header.metadata_kv_count = fread(obj.FileID, 1, 'uint64');
            
            if obj.Verbose
                fprintf('GGUF v%d: %d tensors, %d metadata entries\n', ...
                    obj.Header.version, obj.Header.tensor_count, ...
                    obj.Header.metadata_kv_count);
            end
        end
        
        function readMetadata(obj)
            % Read metadata key-value pairs
            
            obj.Metadata = struct();
            for i = 1:obj.Header.metadata_kv_count
                key = obj.readString();
                value_type = fread(obj.FileID, 1, 'uint32');
                value = obj.readValue(value_type);
                
                % Store in hierarchical structure
                obj.Metadata = setNestedField(obj.Metadata, key, value);
            end
        end
        
        function str = readString(obj)
            % Read a GGUF string (length-prefixed)
            
            len = fread(obj.FileID, 1, 'uint64');
            if isempty(len) || len == 0
                str = '';
                return;
            end
            str = char(fread(obj.FileID, len, 'uint8=>char')');
        end
        
        function value = readValue(obj, value_type)
            % Read a value based on GGUF type
            
            switch value_type
                case 0  % UINT8
                    value = fread(obj.FileID, 1, 'uint8');
                case 4  % UINT32
                    value = fread(obj.FileID, 1, 'uint32');
                case 5  % INT32
                    value = fread(obj.FileID, 1, 'int32');
                case 6  % FLOAT32
                    value = fread(obj.FileID, 1, 'float32');
                case 7  % BOOL
                    value = fread(obj.FileID, 1, 'uint8') ~= 0;
                case 8  % STRING
                    value = obj.readString();
                case 10 % UINT64
                    value = fread(obj.FileID, 1, 'uint64');
                case 11 % INT64
                    value = fread(obj.FileID, 1, 'int64');
                case 12 % FLOAT64
                    value = fread(obj.FileID, 1, 'float64');
                case 9  % ARRAY
                    array_type = fread(obj.FileID, 1, 'uint32');
                    array_len = fread(obj.FileID, 1, 'uint64');
                    value = cell(array_len, 1);
                    for i = 1:array_len
                        value{i} = obj.readValue(array_type);
                    end
                otherwise
                    error('gguf_reader:UnsupportedType', ...
                        'Unsupported GGUF value type: %d', value_type);
            end
        end
        
        function readTensorInfo(obj)
            % Read tensor information
            
            obj.TensorInfo = struct('name', {}, 'type', {}, 'dims', {}, 'offset', {});
            
            for i = 1:obj.Header.tensor_count
                info = struct();
                info.name = obj.readString();
                n_dims = fread(obj.FileID, 1, 'uint32');
                info.dims = fread(obj.FileID, n_dims, 'uint64')';
                info.type = fread(obj.FileID, 1, 'uint32');
                info.offset = fread(obj.FileID, 1, 'uint64');
                
                obj.TensorInfo(i) = info;
            end
            
            % Calculate tensor data start offset with alignment
            alignment = 32;
            if isfield(obj.Metadata, 'general') && isfield(obj.Metadata.general, 'alignment')
                alignment = double(obj.Metadata.general.alignment);
            end
            header_end = ftell(obj.FileID);
            obj.DataStartOffset = ceil(header_end / alignment) * alignment;
        end
        
        function tensor = readTensor(obj, tensor_info)
            % Read and dequantize a specific tensor
            %   tensor = readTensor(obj, tensor_info)
            
            % Seek to tensor position using pre-calculated data start offset
            fseek(obj.FileID, obj.DataStartOffset + tensor_info.offset, 'bof');
            
            % Read tensor data
            tensor = obj.readTensorData(tensor_info);
        end
        
        function tensor = readTensorData(obj, info)
            % Read tensor data based on type
            
            num_elements = prod(info.dims);
            orig_dims = info.dims;
            
            % For MATLAB compatibility, ensure at least 2D
            if length(orig_dims) == 1
                dims_for_reshape = [orig_dims, 1];
            else
                dims_for_reshape = orig_dims;
            end
            
            switch info.type
                case 0 % F32
                    data = fread(obj.FileID, num_elements, 'float32');
                    if length(data) ~= num_elements
                        error('F32 read mismatch: got %d, expected %d', length(data), num_elements);
                    end
                    data_reshaped = reshape(single(data), dims_for_reshape);
                    tensor = qwen2_quant.internal.quantized_weight('NONE', ...
                        data_reshaped, orig_dims);
                    
                case 1 % F16
                    raw = fread(obj.FileID, num_elements*2, '*uint8');
                    % Convert float16 bytes to float32
                    data = zeros(num_elements, 1, 'single');
                    for i = 1:num_elements
                        data(i) = qwen2_quant.internal.half2single(raw(2*i-1:2*i));
                    end
                    data_reshaped = reshape(data, dims_for_reshape);
                    tensor = qwen2_quant.internal.quantized_weight('NONE', ...
                        data_reshaped, orig_dims);
                    
                case 2 % Q4_0
                    block_size = 32;
                    bytes_per_block = 18;
                    num_blocks = ceil(num_elements / block_size);
                    raw_data = fread(obj.FileID, num_blocks * bytes_per_block, '*uint8');
                    tensor = qwen2_quant.internal.quantized_weight('Q4_0', ...
                        raw_data, orig_dims);

                case 12 % Q4_K
                    block_size = 256;
                    bytes_per_block = 144;
                    num_blocks = ceil(num_elements / block_size);
                    raw_data = fread(obj.FileID, num_blocks * bytes_per_block, '*uint8');
                    tensor = qwen2_quant.internal.quantized_weight('Q4_K', ...
                        raw_data, orig_dims);

                case 14 % Q6_K
                    block_size = 256;
                    bytes_per_block = 210;
                    num_blocks = ceil(num_elements / block_size);
                    raw_data = fread(obj.FileID, num_blocks * bytes_per_block, '*uint8');
                    tensor = qwen2_quant.internal.quantized_weight('Q6_K', ...
                        raw_data, orig_dims);
                    
                case 8 % Q8_0
                    block_size = 32;
                    bytes_per_block = 34;
                    num_blocks = ceil(num_elements / block_size);
                    raw_data = fread(obj.FileID, num_blocks * bytes_per_block, '*uint8');
                    tensor = qwen2_quant.internal.quantized_weight('Q8_0', ...
                        raw_data, orig_dims);
                    
                case 39 % MXFP4
                    block_size = 32;
                    bytes_per_block = 17;
                    num_blocks = ceil(num_elements / block_size);
                    raw_data = fread(obj.FileID, num_blocks * bytes_per_block, '*uint8');
                    tensor = qwen2_quant.internal.quantized_weight('MXFP4', ...
                        raw_data, orig_dims);
                    
                otherwise
                    error('gguf_reader:UnsupportedTensorType', ...
                        'Unsupported ggml_type: %d for tensor %s', ...
                        info.type, info.name);
            end
        end
        
        function delete(obj)
            % Destructor - close file
            if obj.FileID ~= -1
                fclose(obj.FileID);
            end
        end
    end
end

function s = setNestedField(s, key, value)
    % Helper to set nested struct fields from dot-separated key
    % Example: key = "qwen2.attention.head_count" creates:
    %          s.qwen2.attention.head_count = value
    
    parts = strsplit(key, '.');
    
    if length(parts) == 1
        % Base case: single field
        field = matlab.lang.makeValidName(parts{1});
        s.(field) = value;
    else
        % Recursive case: navigate into nested structure
        field = matlab.lang.makeValidName(parts{1});
        
        % Create field if it doesn't exist
        if ~isfield(s, field)
            s.(field) = struct();
        end
        
        % Recursively set nested field
        remaining_key = strjoin(parts(2:end), '.');
        s.(field) = setNestedField(s.(field), remaining_key, value);
    end
end
