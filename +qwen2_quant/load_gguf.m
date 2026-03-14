function parameters = load_gguf(filepath, options)
% load_gguf   Load Qwen2 Parameters from GGUF file
%
%   parameters = load_gguf(filepath)
%   parameters = load_gguf(filepath, 'DequantizeNow', true)
%
%   Inputs:
%       filepath - Path to .gguf file
%       Name-Value Arguments:
%           DequantizeNow - If true, dequantize all weights immediately (default: false)
%           Verbose       - Print loading progress (default: true)
%
%   Output:
%       parameters - Struct with .Hyperparameters and .Weights

    arguments
        filepath {mustBeFile}
        options.DequantizeNow (1,1) logical = false
        options.Verbose (1,1) logical = true
    end
    
    if options.Verbose
        fprintf('Loading GGUF file: %s\n', filepath);
    end
    
    % Open GGUF file
    reader = qwen2_quant.internal.gguf_reader(filepath, options.Verbose);
    
    % Extract hyperparameters
    parameters = struct();
    parameters.Hyperparameters = struct();
    
    meta = reader.Metadata;
    
    % Detect architecture
    if isfield(meta, 'qwen2')
        arch = meta.qwen2;
    elseif isfield(meta, 'llama')
        arch = meta.llama;
    else
        error('load_gguf:UnknownArch', 'Unknown architecture in GGUF file');
    end
    
    % Extract standard hyperparameters
    parameters.Hyperparameters.NumLayers = double(arch.block_count);
    parameters.Hyperparameters.HiddenSize = double(arch.embedding_length);
    parameters.Hyperparameters.NumHeads = double(arch.attention.head_count);
    
    if isfield(arch.attention, 'head_count_kv')
        parameters.Hyperparameters.NumKVHeads = double(arch.attention.head_count_kv);
    else
        parameters.Hyperparameters.NumKVHeads = parameters.Hyperparameters.NumHeads;
    end
    
    parameters.Hyperparameters.HeadDim = parameters.Hyperparameters.HiddenSize / parameters.Hyperparameters.NumHeads;
    
    if isfield(meta, 'tokenizer') && isfield(meta.tokenizer, 'ggml') && isfield(meta.tokenizer.ggml, 'tokens')
        parameters.Hyperparameters.VocabSize = length(meta.tokenizer.ggml.tokens);
    elseif isfield(arch, 'vocab_size')
        parameters.Hyperparameters.VocabSize = double(arch.vocab_size);
    else
        parameters.Hyperparameters.VocabSize = 151936; % Qwen2 default
    end
    
    if isfield(arch, 'rope') && isfield(arch.rope, 'freq_base')
        parameters.Hyperparameters.RopeTheta = double(arch.rope.freq_base);
    else
        parameters.Hyperparameters.RopeTheta = 1000000.0; % Qwen2 default
    end
    
    % Load weights
    parameters.Weights = struct();
    
    if options.Verbose
        fprintf('Loading %d tensors...\n', length(reader.TensorInfo));
    end
    
    for i = 1:length(reader.TensorInfo)
        info = reader.TensorInfo(i);
        
        if options.Verbose && mod(i, 50) == 0
            fprintf('  Progress: %d/%d\n', i, length(reader.TensorInfo));
        end
        
        % Read tensor (possibly quantized)
        tensor = reader.readTensor(info);
        
        % Map GGUF name to MATLAB structure first
        matlab_name = mapGGUFNameToMatlab(info.name);
        
        % Determine if this weight needs transpose
        % GGUF already stores dimensions in MATLAB-compatible order
        % Linear weights still need transpose for W*X convention
        is_2d_weight = length(info.dims) == 2;
        % For linear weights [out, in], we get [in, out], need transpose to [out, in]
        % For embeddings [vocab, hidden], we get [hidden, vocab], already correct!
        needs_transpose = is_2d_weight && ~contains(matlab_name, 'embed_tokens') && ...
            ~contains(matlab_name, 'norm');
        
        if tensor.isQuantized()
            % Set transpose flag for quantized 2D weights (except embeddings)
            if needs_transpose
                tensor.NeedsTranspose = true;
            end
        end
        
        % Materialize tensor data
        if tensor.isQuantized()
            if options.DequantizeNow
                tensor_data = dlarray(single(tensor.dequantize()));
            else
                tensor_data = tensor;
            end
        else
            % F32/F16 tensors are stored as quantized_weight('NONE', ...).
            % Always unwrap to numeric to avoid passing wrapper objects into model code.
            data = single(tensor.dequantize());
            if needs_transpose
                data = data';
            end
            tensor_data = dlarray(data);
        end
        
        % Store in hierarchical structure
        matlab_name = mapGGUFNameToMatlab(info.name);
        
        % Store in hierarchical structure
        if startsWith(matlab_name, 'h')
            % Layer weight: h0.self_attn_q_proj
            parts = strsplit(matlab_name, '.');
            layer_name = parts{1};
            weight_name = strjoin(parts(2:end), '_');
            
            if ~isfield(parameters.Weights, layer_name)
                parameters.Weights.(layer_name) = struct();
            end
            parameters.Weights.(layer_name).(weight_name) = tensor_data;
        else
            % Global weight: embed_tokens, lm_head, norm
            parameters.Weights.(matlab_name) = tensor_data;
        end
    end
    
    delete(reader);
    
    if options.Verbose
        fprintf('GGUF loading complete.\n');
    end
end

function matlab_name = mapGGUFNameToMatlab(gguf_name)
    % Map GGUF tensor names to MATLAB structure names
    
    % Remove .weight suffix
    name = strrep(gguf_name, '.weight', '');
    
    % Map global tensors
    if strcmp(name, 'token_embd')
        matlab_name = 'embed_tokens';
    elseif strcmp(name, 'output_norm')
        matlab_name = 'norm';
    elseif strcmp(name, 'output')
        matlab_name = 'lm_head';
    elseif startsWith(name, 'blk.')
        % blk.5.attn_q -> h5.self_attn_q_proj
        tokens = strsplit(name, '.');
        layer_num = tokens{2};
        component = strjoin(tokens(3:end), '_');
        
        % Map component names to Qwen2 convention
        component = strrep(component, 'attn_q', 'self_attn_q_proj');
        component = strrep(component, 'attn_k', 'self_attn_k_proj');
        component = strrep(component, 'attn_v', 'self_attn_v_proj');
        component = strrep(component, 'attn_output', 'self_attn_o_proj');
        component = strrep(component, 'self_attn_q_proj_bias', 'self_attn_q_bias');
        component = strrep(component, 'self_attn_k_proj_bias', 'self_attn_k_bias');
        component = strrep(component, 'self_attn_v_proj_bias', 'self_attn_v_bias');
        component = strrep(component, 'self_attn_o_proj_bias', 'self_attn_o_bias');
        component = strrep(component, 'attn_norm', 'input_layernorm');
        component = strrep(component, 'ffn_gate', 'mlp_gate_proj');
        component = strrep(component, 'ffn_up', 'mlp_up_proj');
        component = strrep(component, 'ffn_down', 'mlp_down_proj');
        component = strrep(component, 'ffn_norm', 'post_attention_layernorm');
        
        matlab_name = sprintf('h%s.%s', layer_num, component);
    else
        matlab_name = name;
    end
end
