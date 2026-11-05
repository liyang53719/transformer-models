function parameters = load(modelFolder)
%LOAD - Load the Qwen3.5-0.8B text decoder as FP32 arrays
%   PARAMETERS = LOAD(modelFolder) reads local safetensors and config.json.
%   BF16 weights are expanded exactly to SINGLE. No quantization is used.
%   RMSNorm offset weights are converted to multiplicative weights once.
%
%   See also qwen3_5.model, qwen3_5.generate

cfg = jsondecode(fileread(fullfile(modelFolder, 'config.json')));
if isfield(cfg, 'text_config')
    cfg = cfg.text_config;
end
assert(strcmp(cfg.model_type, 'qwen3_5_text') && ...
    cfg.hidden_size == 1024 && cfg.num_hidden_layers == 24 && ...
    cfg.num_attention_heads == 8 && cfg.num_key_value_heads == 2 && ...
    cfg.linear_num_key_heads == 16 && cfg.linear_num_value_heads == 16 && ...
    cfg.linear_key_head_dim == 128 && cfg.linear_value_head_dim == 128, ...
    'qwen3_5:UnsupportedConfig', 'Expected Qwen3.5-0.8B text config.');

hp = struct('NumLayers', cfg.num_hidden_layers, ...
    'HiddenSize', cfg.hidden_size, 'IntermediateSize', cfg.intermediate_size, ...
    'NumHeads', cfg.num_attention_heads, 'NumKVHeads', cfg.num_key_value_heads, ...
    'HeadDim', cfg.head_dim, 'VocabSize', cfg.vocab_size, ...
    'RmsNormEpsilon', single(cfg.rms_norm_eps), ...
    'LinearKeyHeads', cfg.linear_num_key_heads, ...
    'LinearValueHeads', cfg.linear_num_value_heads, ...
    'LinearKeyDim', cfg.linear_key_head_dim, ...
    'LinearValueDim', cfg.linear_value_head_dim, ...
    'ConvKernel', cfg.linear_conv_kernel_dim, ...
    'RopeTheta', single(cfg.rope_parameters.rope_theta), ...
    'RotaryDim', cfg.head_dim * cfg.rope_parameters.partial_rotary_factor, ...
    'LayerTypes', string(cfg.layer_types), 'EosTokenId', cfg.eos_token_id, ...
    'GdnImplementation', 'auto', 'GdnChunkSize', 64);
[hp.EosTokenIds, hp.EosTokenId] = qwen3_5.internal.stopTokenIds(modelFolder);

indexFile = fullfile(modelFolder, 'model.safetensors.index.json');
if isfile(indexFile)
    index = jsondecode(fileread(indexFile));
    shardNames = unique(struct2cell(index.weight_map));
else
    shardNames = {'model.safetensors'};
end
shards = cell(size(shardNames));
for i = 1:numel(shardNames)
    filename = fullfile(modelFolder, shardNames{i});
    fid = fopen(filename, 'r', 'ieee-le');
    assert(fid >= 0, 'qwen3_5:MissingWeights', 'Cannot open %s.', filename);
    closer = onCleanup(@() fclose(fid));
    headerLength = fread(fid, 1, 'uint64=>double');
    header = jsondecode(char(fread(fid, headerLength, '*uint8').'));
    shards{i} = struct('File', filename, 'Offset', 8 + headerLength, ...
        'Header', header);
    clear closer
end

prefix = 'model.language_model.';
w = struct();
w.embed_tokens = iRead(shards, [prefix 'embed_tokens.weight'], true);
w.norm = iRead(shards, [prefix 'norm.weight']) + single(1);
for layer = 0:hp.NumLayers-1
    pfx = sprintf('%slayers.%d.', prefix, layer);
    lw.input_layernorm = iRead(shards, [pfx 'input_layernorm.weight']) + single(1);
    lw.post_attention_layernorm = iRead(shards, ...
        [pfx 'post_attention_layernorm.weight']) + single(1);
    for name = ["gate_proj", "up_proj", "down_proj"]
        lw.('mlp_' + name) = iRead(shards, [pfx 'mlp.' char(name) '.weight']);
    end
    if hp.LayerTypes(layer + 1) == "linear_attention"
        for name = ["in_proj_qkv", "in_proj_z", "in_proj_a", ...
                "in_proj_b", "out_proj", "conv1d", "norm"]
            lw.('linear_attn_' + name) = iRead(shards, ...
                [pfx 'linear_attn.' char(name) '.weight']);
        end
        lw.linear_attn_dt_bias = iRead(shards, [pfx 'linear_attn.dt_bias']);
        lw.linear_attn_A = -exp(iRead(shards, [pfx 'linear_attn.A_log']));
    else
        for name = ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]
            value = iRead(shards, [pfx 'self_attn.' char(name) '.weight']);
            if endsWith(name, '_norm')
                value = value + single(1);
            end
            lw.('self_attn_' + name) = value;
        end
    end
    w.(sprintf('h%d', layer)) = lw;
    clear lw
end
parameters = struct('Hyperparameters', hp, 'Weights', w, ...
    'Source', struct('Folder', char(modelFolder), ...
        'Shards', {shardNames}, 'Precision', 'single', ...
        'Scope', 'text decoder; tied embeddings; no vision or MTP'));
end

function value = iRead(shards, name, embedding)
if nargin < 3
    embedding = false;
end
key = matlab.lang.makeValidName(name);
for i = 1:numel(shards)
    if ~isfield(shards{i}.Header, key)
        continue
    end
    info = shards{i}.Header.(key);
    fid = fopen(shards{i}.File, 'r', 'ieee-le');
    closer = onCleanup(@() fclose(fid));
    assert(fseek(fid, shards{i}.Offset + info.data_offsets(1), 'bof') == 0, ...
        'qwen3_5:BadOffset', 'Invalid safetensors offset for %s.', name);
    count = prod(info.shape);
    switch info.dtype
        case 'BF16'
            raw = fread(fid, count, '*uint16');
            value = typecast(bitshift(uint32(raw), 16), 'single');
        case 'F32'
            value = fread(fid, count, '*single');
        otherwise
            error('qwen3_5:UnsupportedDtype', ...
                'Expected BF16 or F32, found %s in %s.', info.dtype, name);
    end
    assert(numel(value) == count, 'qwen3_5:TruncatedWeights', ...
        'Truncated tensor %s.', name);
    shape = double(info.shape(:).');
    if embedding
        value = reshape(value, fliplr(shape));
    elseif isscalar(shape)
        value = value(:);
    else
        value = reshape(value, fliplr(shape));
        value = permute(value, numel(shape):-1:1);
        if numel(shape) == 3 && shape(2) == 1
            value = reshape(value, shape(1), shape(3));
        end
    end
    return
end
error('qwen3_5:MissingTensor', 'Missing tensor %s.', name);
end
