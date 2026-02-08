function parameters = load(filepath)
% load   Load Llama Parameters
%
%   parameters = load(filepath) loads the TinyLlama parameters from a .mat file
%   and structures them for use with the llama.model function.

    % 1. Load raw variables
    raw = load(filepath);
    
    parameters = struct;
    parameters.Hyperparameters = struct;
    parameters.Weights = struct;
    
    % 2. Extract Hyperparameters
    hpFields = {'NumLayers', 'HiddenSize', 'NumHeads', 'NumKVHeads', 'HeadDim', 'VocabSize', 'RopeTheta'};
    for i = 1:numel(hpFields)
        if isfield(raw, hpFields{i})
            parameters.Hyperparameters.(hpFields{i}) = double(raw.(hpFields{i}));
            raw = rmfield(raw, hpFields{i});
        end
    end
    
    % Default RoPE Theta if missing
    if ~isfield(parameters.Hyperparameters, 'RopeTheta')
        parameters.Hyperparameters.RopeTheta = 10000.0;
    end

    % 3. Structure Weights
    % Llama weights are flat in the .mat file (e.g., layer_0_self_attn_q_proj)
    % We want to organize them: weights.h0.self_attn_q_proj
    % Also: embed_tokens, norm, lm_head
    
    fields = fieldnames(raw);
    for i = 1:numel(fields)
        name = fields{i};
        val = raw.(name);
        
        % Check for layer weights
        if startsWith(name, 'layer_')
            % Parse layer index: layer_XX_...
            parts = split(name, '_');
            layerIdx = str2double(parts{2}); % layer_0 -> 0
            
            % Handle internal structure
            % layer_0_self_attn_q_proj -> h0.self_attn_q_proj
            % OR better: keep the suffix as the field name
            % suffix = join(parts(3:end), '_');
            
            layerName = sprintf('h%d', layerIdx);
            
            suffix = strjoin(parts(3:end), '_');
            
            parameters.Weights.(layerName).(suffix) = dlarray(single(val));
        else
            % Global weights: embed_tokens, norm, lm_head
            parameters.Weights.(name) = dlarray(single(val));
        end
    end

end
