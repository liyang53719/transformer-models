function [logits, state, trace] = model(X, parameters, state)
%MODEL - Evaluate the Qwen3.5-0.8B text decoder in FP32
%   LOGITS = MODEL(X,PARAMETERS) evaluates a nonempty vector of zero-based
%   token IDs and returns vocab-by-time logits. All 24 layers execute.
%
%   LOGITS = MODEL(X,PARAMETERS,STATE) continues an existing sequence.
%
%   [LOGITS,STATE] = MODEL(...) also returns KV, convolution, and SSM state.
%
%   [LOGITS,STATE,TRACE] = MODEL(...) also returns layer operator outputs.
%
%   See also qwen3_5.load, qwen3_5.generate

hp = parameters.Hyperparameters;
validateattributes(X, {'numeric'}, ...
    {'vector', 'nonempty', 'real', 'finite', 'integer', '>=', 0, '<', hp.VocabSize});
if nargin < 3 || isempty(state)
    state = struct('Position', 0, 'Layers', {cell(hp.NumLayers, 1)});
end
assert(numel(state.Layers) == hp.NumLayers && ...
    state.Position >= 0 && state.Position == fix(state.Position), ...
    'qwen3_5:InvalidState', 'Invalid decoder state.');
w = parameters.Weights;
hidden = w.embed_tokens(:, double(X(:).') + 1);
if nargout > 2
    trace = struct('Embedding', hidden, 'Layers', {cell(hp.NumLayers, 1)});
end
for layer = 1:hp.NumLayers
    [hidden, state.Layers{layer}, layerTrace] = qwen3_5.layer.block( ...
        hidden, state.Layers{layer}, w.(sprintf('h%d', layer-1)), ...
        hp, layer, state.Position);
    if nargout > 2
        trace.Layers{layer} = layerTrace;
    end
end
hidden = qwen3_5.layer.rmsNormalization(hidden, w.norm, hp.RmsNormEpsilon);
logits = w.embed_tokens.' * hidden;
state.Position = state.Position + numel(X);
if nargout > 2
    trace.result_norm = hidden;
end
end
