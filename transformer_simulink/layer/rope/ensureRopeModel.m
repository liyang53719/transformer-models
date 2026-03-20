function modelPath = ensureRopeModel(forceRebuild)
% ensureRopeModel   Ensure the RoPE Simulink model exists and is loaded.

if nargin < 1
    forceRebuild = false;
end

thisDir = fileparts(mfilename('fullpath'));
modelName = 'rope';
modelPath = fullfile(thisDir, [modelName '.slx']);

if forceRebuild
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
    modelPath = buildRopeModel();
    load_system(modelName);
    return;
end

if bdIsLoaded(modelName)
    return;
end

if ~isfile(modelPath)
    modelPath = buildRopeModel();
end

load_system(modelName);
end