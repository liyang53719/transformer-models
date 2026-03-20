function modelPath = ensureRmsNormalizationModel(forceRebuild)
% ensureRmsNormalizationModel   Ensure the RMSNorm Simulink model exists and is loaded.

if nargin < 1
    forceRebuild = false;
end

thisDir = fileparts(mfilename('fullpath'));
modelName = 'rmsNormalization';
modelPath = fullfile(thisDir, [modelName '.slx']);

if forceRebuild
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
    modelPath = buildRmsNormalizationModel();
    load_system(modelName);
    return;
end

if bdIsLoaded(modelName)
    return;
end

if ~isfile(modelPath)
    modelPath = buildRmsNormalizationModel();
end

load_system(modelName);
end