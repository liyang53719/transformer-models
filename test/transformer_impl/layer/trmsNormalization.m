classdef trmsNormalization < matlab.unittest.TestCase
    properties(Constant)
        Eps = single(1e-6)
        ComparisonTolerance = single(1e-5)
    end

    methods(TestClassSetup)
        function addRepoToPath(~)
            repoRoot = iGetRepoRoot();
            addpath(repoRoot);
            addpath(fullfile(repoRoot, 'transformer_simulink', 'layer'));
        end
    end

    methods(Test)
        function matchesReferenceImplementation(test)
            rng(0);
            X = single(randn(64, 1536));
            g = single(randn(1, 1536));

            stim = transformer_impl.layer.createRmsNormalizationStimulus(X, g, test.Eps);
            actual = stim.output;
            expected = transformer.layer.rmsNormalization(X.', g.', test.Eps).';

            maxError = max(abs(actual - expected), [], 'all');
            test.verifyLessThanOrEqual(maxError, test.ComparisonTolerance);
        end

        function simulinkMatchesMatlabImplementation(test)
            repoRoot = iGetRepoRoot();
            addpath(repoRoot);
            addpath(fullfile(repoRoot, 'transformer_simulink', 'layer'));
            buildRmsNormalizationModel();

            modelName = 'rmsNormalization';
            modelPath = fullfile(repoRoot, 'transformer_simulink', 'layer', [modelName '.slx']);

            test.assertTrue(isfile(modelPath));

            load_system(modelPath);
            cleaner = onCleanup(@() close_system(modelName, 0));

            rng(1);
            X = single(randn(64, 1536));
            g = single(randn(1, 1536));
            epsilon = test.Eps;
            stim = transformer_impl.layer.createRmsNormalizationStimulus(X, g, epsilon);
            expected = transformer.layer.rmsNormalization(X.', g.', epsilon).';
            [ddrDataValidSeq, ddrDataBeatSeq, simStopTime] = iCreateSimulinkMemoryTrace(X);

            simIn = Simulink.SimulationInput(modelName);
            simIn = simIn.setVariable('simStopTime', simStopTime);
            simIn = simIn.setVariable('resetSeq', stim.resetSeq);
            simIn = simIn.setVariable('startSeq', stim.startSeq);
            simIn = simIn.setVariable('cfgGammaValidSeq', stim.cfgGammaValidSeq);
            simIn = simIn.setVariable('cfgGammaBeatSeq', stim.cfgGammaBeatSeq);
            simIn = simIn.setVariable('ddrDataValidSeq', ddrDataValidSeq);
            simIn = simIn.setVariable('ddrDataBeatSeq', ddrDataBeatSeq);
            simOut = sim(simIn);

            yBeatMatrix = squeeze(simOut.YBeatOut).';
            yValidMatrix = squeeze(simOut.YValidOut);
            validRows = yValidMatrix ~= 0;
            actualBeats = yBeatMatrix(validRows, :);
            actual = reshape(actualBeats.', 1536, 64).';
            maxError = max(abs(actual - expected), [], 'all');
            test.verifyLessThanOrEqual(maxError, test.ComparisonTolerance);

            clear cleaner;
        end
    end
end

function repoRoot = iGetRepoRoot()
thisFile = mfilename('fullpath');
repoRoot = fullfile(fileparts(thisFile), '..', '..', '..');
repoRoot = char(java.io.File(repoRoot).getCanonicalPath());
end

function [ddrDataValidSeq, ddrDataBeatSeq, simStopTime] = iCreateSimulinkMemoryTrace(X)
numTokens = 64;
hiddenSize = 1536;
lanesPerBeat = 8;
beatsPerToken = hiddenSize / lanesPerBeat;
flushGapCycles = 22;
readToReadStride = beatsPerToken + beatsPerToken + flushGapCycles + 3;
firstResponseIndex = 1 + beatsPerToken + 1 + 2;
lastResponseIndex = firstResponseIndex + (numTokens - 1) * readToReadStride + (beatsPerToken - 1);
tailCycles = beatsPerToken + flushGapCycles + 8;
traceLength = lastResponseIndex + tailCycles;

ddrValidTrace = false(traceLength, 1);
ddrBeatTrace = zeros(traceLength, lanesPerBeat, 'single');

for tokenIndex = 1:numTokens
    tokenStartIndex = firstResponseIndex + (tokenIndex - 1) * readToReadStride;
    for beatIndex = 1:beatsPerToken
        traceIndex = tokenStartIndex + beatIndex - 1;
        featureRange = (beatIndex - 1) * lanesPerBeat + (1:lanesPerBeat);
        ddrValidTrace(traceIndex) = true;
        ddrBeatTrace(traceIndex, :) = X(tokenIndex, featureRange);
    end
end

time = (0:traceLength-1).';
ddrDataValidSeq = timeseries(ddrValidTrace, time);
ddrDataBeatSeq = timeseries(ddrBeatTrace, time);
simStopTime = traceLength;
end