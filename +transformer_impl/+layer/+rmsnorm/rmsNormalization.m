function [outBeat, outValid, ddrReadAddr, ddrReadEn, done, busy] = rmsNormalization( ...
    reset, start, cfgGammaBeat, cfgGammaValid, ddrDataBeat, ddrDataValid, epsilon)
% rmsNormalization   Streaming RMSNorm DUT for HDL Coder.
%#codegen

numTokens = coder.const(uint8(64));
hiddenSize = coder.const(uint16(1536));
lanesPerBeat = coder.const(uint8(8));
beatsPerToken = coder.const(uint16(192));

if nargin < 7
    epsilon = single(1e-6);
end

persistent state gammaMem tokenMem gammaWriteBeat gammaLoaded tokenIndex
persistent requestBeatIndex receiveBeatIndex outputBeatIndex currentSum invRmsLatched

if isempty(state)
    [state, gammaMem, tokenMem, gammaWriteBeat, gammaLoaded, tokenIndex, ...
        requestBeatIndex, receiveBeatIndex, outputBeatIndex, currentSum, invRmsLatched] = iResetState();
end

outBeat = zeros(1, double(lanesPerBeat), 'single');
outValid = false;
ddrReadAddr = uint16(0);
ddrReadEn = false;
done = false;
busy = state ~= uint8(0);

if reset
    [state, gammaMem, tokenMem, gammaWriteBeat, gammaLoaded, tokenIndex, ...
        requestBeatIndex, receiveBeatIndex, outputBeatIndex, currentSum, invRmsLatched] = iResetState();
    return;
end

if cfgGammaValid
    gammaMem = iWriteBeat(gammaMem, cfgGammaBeat, gammaWriteBeat, lanesPerBeat);

    if gammaWriteBeat == beatsPerToken - 1
        gammaWriteBeat = uint16(0);
        gammaLoaded = true;
    else
        gammaWriteBeat = gammaWriteBeat + uint16(1);
    end
end

epsilonScaled = cast(hiddenSize, 'like', epsilon) .* epsilon;

switch state
    case uint8(0)
        if start && gammaLoaded
            state = uint8(1);
            tokenIndex = uint8(0);
            requestBeatIndex = uint16(0);
            receiveBeatIndex = uint16(0);
            outputBeatIndex = uint16(0);
            currentSum = single(0);
            busy = true;
        end

    case uint8(1)
        busy = true;

        if requestBeatIndex < beatsPerToken
            ddrReadEn = true;
            ddrReadAddr = uint16(tokenIndex) * beatsPerToken + requestBeatIndex;
            requestBeatIndex = requestBeatIndex + uint16(1);
        end

        if ddrDataValid
            tokenMem = iWriteBeat(tokenMem, ddrDataBeat, receiveBeatIndex, lanesPerBeat);
            currentSum = currentSum + iReduceSquaredBeat(ddrDataBeat, lanesPerBeat);

            if receiveBeatIndex == beatsPerToken - 1
                receiveBeatIndex = uint16(0);
                requestBeatIndex = uint16(0);
                state = uint8(2);
            else
                receiveBeatIndex = receiveBeatIndex + uint16(1);
            end
        end

    case uint8(2)
        busy = true;
        invRmsLatched = iReciprocalSqrt(currentSum + epsilonScaled);
        currentSum = single(0);
        outputBeatIndex = uint16(0);
        state = uint8(3);

    otherwise
        busy = true;
        outValid = true;
        outBeat = iMultiplyBeat( ...
            iReadBeat(tokenMem, outputBeatIndex, lanesPerBeat), ...
            iReadBeat(gammaMem, outputBeatIndex, lanesPerBeat), ...
            invRmsLatched, lanesPerBeat);

        if outputBeatIndex == beatsPerToken - 1
            if tokenIndex == numTokens - uint8(1)
                done = true;
                state = uint8(0);
                tokenIndex = uint8(0);
                outputBeatIndex = uint16(0);
                busy = false;
            else
                tokenIndex = tokenIndex + uint8(1);
                outputBeatIndex = uint16(0);
                requestBeatIndex = uint16(0);
                receiveBeatIndex = uint16(0);
                state = uint8(1);
            end
        else
            outputBeatIndex = outputBeatIndex + uint16(1);
        end
end

end

function [state, gammaMem, tokenMem, gammaWriteBeat, gammaLoaded, tokenIndex, ...
    requestBeatIndex, receiveBeatIndex, outputBeatIndex, currentSum, invRmsLatched] = iResetState()
%#codegen

state = uint8(0);
gammaMem = zeros(1, 1536, 'single');
tokenMem = zeros(1, 1536, 'single');
gammaWriteBeat = uint16(0);
gammaLoaded = false;
tokenIndex = uint8(0);
requestBeatIndex = uint16(0);
receiveBeatIndex = uint16(0);
outputBeatIndex = uint16(0);
currentSum = single(0);
invRmsLatched = single(0);

end

function mem = iWriteBeat(mem, beat, beatIndex, lanesPerBeat)
%#codegen

baseIndex = double(beatIndex) * double(lanesPerBeat);

for laneIndex = 1:double(lanesPerBeat)
    mem(baseIndex + laneIndex) = beat(laneIndex);
end

end

function beat = iReadBeat(mem, beatIndex, lanesPerBeat)
%#codegen

beat = zeros(1, double(lanesPerBeat), 'single');
baseIndex = double(beatIndex) * double(lanesPerBeat);

for laneIndex = 1:double(lanesPerBeat)
    beat(laneIndex) = mem(baseIndex + laneIndex);
end

end

function sumSquares = iReduceSquaredBeat(beat, lanesPerBeat)
%#codegen

sumSquares = single(0);

for laneIndex = 1:double(lanesPerBeat)
    value = beat(laneIndex);
    sumSquares = sumSquares + value .* value;
end

end

function outBeat = iMultiplyBeat(xBeat, gBeat, invRms, lanesPerBeat)
%#codegen

outBeat = zeros(1, double(lanesPerBeat), 'single');
featureScale = single(sqrt(1536));

for laneIndex = 1:double(lanesPerBeat)
    outBeat(laneIndex) = xBeat(laneIndex) .* gBeat(laneIndex) .* invRms .* featureScale;
end

end

function invValue = iReciprocalSqrt(value)
%#codegen

invValue = single(1) ./ sqrt(value);

end