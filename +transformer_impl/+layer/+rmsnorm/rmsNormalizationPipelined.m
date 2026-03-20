function [outBeat, outValid, ddrReadAddr, ddrReadEn, done, busy] = rmsNormalizationPipelined( ...
    reset, start, cfgGammaBeat, cfgGammaValid, ddrDataBeat, ddrDataValid, epsilon)
% rmsNormalizationPipelined   Overlapped streaming RMSNorm prototype.
%#codegen

numTokens = coder.const(uint8(64));
hiddenSize = coder.const(uint16(1536));
lanesPerBeat = coder.const(uint8(8));
beatsPerToken = coder.const(uint16(192));

if nargin < 7
    epsilon = single(1e-6);
end

persistent phase gammaMem tokenMem gammaWriteBeat gammaLoaded
persistent outputBank fillBank outputTokenIndex prefetchTokenIndex
persistent requestBeatIndex receiveBeatIndex outputBeatIndex currentSum
persistent invRmsCurrent invRmsNext nextReady loadingActive

if isempty(phase)
    [phase, gammaMem, tokenMem, gammaWriteBeat, gammaLoaded, outputBank, fillBank, ...
        outputTokenIndex, prefetchTokenIndex, requestBeatIndex, receiveBeatIndex, ...
        outputBeatIndex, currentSum, invRmsCurrent, invRmsNext, nextReady, loadingActive] = iResetState();
end

outBeat = zeros(1, double(lanesPerBeat), 'single');
outValid = false;
ddrReadAddr = uint16(0);
ddrReadEn = false;
done = false;
busy = phase ~= uint8(0);

if reset
    [phase, gammaMem, tokenMem, gammaWriteBeat, gammaLoaded, outputBank, fillBank, ...
        outputTokenIndex, prefetchTokenIndex, requestBeatIndex, receiveBeatIndex, ...
        outputBeatIndex, currentSum, invRmsCurrent, invRmsNext, nextReady, loadingActive] = iResetState();
    return;
end

if cfgGammaValid
    gammaMem = iWriteBeat(gammaMem, cfgGammaBeat, uint8(1), gammaWriteBeat, lanesPerBeat);
    if gammaWriteBeat == beatsPerToken - 1
        gammaWriteBeat = uint16(0);
        gammaLoaded = true;
    else
        gammaWriteBeat = gammaWriteBeat + uint16(1);
    end
end

switch phase
    case uint8(0)
        if start && gammaLoaded
            phase = uint8(1);
            outputBank = uint8(1);
            fillBank = uint8(2);
            outputTokenIndex = uint8(0);
            prefetchTokenIndex = uint8(0);
            requestBeatIndex = uint16(0);
            receiveBeatIndex = uint16(0);
            outputBeatIndex = uint16(0);
            currentSum = single(0);
            nextReady = false;
            loadingActive = true;
            busy = true;
        end

    case uint8(1)
        busy = true;
        [ddrReadEn, ddrReadAddr, requestBeatIndex] = iIssueRead(ddrReadEn, ddrReadAddr, outputTokenIndex, requestBeatIndex, beatsPerToken);
        if ddrDataValid
            tokenMem = iWriteBeat(tokenMem, ddrDataBeat, outputBank, receiveBeatIndex, lanesPerBeat);
            [currentSum, receiveBeatIndex, doneReceiving] = iAccumulateBeat(currentSum, ddrDataBeat, receiveBeatIndex, lanesPerBeat, beatsPerToken);
            if doneReceiving
                invRmsCurrent = iReciprocalSqrt(currentSum + cast(hiddenSize, 'like', epsilon) .* epsilon);
                currentSum = single(0);
                outputBeatIndex = uint16(0);
                if numTokens > uint8(1)
                    prefetchTokenIndex = uint8(1);
                    requestBeatIndex = uint16(1);
                    ddrReadEn = true;
                    ddrReadAddr = beatsPerToken;
                    receiveBeatIndex = uint16(0);
                    loadingActive = true;
                    nextReady = false;
                    phase = uint8(2);
                else
                    loadingActive = false;
                    phase = uint8(3);
                end
            end
        end

    case uint8(2)
        busy = true;
        outValid = true;
        outBeat = iMultiplyBeat(iReadBeat(tokenMem, outputBank, outputBeatIndex, lanesPerBeat), ...
            iReadBeat(gammaMem, uint8(1), outputBeatIndex, lanesPerBeat), invRmsCurrent, lanesPerBeat);

        if loadingActive && requestBeatIndex < beatsPerToken
            ddrReadEn = true;
            ddrReadAddr = uint16(prefetchTokenIndex) * beatsPerToken + requestBeatIndex;
            requestBeatIndex = requestBeatIndex + uint16(1);
        end

        if ddrDataValid
            tokenMem = iWriteBeat(tokenMem, ddrDataBeat, fillBank, receiveBeatIndex, lanesPerBeat);
            [nextSum, nextReceiveBeatIndex, doneReceiving] = iAccumulateBeat(currentSum, ddrDataBeat, receiveBeatIndex, lanesPerBeat, beatsPerToken);
            if doneReceiving
                invRmsNext = iReciprocalSqrt(nextSum + cast(hiddenSize, 'like', epsilon) .* epsilon);
                currentSum = single(0);
                receiveBeatIndex = uint16(0);
                requestBeatIndex = uint16(0);
                nextReady = true;
                loadingActive = false;
            else
                currentSum = nextSum;
                receiveBeatIndex = nextReceiveBeatIndex;
            end
        end

        if outputBeatIndex == beatsPerToken - 1
            if nextReady
                oldOutputBank = outputBank;
                outputBank = fillBank;
                fillBank = oldOutputBank;
                outputTokenIndex = outputTokenIndex + uint8(1);
                outputBeatIndex = uint16(0);
                invRmsCurrent = invRmsNext;
                nextReady = false;

                if outputTokenIndex == numTokens - uint8(1)
                    loadingActive = false;
                    phase = uint8(3);
                else
                    prefetchTokenIndex = outputTokenIndex + uint8(1);
                    requestBeatIndex = uint16(1);
                    receiveBeatIndex = uint16(0);
                    currentSum = single(0);
                    loadingActive = true;
                    ddrReadEn = true;
                    ddrReadAddr = uint16(prefetchTokenIndex) * beatsPerToken;
                end
            else
                phase = uint8(4);
            end
        else
            outputBeatIndex = outputBeatIndex + uint16(1);
        end

    case uint8(3)
        busy = true;
        outValid = true;
        outBeat = iMultiplyBeat(iReadBeat(tokenMem, outputBank, outputBeatIndex, lanesPerBeat), ...
            iReadBeat(gammaMem, uint8(1), outputBeatIndex, lanesPerBeat), invRmsCurrent, lanesPerBeat);
        if outputBeatIndex == beatsPerToken - 1
            done = true;
            [phase, gammaMem, tokenMem, gammaWriteBeat, gammaLoaded, outputBank, fillBank, ...
                outputTokenIndex, prefetchTokenIndex, requestBeatIndex, receiveBeatIndex, ...
                outputBeatIndex, currentSum, invRmsCurrent, invRmsNext, nextReady, loadingActive] = iResetState();
            gammaLoaded = true;
        else
            outputBeatIndex = outputBeatIndex + uint16(1);
        end

    otherwise
        busy = true;
        if ddrDataValid
            tokenMem = iWriteBeat(tokenMem, ddrDataBeat, fillBank, receiveBeatIndex, lanesPerBeat);
            [nextSum, nextReceiveBeatIndex, doneReceiving] = iAccumulateBeat(currentSum, ddrDataBeat, receiveBeatIndex, lanesPerBeat, beatsPerToken);
            if doneReceiving
                invRmsNext = iReciprocalSqrt(nextSum + cast(hiddenSize, 'like', epsilon) .* epsilon);
                currentSum = single(0);
                receiveBeatIndex = uint16(0);
                requestBeatIndex = uint16(0);
                nextReady = true;
                loadingActive = false;
            else
                currentSum = nextSum;
                receiveBeatIndex = nextReceiveBeatIndex;
            end
        end

        if nextReady
            oldOutputBank = outputBank;
            outputBank = fillBank;
            fillBank = oldOutputBank;
            outputTokenIndex = outputTokenIndex + uint8(1);
            outputBeatIndex = uint16(0);
            invRmsCurrent = invRmsNext;
            nextReady = false;
            if outputTokenIndex == numTokens - uint8(1)
                loadingActive = false;
                phase = uint8(3);
            else
                prefetchTokenIndex = outputTokenIndex + uint8(1);
                requestBeatIndex = uint16(1);
                receiveBeatIndex = uint16(0);
                currentSum = single(0);
                loadingActive = true;
                ddrReadEn = true;
                ddrReadAddr = uint16(prefetchTokenIndex) * beatsPerToken;
                phase = uint8(2);
            end
        end
end

end

function [phase, gammaMem, tokenMem, gammaWriteBeat, gammaLoaded, outputBank, fillBank, ...
    outputTokenIndex, prefetchTokenIndex, requestBeatIndex, receiveBeatIndex, ...
    outputBeatIndex, currentSum, invRmsCurrent, invRmsNext, nextReady, loadingActive] = iResetState()
phase = uint8(0);
gammaMem = zeros(1, 1536, 'single');
tokenMem = zeros(2, 1536, 'single');
gammaWriteBeat = uint16(0);
gammaLoaded = false;
outputBank = uint8(1);
fillBank = uint8(2);
outputTokenIndex = uint8(0);
prefetchTokenIndex = uint8(0);
requestBeatIndex = uint16(0);
receiveBeatIndex = uint16(0);
outputBeatIndex = uint16(0);
currentSum = single(0);
invRmsCurrent = single(0);
invRmsNext = single(0);
nextReady = false;
loadingActive = false;
end

function mem = iWriteBeat(mem, beat, bankIndex, beatIndex, lanesPerBeat)
baseIndex = double(beatIndex) * double(lanesPerBeat);
for laneIndex = 1:double(lanesPerBeat)
    mem(double(bankIndex), baseIndex + laneIndex) = beat(laneIndex);
end
end

function beat = iReadBeat(mem, bankIndex, beatIndex, lanesPerBeat)
beat = zeros(1, double(lanesPerBeat), 'single');
baseIndex = double(beatIndex) * double(lanesPerBeat);
for laneIndex = 1:double(lanesPerBeat)
    beat(laneIndex) = mem(double(bankIndex), baseIndex + laneIndex);
end
end

function [ddrReadEn, ddrReadAddr, requestBeatIndex] = iIssueRead(ddrReadEn, ddrReadAddr, tokenIndex, requestBeatIndex, beatsPerToken)
if requestBeatIndex < beatsPerToken
    ddrReadEn = true;
    ddrReadAddr = uint16(tokenIndex) * beatsPerToken + requestBeatIndex;
    requestBeatIndex = requestBeatIndex + uint16(1);
end
end

function [nextSum, nextReceiveBeatIndex, doneReceiving] = iAccumulateBeat(currentSum, ddrDataBeat, receiveBeatIndex, lanesPerBeat, beatsPerToken)
nextSum = currentSum + iReduceSquaredBeat(ddrDataBeat, lanesPerBeat);
if receiveBeatIndex == beatsPerToken - 1
    nextReceiveBeatIndex = uint16(0);
    doneReceiving = true;
else
    nextReceiveBeatIndex = receiveBeatIndex + uint16(1);
    doneReceiving = false;
end
end

function sumSquares = iReduceSquaredBeat(beat, lanesPerBeat)
sumSquares = single(0);
for laneIndex = 1:double(lanesPerBeat)
    value = beat(laneIndex);
    sumSquares = sumSquares + value .* value;
end
end

function outBeat = iMultiplyBeat(xBeat, gBeat, invRms, lanesPerBeat)
outBeat = zeros(1, double(lanesPerBeat), 'single');
featureScale = single(sqrt(1536));
for laneIndex = 1:double(lanesPerBeat)
    outBeat(laneIndex) = xBeat(laneIndex) .* gBeat(laneIndex) .* invRms .* featureScale;
end
end

function invValue = iReciprocalSqrt(value)
invValue = single(1) ./ sqrt(value);
end