function [outBeat, outValid, busy, done] = streamingRopeSimulink(start, cfgNumTokens, cfgNumHeads, inBeat, inValid)
% streamingRopeSimulink   HDL-friendly RoPE core with explicit functional state only.

%#codegen

persistent firstHalfBuf firstHalfRowValid
persistent secondHalfBuf secondHalfRowReady
persistent pendingFirstHalfBeat pendingFirstHalfValid
persistent currentCos currentSin deltaCos deltaSin
persistent runActive ingestDone drainIndex
persistent activeTokenCount activeHeadCount activeBeatsPerToken
persistent activeTokenIndex beatInToken remainingInputBeats remainingOutputBeats constantsReady

if isempty(constantsReady)
    firstHalfBuf = zeros(8, 8, 'single');
    firstHalfRowValid = false(1, 8);
    secondHalfBuf = zeros(8, 8, 'single');
    secondHalfRowReady = false(1, 8);
    pendingFirstHalfBeat = zeros(1, 8, 'single');
    pendingFirstHalfValid = false;
    currentCos = ones(8, 8, 'single');
    currentSin = zeros(8, 8, 'single');
    deltaCos = [ ...
        single(0.54030230586813977) single(0.69250391454294158) single(0.79645787448595906) single(0.86617519535750209) single(0.91239585964625614) single(0.9428144003135146) single(0.96273901351292701) single(0.97575027073245013); ...
        single(0.98423023447009461) single(0.98974993353248431) single(0.99333979904393477) single(0.99567330204062576) single(0.99718961074278156) single(0.99817468493002592) single(0.99881454743753428) single(0.99923013553913631); ...
        single(0.99950004166527784) single(0.9996753267545192) single(0.99978915915767907) single(0.99986308214281749) single(0.99991108734710599) single(0.99994226145639731) single(0.99996250552384169) single(0.99997565172254876); ...
        single(0.9999841886533658) single(0.99998973239243838) single(0.99999333240024868) single(0.99999567018150781) single(0.99999718829469164) single(0.99999817412991931) single(0.99999881431338145) single(0.99999923003683577); ...
        single(0.99999950000004167) single(0.99999967530920175) single(0.99999978915175569) single(0.99999986307902144) single(0.99999991108603081) single(0.99999994226090128) single(0.99999996250528977) single(0.9999999756516238); ...
        single(0.99999998418861169) single(0.99999998973237492) single(0.99999999333239287) single(0.99999999567017839) single(0.99999999718829335) single(0.99999999817412932) single(0.99999999881431312) single(0.99999999923003668); ...
        single(0.99999999949999996) single(0.99999999967530917) single(0.99999999978915177) single(0.99999999986307897) single(0.99999999991108601) single(0.99999999994226085) single(0.99999999996250533) single(0.99999999997565159); ...
        single(0.99999999998418865) single(0.99999999998973232) single(0.99999999999333244) single(0.99999999999567013) single(0.99999999999718825) single(0.99999999999817413) single(0.99999999999881428) single(0.99999999999923006) ...
    ];
    deltaSin = [ ...
        single(0.8414709848078965) single(0.72141411709412939) single(0.60469401697826342) single(0.49974046358824409) single(0.40930892404193842) single(0.33331817616425879) single(0.27043223154823115) single(0.21888674963448632); ...
        single(0.17689218624615002) single(0.14281130582850568) single(0.11522171512069776) single(0.092922955202236493) single(0.074919157941476022) single(0.060392866837429587) single(0.048677508432062339) single(0.03923182675378966); ...
        single(0.031617506402433708) single(0.025480209540842826) single(0.020533807021130068) single(0.016547415745833297) single(0.01333481909619642) single(0.010745871461447223) single(0.0086595350037345295) single(0.0069782492119310818); ...
        single(0.0056233836139601857) single(0.0045315681280827273) single(0.0036517331564283693) single(0.002942722929049874) single(0.0023713714831265597) single(0.0019109518119196781) single(0.0015399259174360186) single(0.0012409374422595815); ...
        single(0.00099999983333334168) single(0.00080584210054496566) single(0.00064938158593588497) single(0.00052329909079795224) single(0.00042169649093034558) single(0.00033982082635393966) single(0.00027384196000389445) single(0.00022067340511744592); ...
        single(0.00017782794006665674) single(0.00014330125653324175) single(0.0001154781982122914) single(9.3057203958662866e-05) single(7.498942086296283e-05) single(6.0429638987034373e-05) single(4.8696752497339948e-05) single(3.9241897574773755e-05); ...
        single(3.1622776596413333e-05) single(2.548296747703544e-05) single(2.0535250263128189e-05) single(1.6548170998676548e-05) single(1.3335214321238012e-05) single(1.0746078283006351e-05) single(8.6596432334924231e-06) single(6.9783058485420272e-06); ...
        single(5.6234132518738535e-06) single(4.5315836375853089e-06) single(3.6517412725402608e-06) single(2.942727176205035e-06) single(2.3713737056594326e-06) single(1.9109529749692775e-06) single(1.5399265260588833e-06) single(1.240937760751401e-06) ...
    ];
    runActive = false;
    ingestDone = false;
    drainIndex = uint8(0);
    activeTokenCount = uint16(0);
    activeHeadCount = uint8(0);
    activeBeatsPerToken = uint16(0);
    activeTokenIndex = uint16(0);
    beatInToken = uint16(0);
    remainingInputBeats = uint32(0);
    remainingOutputBeats = uint32(0);
    constantsReady = true;
end

outBeat = zeros(1, 8, 'single');
outValid = false;
done = false;

if start
    activeTokenCount = iClipTokenCount(cfgNumTokens);
    activeHeadCount = iClipHeadCount(cfgNumHeads);
    activeBeatsPerToken = uint16(activeHeadCount) * uint16(16);
    remainingInputBeats = uint32(activeTokenCount) * uint32(activeBeatsPerToken);
    remainingOutputBeats = remainingInputBeats;

    firstHalfBuf(:) = single(0);
    firstHalfRowValid(:) = false;
    secondHalfBuf(:) = single(0);
    secondHalfRowReady(:) = false;
    pendingFirstHalfBeat(:) = single(0);
    pendingFirstHalfValid = false;
    currentCos(:) = single(1);
    currentSin(:) = single(0);
    runActive = remainingInputBeats ~= 0;
    ingestDone = false;
    drainIndex = uint8(0);
    activeTokenIndex = uint16(0);
    beatInToken = uint16(0);
end

if runActive
    if pendingFirstHalfValid
        outBeat = pendingFirstHalfBeat;
        outValid = true;
        pendingFirstHalfValid = false;
    else
        drainRow = double(drainIndex) + 1;
        if secondHalfRowReady(drainRow)
            outBeat = secondHalfBuf(drainRow, :);
            outValid = true;
            secondHalfRowReady(drainRow) = false;
            if drainIndex == uint8(7)
                drainIndex = uint8(0);
            else
                drainIndex = drainIndex + uint8(1);
            end
        end
    end

    if inValid && ~ingestDone
        beatInHead = iBeatInHead(beatInToken);

        if beatInHead < 8
            rowIndex = double(beatInHead) + 1;

            if activeTokenIndex > 0 && beatInToken < 8
                [currentCos(rowIndex, :), currentSin(rowIndex, :)] = iAdvancePhaseRow( ...
                    currentCos(rowIndex, :), currentSin(rowIndex, :), deltaCos(rowIndex, :), deltaSin(rowIndex, :));
            end

            firstHalfBuf(rowIndex, :) = inBeat;
            firstHalfRowValid(rowIndex) = true;
        else
            rowIndex = double(beatInHead - uint8(7));
            if firstHalfRowValid(rowIndex)
                [firstHalfOut, secondHalfOut] = iRotateBeat( ...
                    firstHalfBuf(rowIndex, :), inBeat, currentCos(rowIndex, :), currentSin(rowIndex, :));
                pendingFirstHalfBeat = firstHalfOut;
                pendingFirstHalfValid = true;
                secondHalfBuf(rowIndex, :) = secondHalfOut;
                secondHalfRowReady(rowIndex) = true;
            end

            if rowIndex == 8
                firstHalfRowValid(:) = false;
            end
        end

        remainingInputBeats = remainingInputBeats - uint32(1);
        if remainingInputBeats == 0
            ingestDone = true;
        end

        if beatInToken + uint16(1) == activeBeatsPerToken
            beatInToken = uint16(0);
            if activeTokenIndex + uint16(1) < activeTokenCount
                activeTokenIndex = activeTokenIndex + uint16(1);
            end
        else
            beatInToken = beatInToken + uint16(1);
        end
    end

    if outValid
        if remainingOutputBeats == uint32(1)
            runActive = false;
            ingestDone = false;
            firstHalfRowValid(:) = false;
            secondHalfRowReady(:) = false;
            pendingFirstHalfValid = false;
            done = true;
        else
            remainingOutputBeats = remainingOutputBeats - uint32(1);
        end
    end
end

busy = runActive;

end

function tokenCount = iClipTokenCount(cfgNumTokens)
tokenCount = uint16(cfgNumTokens);
if tokenCount < uint16(64)
    tokenCount = uint16(64);
elseif tokenCount > uint16(1024)
    tokenCount = uint16(1024);
end
end

function headCount = iClipHeadCount(cfgNumHeads)
headCount = uint8(cfgNumHeads);
if ~(headCount == uint8(2) || headCount == uint8(12))
    if headCount < uint8(7)
        headCount = uint8(2);
    else
        headCount = uint8(12);
    end
end
end

function [nextCos, nextSin] = iAdvancePhaseRow(curCos, curSin, deltaCos, deltaSin)
nextCos = zeros(1, 8, 'single');
nextSin = zeros(1, 8, 'single');
for laneIndex = 1:8
    nextCos(laneIndex) = curCos(laneIndex) .* deltaCos(laneIndex) - curSin(laneIndex) .* deltaSin(laneIndex);
    nextSin(laneIndex) = curSin(laneIndex) .* deltaCos(laneIndex) + curCos(laneIndex) .* deltaSin(laneIndex);
end
end

function [firstHalfOut, secondHalfOut] = iRotateBeat(firstHalfIn, secondHalfIn, cosBeat, sinBeat)
firstHalfOut = zeros(1, 8, 'single');
secondHalfOut = zeros(1, 8, 'single');
for laneIndex = 1:8
    firstHalfOut(laneIndex) = firstHalfIn(laneIndex) .* cosBeat(laneIndex) - secondHalfIn(laneIndex) .* sinBeat(laneIndex);
    secondHalfOut(laneIndex) = firstHalfIn(laneIndex) .* sinBeat(laneIndex) + secondHalfIn(laneIndex) .* cosBeat(laneIndex);
end
end

function beatInHead = iBeatInHead(beatInToken)
beatInHead = uint8(bitand(beatInToken, uint16(15)));
end
