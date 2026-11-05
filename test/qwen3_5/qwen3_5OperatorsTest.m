classdef qwen3_5OperatorsTest < matlab.unittest.TestCase
    properties (TestParameter)
        GdnLength = struct('one', 1, 'partial', 17, 'boundary', 64, ...
            'tail', 65, 'multiple', 129);
    end
    methods (TestClassSetup)
        function addSourcePath(test)
            repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
            test.applyFixture(matlab.unittest.fixtures.PathFixture(repoRoot));
        end
    end
    methods (TestMethodSetup)
        function restoreRandomState(test)
            previous = rng(7);
            test.addTeardown(@() rng(previous));
        end
    end
    methods (Test)
        function linearAttentionChunkwiseMatchesRecurrent(test)
            [x, w, hp] = iSmallLinearAttention();
            hp.GdnImplementation = 'recurrent';
            [expected, expectedState] = qwen3_5.layer.linearAttention(x, [], w, hp);
            hp.GdnImplementation = 'chunkwise';
            [actual, actualState] = qwen3_5.layer.linearAttention(x, [], w, hp);
            test.verifyEqual(actual, expected, 'AbsTol', single(2e-5), 'RelTol', single(2e-5));
            test.verifyEqual(actualState.SSM, expectedState.SSM, 'AbsTol', single(2e-5), 'RelTol', single(2e-5));
            test.verifyEqual(actualState.History, expectedState.History, 'AbsTol', single(0));
        end
        function chunkwiseDeltaMatchesRecurrent(test, GdnLength)
            [q, k, v, g, beta, state] = iGdnInputs(GdnLength);
            [expected, expectedState] = qwen3_5.layer.gatedDeltaNet(q, k, v, g, beta, state);
            [actual, actualState] = qwen3_5.layer.gatedDeltaNetChunkwise(q, k, v, g, beta, state);
            test.verifyEqual(actual, expected, 'AbsTol', single(2e-5), 'RelTol', single(2e-5));
            test.verifyEqual(actualState, expectedState, 'AbsTol', single(2e-5), 'RelTol', single(2e-5));
            test.verifyClass(actual, 'single');
            test.verifyClass(actualState, 'single');
        end
        function chunkwiseDeltaPreservesStateAcrossCalls(test)
            [q, k, v, g, beta, state] = iGdnInputs(129);
            [expected, expectedState] = qwen3_5.layer.gatedDeltaNetChunkwise(q, k, v, g, beta, state);
            [left, next] = qwen3_5.layer.gatedDeltaNetChunkwise( ...
                q(:,:,1:17), k(:,:,1:17), v(:,:,1:17), g(:,1:17), beta(:,1:17), state);
            [right, final] = qwen3_5.layer.gatedDeltaNetChunkwise( ...
                q(:,:,18:end), k(:,:,18:end), v(:,:,18:end), g(:,18:end), beta(:,18:end), next);
            test.verifyEqual(cat(3, left, right), expected, 'AbsTol', single(2e-5), 'RelTol', single(2e-5));
            test.verifyEqual(final, expectedState, 'AbsTol', single(2e-5), 'RelTol', single(2e-5));
        end
        function chunkwiseDeltaHandlesStrongDecay(test)
            [q, k, v, ~, beta, state] = iGdnInputs(65);
            g = -1000 * ones(2, 65, 'single');
            [expected, expectedState] = qwen3_5.layer.gatedDeltaNet(q, k, v, g, beta, state);
            [actual, actualState] = qwen3_5.layer.gatedDeltaNetChunkwise(q, k, v, g, beta, state);
            test.verifyEqual(actual, expected, 'AbsTol', single(2e-5));
            test.verifyEqual(actualState, expectedState, 'AbsTol', single(2e-5));
        end
        function chunkwiseDeltaZeroBetaKeepsState(test)
            [q, k, v, ~, ~, state] = iGdnInputs(65);
            [~, actualState] = qwen3_5.layer.gatedDeltaNetChunkwise( ...
                q, k, v, zeros(2, 65, 'single'), zeros(2, 65, 'single'), state);
            test.verifyEqual(actualState, state, 'AbsTol', single(0));
        end
        function stopTokensIncludeChatTerminator(test)
            folder = tempname;
            mkdir(folder);
            test.addTeardown(@() rmdir(folder, 's'));
            iWriteStopTokenFixture(folder, true);
            [ids, primary] = qwen3_5.internal.stopTokenIds(folder);
            test.verifyEqual(ids, [1 3 5]);
            test.verifyEqual(primary, 3);
        end
        function stopTokensFallBackToModelConfig(test)
            folder = tempname;
            mkdir(folder);
            test.addTeardown(@() rmdir(folder, 's'));
            iWriteStopTokenFixture(folder, false);
            [ids, primary] = qwen3_5.internal.stopTokenIds(folder);
            test.verifyEqual(ids, 1);
            test.verifyEqual(primary, 1);
        end
        function convolutionPreservesChunkHistory(test)
            x = single(randn(6, 11));
            weight = single(randn(6, 4));
            history = single(randn(6, 3));
            [whole, wholeHistory] = qwen3_5.layer.causalDepthwiseConv(x, weight, history);
            [left, next] = qwen3_5.layer.causalDepthwiseConv(x(:, 1:3), weight, history);
            [right, final] = qwen3_5.layer.causalDepthwiseConv(x(:, 4:end), weight, next);
            test.verifyEqual([left right], whole, 'AbsTol', single(1e-6));
            test.verifyEqual(final, wholeHistory, 'AbsTol', single(1e-6));
        end
        function deltaRulePreservesNonzeroStateAcrossChunks(test)
            q = qwen3_5.layer.l2Normalization(single(randn(8, 2, 7)), single(1e-6));
            k = qwen3_5.layer.l2Normalization(single(randn(8, 2, 7)), single(1e-6));
            v = single(randn(8, 2, 7));
            g = -single(rand(2, 7));
            beta = single(rand(2, 7));
            state = single(randn(8, 8, 2));
            [whole, wholeState] = qwen3_5.layer.gatedDeltaNet(q, k, v, g, beta, state);
            [left, next] = qwen3_5.layer.gatedDeltaNet( ...
                q(:, :, 1:3), k(:, :, 1:3), v(:, :, 1:3), g(:, 1:3), beta(:, 1:3), state);
            [right, final] = qwen3_5.layer.gatedDeltaNet( ...
                q(:, :, 4:end), k(:, :, 4:end), v(:, :, 4:end), g(:, 4:end), beta(:, 4:end), next);
            test.verifyEqual(cat(3, left, right), whole, 'AbsTol', single(1e-6));
            test.verifyEqual(final, wholeState, 'AbsTol', single(1e-6));
        end
        function rotaryEmbeddingPreservesUnrotatedChannels(test)
            x = single(randn(16, 3, 7));
            actual = qwen3_5.layer.rotaryEmbedding(x, 5:11, 8, single(1e7));
            test.verifyEqual(actual(9:end, :, :), x(9:end, :, :), 'AbsTol', single(0));
            test.verifyEqual(sum(actual(1:8, :, :).^2, 1), ...
                sum(x(1:8, :, :).^2, 1), 'AbsTol', single(4e-6));
        end
        function attentionCacheMatchesCausalPrefill(test)
            [x, w, hp] = iSmallAttention();
            whole = qwen3_5.layer.fullAttention(x, [], w, hp, 0);
            [left, cache] = qwen3_5.layer.fullAttention(x(:, 1:3), [], w, hp, 0);
            right = qwen3_5.layer.fullAttention(x(:, 4:end), cache, w, hp, 3);
            test.verifyEqual([left right], whole, 'AbsTol', single(2e-6));
        end
        function attentionDoesNotReadFutureTokens(test)
            [x, w, hp] = iSmallAttention();
            changed = [x(:, 1:3) single(randn(6, 4))];
            a = qwen3_5.layer.fullAttention(x, [], w, hp, 0);
            b = qwen3_5.layer.fullAttention(changed, [], w, hp, 0);
            test.verifyEqual(a(:, 1:3), b(:, 1:3), 'AbsTol', single(1e-6));
        end
    end
end

function [q, k, v, g, beta, state] = iGdnInputs(tokens)
q = qwen3_5.layer.l2Normalization(single(randn(8, 2, tokens)), single(1e-6));
k = qwen3_5.layer.l2Normalization(single(randn(8, 2, tokens)), single(1e-6));
v = single(randn(12, 2, tokens));
g = -single(rand(2, tokens)) / 4;
beta = single(rand(2, tokens));
state = single(randn(8, 12, 2));
end

function [x, w, hp] = iSmallLinearAttention()
x = single(randn(6, 65));
hp = struct('LinearKeyDim', 8, 'LinearValueDim', 12, ...
    'LinearKeyHeads', 2, 'LinearValueHeads', 2, ...
    'RmsNormEpsilon', single(1e-6), 'GdnChunkSize', 32);
w = struct('linear_attn_in_proj_qkv', single(randn(56, 6))/4, ...
    'linear_attn_in_proj_z', single(randn(24, 6))/4, ...
    'linear_attn_in_proj_a', single(randn(2, 6))/4, ...
    'linear_attn_in_proj_b', single(randn(2, 6))/4, ...
    'linear_attn_dt_bias', single(randn(2, 1)), ...
    'linear_attn_A', -ones(2, 1, 'single'), ...
    'linear_attn_conv1d', single(randn(56, 4))/4, ...
    'linear_attn_norm', ones(12, 1, 'single'), ...
    'linear_attn_out_proj', single(randn(6, 24))/4);
end

function iWriteStopTokenFixture(folder, includeTokenizer)
cfg = struct('text_config', struct('eos_token_id', 1, 'vocab_size', 8));
iWriteFixture(fullfile(folder, 'config.json'), cfg);
if includeTokenizer
    meta = struct('eos_token', '<|im_end|>', ...
        'added_tokens_decoder', struct('x3', struct('content', '<|im_end|>'), ...
        'x5', struct('content', '<|fim_suffix|>')));
    iWriteFixture(fullfile(folder, 'tokenizer_config.json'), meta);
end
end

function iWriteFixture(filePath, value)
fid = fopen(filePath, 'w');
closer = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(value));
end

function [x, w, hp] = iSmallAttention()
x = single(randn(6, 7));
hp = struct('HeadDim', 8, 'NumHeads', 4, 'NumKVHeads', 2, ...
    'RotaryDim', 4, 'RopeTheta', single(1e7), 'RmsNormEpsilon', single(1e-6));
w = struct('self_attn_q_proj', single(randn(64, 6)) / 4, ...
    'self_attn_k_proj', single(randn(16, 6)) / 4, ...
    'self_attn_v_proj', single(randn(16, 6)) / 4, ...
    'self_attn_o_proj', single(randn(6, 32)) / 4, ...
    'self_attn_q_norm', ones(8, 1, 'single'), ...
    'self_attn_k_norm', ones(8, 1, 'single'));
end
