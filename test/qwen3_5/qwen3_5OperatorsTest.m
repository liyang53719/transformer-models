classdef qwen3_5OperatorsTest < matlab.unittest.TestCase
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
