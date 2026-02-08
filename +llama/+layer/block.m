function [h, present] = block(h, past, weights, hyperParameters, freqs_cis)
% block   Transformer block for Llama
%
%   [h, present] = block(h, past, weights, hyperParameters, freqs_cis)
%
%   Inputs:
%       h               - Input tensor [Hidden, Seq, Batch]
%       past            - Struct with .keys and .values
%       weights         - Struct of weights for this layer
%       hyperParameters - NumHeads, NumKVHeads, HeadDim
%       freqs_cis       - Precomputed RoPE frequencies
%
%   Outputs:
%       h               - Output tensor
%       present         - Updated KV cache struct

    import transformer.layer.*
    
    resid = h;
    
    % 1. Input Norm
    % Weight name convention: input_layernorm
    h = rmsNormalization(h, weights.input_layernorm, 1e-5);
    
    % 2. Attention
    attnWeights.q_proj = weights.self_attn_q_proj;
    attnWeights.k_proj = weights.self_attn_k_proj;
    attnWeights.v_proj = weights.self_attn_v_proj;
    attnWeights.o_proj = weights.self_attn_o_proj;
    
    [h_attn, present] = attentionGQA(h, past, attnWeights, freqs_cis, hyperParameters);
    
    h = resid + h_attn;
    
    % 3. Post Attention Norm
    resid = h;
    h = rmsNormalization(h, weights.post_attention_layernorm, 1e-5);
    
    % 4. MLP
    ffnWeights.gate_proj = weights.mlp_gate_proj;
    ffnWeights.up_proj   = weights.mlp_up_proj;
    ffnWeights.down_proj = weights.mlp_down_proj;
    
    h_ffn = gatedMLP(h, ffnWeights);
    
    h = resid + h_ffn;

end
