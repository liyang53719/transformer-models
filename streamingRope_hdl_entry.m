function [outBeat, outValid, busy, done] = streamingRope_hdl_entry(start, cfgNumTokens, cfgNumHeads, inBeat, inValid)
% streamingRope_hdl_entry   HDL entry wrapper for the streaming RoPE core.
%
%#codegen

[outBeat, outValid, busy, done] = transformer_impl.layer.rope.streamingRope( ...
    start, cfgNumTokens, cfgNumHeads, inBeat, inValid);

end