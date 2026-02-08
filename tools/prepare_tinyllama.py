import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import scipy.io
import numpy as np
import os

def export_tinyllama():
    # Check if local model directory exists (from llama.download)
    local_model_dir = "tinyllama_model"
    if os.path.exists(local_model_dir) and os.path.isdir(local_model_dir):
        print(f"Found local model folder '{local_model_dir}'. Loading from there...")
        model_id = local_model_dir
    else:
        print("Local model folder not found. Loading TinyLlama-1.1B-Chat from HuggingFace Hub...")
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
        # local_files_only=True is redundant if path is a directory, but good for safety if it falls back to repo ID
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, local_files_only=True, attn_implementation="eager")
    except Exception as e:
        print(f"Local load failed: {e}")
        if model_id == local_model_dir:
            raise RuntimeError("Could not load from local directory. Please check if download was successful.")
        
        print("Trying online...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, attn_implementation="eager")

    print("Model loaded. Converting weights...")
    
    mat_params = {}
    config = model.config
    mat_params['NumLayers'] = config.num_hidden_layers
    mat_params['HiddenSize'] = config.hidden_size
    mat_params['NumHeads'] = config.num_attention_heads
    mat_params['NumKVHeads'] = config.num_key_value_heads
    mat_params['HeadDim'] = config.hidden_size // config.num_attention_heads
    mat_params['VocabSize'] = config.vocab_size
    
    if hasattr(config, 'rope_theta'):
        mat_params['RopeTheta'] = config.rope_theta
    else:
        mat_params['RopeTheta'] = 10000.0

    print(f"RoPE Theta: {mat_params['RopeTheta']}")

    # Embeddings [Hidden, Vocab]
    mat_params['embed_tokens'] = model.model.embed_tokens.weight.detach().cpu().numpy().T
    
    # Layers
    for i, layer in enumerate(model.model.layers):
        prefix = f"layer_{i}_"
        mat_params[f"{prefix}self_attn_q_proj"] = layer.self_attn.q_proj.weight.detach().cpu().numpy()
        mat_params[f"{prefix}self_attn_k_proj"] = layer.self_attn.k_proj.weight.detach().cpu().numpy()
        mat_params[f"{prefix}self_attn_v_proj"] = layer.self_attn.v_proj.weight.detach().cpu().numpy()
        mat_params[f"{prefix}self_attn_o_proj"] = layer.self_attn.o_proj.weight.detach().cpu().numpy()
        
        mat_params[f"{prefix}mlp_gate_proj"] = layer.mlp.gate_proj.weight.detach().cpu().numpy()
        mat_params[f"{prefix}mlp_up_proj"]   = layer.mlp.up_proj.weight.detach().cpu().numpy()
        mat_params[f"{prefix}mlp_down_proj"] = layer.mlp.down_proj.weight.detach().cpu().numpy()
        
        mat_params[f"{prefix}input_layernorm"] = layer.input_layernorm.weight.detach().cpu().numpy()
        mat_params[f"{prefix}post_attention_layernorm"] = layer.post_attention_layernorm.weight.detach().cpu().numpy()
        
    mat_params['norm'] = model.model.norm.weight.detach().cpu().numpy()
    mat_params['lm_head'] = model.lm_head.weight.detach().cpu().numpy()
    
    print("Saving params to tinyllama_params.mat...")
    scipy.io.savemat('tinyllama_params.mat', mat_params)
    
    # Generate Test Data
    print("Generating Test Data...")
    text = "The quick brown fox jumps over the lazy dog."
    model_inputs = tokenizer(text, return_tensors="pt")
    input_ids = model_inputs.input_ids.detach().cpu().numpy()
    
    with torch.no_grad():
        outputs = model(**model_inputs, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions # Tuple of (Batch, NumHeads, Seq, Seq)
        
    ref_logits = logits.detach().cpu().numpy()
    # Transpose [Vocab, Seq, Batch]
    ref_logits = np.transpose(ref_logits, (2, 1, 0))
    
    # Extract reference attention weights for Layer 0
    # Shape: (Batch, NumHeads, Seq, Seq)
    # Target: (NumHeads, Seq, Seq) for batch 0
    ref_attn_weights = attentions[0][0].detach().cpu().numpy()

    # Layer 0 Debug
    l0 = model.model.layers[0]
    
    # Define position_ids early
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=model.device).unsqueeze(0)
    
    # 1. Input Norm
    hidden_states = model.model.embed_tokens(model_inputs.input_ids)
    ref_embed = hidden_states.detach().cpu().numpy().transpose(2, 1, 0)
    
    residual = hidden_states
    hidden_states = l0.input_layernorm(hidden_states)
    ref_l0_ln1 = hidden_states.detach().cpu().numpy().transpose(2, 1, 0)
    
    # 2. Attention Components (Manual Extraction)
    bsz, q_len, _ = hidden_states.size()
    query_states = l0.self_attn.q_proj(hidden_states)
    key_states = l0.self_attn.k_proj(hidden_states)
    value_states = l0.self_attn.v_proj(hidden_states)
    
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # RoPE
    cos, sin = model.model.rotary_emb(value_states, position_ids)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    q_rot, k_rot = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    ref_q = q_rot.detach().cpu().numpy() # [Batch, NumHeads, Seq, HeadDim]
    ref_k = k_rot.detach().cpu().numpy()
    ref_v = value_states.detach().cpu().numpy()
    
    ref_rotary_cos = cos.detach().cpu().numpy().transpose(2, 1, 0)
    ref_rotary_sin = sin.detach().cpu().numpy().transpose(2, 1, 0)

    # Continue with block execution for other refs
    attn_outputs = l0.self_attn(
        hidden_states,
        attention_mask=None,
        position_ids=position_ids, 
        past_key_value=None,
        position_embeddings=(cos, sin)
    )
    hidden_states = attn_outputs[0]
    
    ref_l0_attn = hidden_states.detach().cpu().numpy().transpose(2, 1, 0)
    hidden_states = residual + hidden_states
    
    # 3. Post Norm
    residual = hidden_states
    hidden_states = l0.post_attention_layernorm(hidden_states)
    ref_l0_ln2 = hidden_states.detach().cpu().numpy().transpose(2, 1, 0)
    
    # 4. MLP
    hidden_states = l0.mlp(hidden_states)
    ref_l0_mlp = hidden_states.detach().cpu().numpy().transpose(2, 1, 0)
    
    hidden_states = residual + hidden_states
    ref_l0_out = hidden_states.detach().cpu().numpy().transpose(2, 1, 0)

    # Create generated data for sequence comparison
    print("Generating Sequence for comparison...")
    # Greedy generation for deterministic comparison
    gen_tokens = model.generate(**model_inputs, max_new_tokens=20, do_sample=False)
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    
    ref_generated_ids = gen_tokens.detach().cpu().numpy() # [Batch, Seq]
    print(f"Generated text: {gen_text}")

    test_data = {
        'input_ids': input_ids,
        'ref_logits': ref_logits,
        'ref_embed': ref_embed,
        'ref_generated_ids': ref_generated_ids, # NEW
        'ref_generated_text': gen_text,         # NEW
        'ref_layer_0': ref_l0_out, 
        'ref_l0_ln1': ref_l0_ln1,
        'ref_l0_attn': ref_l0_attn,
        'ref_l0_ln2': ref_l0_ln2,
        'ref_l0_mlp': ref_l0_mlp,
        'ref_rotary_cos': ref_rotary_cos,
        'ref_rotary_sin': ref_rotary_sin,
        'ref_attn_weights': ref_attn_weights,
        'ref_q': ref_q,
        'ref_k': ref_k,
        'ref_v': ref_v
    }
    
    print("Saving data to tinyllama_data.mat...")
    scipy.io.savemat('tinyllama_data.mat', test_data)
    print("Done.")

if __name__ == "__main__":
    export_tinyllama()
