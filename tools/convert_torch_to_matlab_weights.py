import scipy.io
import torch
import numpy as np
import os
import argparse

def convert_pytorch_to_matlab(pytorch_model_path, output_mat_path):
    """
    Converts a PyTorch GPT-2 model checkpoint to a MATLAB .mat file compatible with
    the Transformer Models for MATLAB repository.
    
    This script handles the necessary transposition of weights from PyTorch's 
    row-major format (N, C) to MATLAB's column-major friendly format (C, N).
    """
    print(f"Loading PyTorch model from: {pytorch_model_path}")
    try:
        # Load the state dict directly
        state_dict = torch.load(pytorch_model_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return

    matlab_params = {}

    print("Converting weights...")
    
    for key, value in state_dict.items():
        # Convert tensor to numpy
        weight = value.numpy()
        
        # MATLAB variable names cannot contain dots
        # e.g., 'wte.weight' -> 'model_wte_0' (following the repo's convention roughly)
        # However, to be generic, we might just replace dots with underscores.
        # But let's try to match the specific naming convention seen in the .mat file 
        # based on the keys we saw earlier:
        # 'wte.weight' -> 'model_wte_0' (Wait, likely 'model_wte_0' is specific)
        
        # Let's map standard GPT-2 keys to the format seen in the downloaded mat file.
        # Based on previous inspection:
        # Pytorch: wte.weight -> MATLAB: model_wte_0 (Shape transposed)
        # Pytorch: wpe.weight -> MATLAB: model_wpe_0 (Shape transposed)
        # Pytorch: h.0.attn.c_attn.weight -> MATLAB: model_h0_attn_c_attn_w_0 (Transposed)
        # Pytorch: h.0.attn.c_attn.bias   -> MATLAB: model_h0_attn_c_attn_b_0
        
        # Heuristic for naming:
        # 1. Prefix with 'model_'
        # 2. Replace dots with underscores
        # 3. If it's a weight, append '_w_0' or just '_0' depending on context?
        # Let's look at the keys from the previous read_file output of compare_weights.py output.
        # 'model_wte_0', 'model_wpe_0'
        # 'model_h0_attn_c_attn_w_0', 'model_h0_attn_c_attn_b_0'
        # 'model_ln_f_g_0' (gamma/weight), 'model_ln_f_b_0' (beta/bias)
        
        new_key = "model_" + key.replace(".", "_")
        
        # Fix endings to match the MathWorks convention observed
        if new_key.endswith("_weight"):
            new_key = new_key[:-7] # remove '_weight'
            if "wte" in new_key or "wpe" in new_key:
                new_key += "_0"
            else:
                new_key += "_w_0"
        elif new_key.endswith("_bias"):
            new_key = new_key[:-5] + "_b_0"
            
        # Special case for LayerNorm which uses weight/bias in PT but often gamma/beta in others
        # In the .mat file we saw 'model_ln_f_g_0' and 'model_ln_f_b_0'
        # PT: ln_f.weight -> model_ln_f_weight -> model_ln_f_w_0 (Our rule above)
        # We need to change '_w_0' to '_g_0' for LayerNorms if that's the convention.
        # Checking previous output: 'model_h0_ln_1_g_0', 'model_ln_f_g_0'.
        # So yes, LN weights are 'g' (gamma).
        if "ln_" in new_key and new_key.endswith("_w_0"):
             new_key = new_key.replace("_w_0", "_g_0")

        # Transpose Logic
        # 1D tensors (biases) don't need transpose usually, but MATLAB treats them as column vectors often?
        # Let's check bias shapes. 
        # In PT: bias is (N,). 
        # In MATLAB: usually (N, 1).
        
        # 2D weights:
        # PT: (Out, In) or (N, C)
        # MATLAB: (In, Out) or (C, N) -> Needs Transpose.
        
        # Conv1D weights in OpenAI GPT-2 (and HF) are often (In, Out) actually...
        # Wait, HF GPT2 Conv1D weights are (Hidden, 3*Hidden). 
        # Let's check the previous comparison output.
        # "MAT attn w shape: (3072, 1024)" vs "HF attn w shape: (1024, 3072)"
        # So HF is (1024, 3072). MATLAB is (3072, 1024).
        # So MATLAB is (HF_Weight).T
        
        if weight.ndim == 2:
            weight = weight.T
        elif weight.ndim == 1:
            # Reshape 1D bias (N,) to (N, 1) for MATLAB column vector
            weight = weight.reshape(-1, 1)
            
        matlab_params[new_key] = weight

    # Wrap in 'Parameters' struct if that's how the .mat is structured
    # The user's code gpt2.m loads parameters: mdl.Parameters = gpt2.load(paramsStructFile);
    # gpt2.load usually just loads the struct.
    # The .mat file had a 'Parameters' key? 
    # Re-reading previous output: 
    # "Could not find 'Parameters' variable in MAT file. Available variables: ['model_h0_attn...']"
    # Ah! The previous script FAILED to find 'Parameters' key and listed individual variables instead.
    # This means the .mat file is NOT a single struct 'Parameters', but a flat list of variables 
    # that implied a struct when loaded into MATLAB?
    # Or maybe `gpt2.load` constructs the struct?
    
    # Let's check gpt2.load.m briefly if possible, but safer to just save as flat variables 
    # matching the observed content of the official .mat file.
    # The observed content was flat variables like 'model_h0_attn_c_attn_w_0'.
    
    print(f"Saving to {output_mat_path}...")
    scipy.io.savemat(output_mat_path, matlab_params)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch GPT-2 weights to MATLAB .mat format')
    parser.add_argument('--input', type=str, required=True, help='Path to PyTorch model (pytorch_model.bin)')
    parser.add_argument('--output', type=str, default='gpt2_custom_params.mat', help='Output .mat file path')
    
    args = parser.parse_args()
    convert_pytorch_to_matlab(args.input, args.output)
