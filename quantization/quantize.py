
import os


import torch
import torch.nn as nn

def quantize_tensor(tensor):
    # find scale factor
    max_val = tensor.abs().max()
    scale = 127.0 / max_val

    # quantize: multiply, round, clamp to int8 range, convert to int8
    quantized = (tensor * scale).round().clamp(-128, 127).to(torch.int8)

    return quantized, scale

def dequantize_tensor(quantized, scale):
    return quantized.to(torch.float32) / scale

def quantize_model(model):
    quantized_state = {}

    for name, module in model.named_modules():
        # only quantize nn.Linear layers
        if isinstance(module, nn.Linear):
            quantized_w, scale = quantize_tensor(module.weight.data)

            quantized_state[f"{name}.weight"] = quantized_w
            quantized_state[f"{name}.scale"]  = scale

            print(f"quantized {name} | shape {module.weight.shape} | scale {scale:.4f}")

    return quantized_state


def save_quantized(quantized_state, path):
    torch.save(quantized_state, path)
    print(f"quantized model saved to {path}")

def load_and_dequantize(model, path):
    quantized_state = torch.load(path)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            quantized_w = quantized_state[f"{name}.weight"]
            scale       = quantized_state[f"{name}.scale"]

            # dequantize and put back into model
            module.weight.data = dequantize_tensor(quantized_w, scale)

    print("model dequantized and loaded")
    return model

def compare_sizes(original_path, quantized_path):
    original_size   = os.path.getsize(original_path)   / (1024 * 1024)
    quantized_size  = os.path.getsize(quantized_path)  / (1024 * 1024)

    print(f"original model:   {original_size:.1f} MB")
    print(f"quantized model:  {quantized_size:.1f} MB")
    print(f"reduction:        {original_size / quantized_size:.1f}x smaller")