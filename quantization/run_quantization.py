import sys
import os
sys.path.insert(0, os.getcwd())

import torch
from src.model.architecture import GPTModel
from src import config
from quantization.quantize import (
    quantize_model,
    save_quantized,
    load_and_dequantize,
    compare_sizes
)

# load model
model = GPTModel(config)
checkpoint = torch.load("checkpoints/model_and_optimizer.pth",
                        map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# quantize and save
quantized_state = quantize_model(model)
save_quantized(quantized_state, "quantization/quantized_weights.pt")

# verify dequantization works
model = load_and_dequantize(model, "quantization/quantized_weights.pt")
print("done")

# save model weights only (no optimizer state)
torch.save(
    {"model_state_dict": model.state_dict()},
    "quantization/model_only.pt"
)

# now compare model only vs quantized
compare_sizes(
    "quantization/model_only.pt",
    "quantization/quantized_weights.pt"
)