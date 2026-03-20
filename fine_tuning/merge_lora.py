import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn

from src.model.architecture import GPTModel
from src import config
from fine_tuning.lora.inject import inject_lora


def merge_lora(model):
    for layer in model.trf_blocks:
        # merge W_query
        W_q = layer.att.W_query.original.weight
        A_q = layer.att.W_query.lora_A.weight
        B_q = layer.att.W_query.lora_B.weight
        W_q.data += (B_q @ A_q)
        layer.att.W_query = layer.att.W_query.original

        # merge W_value
        W_v = layer.att.W_value.original.weight
        A_v = layer.att.W_value.lora_A.weight
        B_v = layer.att.W_value.lora_B.weight
        W_v.data += (B_v @ A_v)
        layer.att.W_value = layer.att.W_value.original

    print("LoRA weights merged into base model successfully")


def load_lora_weights(model, adapter_path):
    lora_state = torch.load(adapter_path, map_location="cpu")

    for i, layer in enumerate(model.trf_blocks):
        layer.att.W_query.lora_A.weight = nn.Parameter(lora_state[f"layer_{i}_Wq_A"])
        layer.att.W_query.lora_B.weight = nn.Parameter(lora_state[f"layer_{i}_Wq_B"])
        layer.att.W_value.lora_A.weight = nn.Parameter(lora_state[f"layer_{i}_Wv_A"])
        layer.att.W_value.lora_B.weight = nn.Parameter(lora_state[f"layer_{i}_Wv_B"])

    print(f"LoRA adapter loaded from {adapter_path}")


# step 1 — load base model
model = GPTModel(config)
checkpoint = torch.load(
    "checkpoints/model_and_optimizer.pth",
    map_location=torch.device("cpu")
)
model.load_state_dict(checkpoint["model_state_dict"])
print("Base model loaded")

# step 2 — freeze and inject lora structure
for param in model.parameters():
    param.requires_grad = False
inject_lora(model, rank=8, alpha=8)

# step 3 — load trained lora weights into injected structure
load_lora_weights(model, "fine_tuning/checkpoints/lora_adapter.pt")

# step 4 — merge lora into base weights
merge_lora(model)

# step 5 — save merged model
torch.save(
    {"model_state_dict": model.state_dict()},
    "fine_tuning/checkpoints/merged_model.pth"
)
print("Merged model saved to fine_tuning/checkpoints/merged_model.pth")

# sanity check — verify no LoraLayer objects remain
has_lora = False
for layer in model.trf_blocks:
    if hasattr(layer.att.W_query, "lora_A"):
        has_lora = True
        break
print(f"LoRA layers remaining: {has_lora}")   # should print False

