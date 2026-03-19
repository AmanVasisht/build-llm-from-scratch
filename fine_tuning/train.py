import sys
import os
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import json
import tiktoken

from src.model.architecture import GPTModel
from src import config
from fine_tuning.lora.inject import inject_lora
from fine_tuning.dataset_preparation.alpaca_dataset import AlpacaDataset, collate_fn
import fine_tuning.config_lora as cfg


# load model
model = GPTModel(config)
checkpoint = torch.load(cfg.pre_trained_model_path,
                        map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])

# freeze all weights
for param in model.parameters():
    param.requires_grad = False

# inject lora
inject_lora(model, rank=cfg.rank)

# sanity check
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,}")
print(f"Frozen params:    {total - trainable:,}")

# load dataset
with open(cfg.data_path, "r") as f:
    dataset = json.load(f)

# tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# create dataset and dataloader
train_dataset = AlpacaDataset(dataset, tokenizer, max_len=cfg.max_len)
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

# device and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg.lr
)

# training loop
num_epochs = cfg.num_epochs


for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for step, (input_ids, target_ids) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids, cache=None)[0]

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-100
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

# save lora weights
lora_state = {}
for i, layer in enumerate(model.trf_blocks):
    lora_state[f"layer_{i}_Wq_A"] = layer.att.W_query.lora_A.weight
    lora_state[f"layer_{i}_Wq_B"] = layer.att.W_query.lora_B.weight
    lora_state[f"layer_{i}_Wv_A"] = layer.att.W_value.lora_A.weight
    lora_state[f"layer_{i}_Wv_B"] = layer.att.W_value.lora_B.weight

torch.save(lora_state, cfg.adapter_path)
print("LoRA weights saved")