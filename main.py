import torch
import tiktoken
import time
from src import config
from src.preprocessing.dataloader import create_dataloader_v1, GPTDatasetV1
from src.model.architecture import GPTModel
from src.training.trainer import train_model_simple


# Load GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")

# Read raw training corpus
with open('data/train.txt', 'r', encoding='utf-8') as f:
    corpus = f.read()

print('Corpus size (characters):', len(corpus))

# Split corpus into train and validation sets
cutoff = int(config.train_ratio * len(corpus))
text_train = corpus[:cutoff]
text_val = corpus[cutoff:]

torch.manual_seed(123)

# Build dataloaders using sliding window approach
loader_train = create_dataloader_v1(
    text_train,
    batch_size=config.batch_size,
    max_length=config.context_length,
    stride=config.stride,
    drop_last=True,
    shuffle=True,
    num_workers=0
)

loader_val = create_dataloader_v1(
    text_val,
    batch_size=config.batch_size,
    max_length=config.context_length,
    stride=config.stride,
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# Count total tokens across splits
n_train_tokens = sum(batch.numel() for batch, _ in loader_train)
n_val_tokens = sum(batch.numel() for batch, _ in loader_val)

print(f"Train tokens : {n_train_tokens}")
print(f"Val tokens   : {n_val_tokens}")
print(f"Total tokens : {n_train_tokens + n_val_tokens}")

# Initialise GPT model from config
torch.manual_seed(123)
gpt = GPTModel(config)
gpt.eval()

# Parameter count (full vs weight-tied)
n_params = sum(p.numel() for p in gpt.parameters())
n_params_tied = n_params - sum(p.numel() for p in gpt.out_head.parameters())
print(f"Total parameters            : {n_params:,}")
print(f"Parameters (weight tied)    : {n_params_tied:,}")

# Estimate model size in memory
model_size_mb = (n_params * 4) / (1024 ** 2)
print(f"Approximate model size      : {model_size_mb:.2f} MB")

# Print architecture summary
print(gpt.eval())

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt.to(device)

torch.manual_seed(123)

# Optimiser
optim = torch.optim.AdamW(gpt.parameters(), lr=0.0004, weight_decay=0.1)

# Run training
t_start = time.time()

loss_train, loss_val, tokens_seen = train_model_simple(
    gpt, loader_train, loader_val, optim, device,
    num_epochs=config.epoch,
    eval_freq=5,
    eval_iter=5,
    start_context="He found the couple",
    tokenizer=enc,
    save_model=True
)

t_end = time.time()
print(f"Training completed in {(t_end - t_start) / 60:.2f} minutes.")