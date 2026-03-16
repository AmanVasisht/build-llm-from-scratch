vocab_size= 50257   # Vocabulary size
context_length= 256 # Shortened context length (orig= 1024)
stride=256
emb_dim= 768        # Embedding dimension
n_heads= 12         # Number of attention heads
n_layers= 12        # Number of layers
drop_rate= 0.2      # Dropout rate
qkv_bias= False     # Query-key-value bias
train_ratio= 0.90
batch_size=2
epoch=10
# scheduler
use_scheduler = True
max_lr        = 0.0004
min_lr        = 1e-5
warmup_steps  = 430

# gradient clipping
max_grad_norm = 1.0

#positional embedding
use_rope = True