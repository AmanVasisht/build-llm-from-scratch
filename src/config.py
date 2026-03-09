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