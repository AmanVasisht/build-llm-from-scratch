# build-llm-from-scratch
Building a Large Language Model from scratch: implementing tokenization, transformer architecture, training pipeline, and optimization techniques.

# Build LLM From Scratch

A GPT-2 style autoregressive language model implemented from scratch in PyTorch. The project covers the full pipeline — tokenization, dataset preparation, model architecture, training, and text generation — with no use of HuggingFace or any high-level ML framework.

---

## Project Structure

```
build_llm_from_scratch/
│
├── data/
│   └── train.txt                        # Raw text corpus for training
│
├── src/
│   ├── __init__.py
│   ├── config.py                        # All hyperparameters in one place
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── dataloader.py                # GPTDatasetV1, create_dataloader_v1
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   └── architecture.py              # GPTModel, TransformerBlock, MultiHeadAttention,
│   │                                      FeedForward, LayerNorm, GELU
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                   # train_model_simple, evaluate_model, loss utils
│   │
│   └── generation/
│       ├── __init__.py
│       └── generate.py                  # generate, generate_with_cache, text_to_token_ids, token_ids_to_text
│
├── checkpoints/
│   └── model_and_optimizer.pth          # Saved after training
│
├── outputs/
├── notebooks/
│
├── main.py                              # Train the model
├── inference.py                         # Load checkpoint and generate text
├── inference_kv.py                      # Generate text with KV cache optimization
├── requirements.txt
└── .gitignore
```

---

## Pipeline Overview

```
Raw Text
   │
   ▼
Tokenizer (tiktoken GPT-2 BPE)
   │
   ▼
Sliding Window Dataset  ──►  DataLoader (train / val split)
   │
   ▼
GPTModel
   ├── Token Embedding + Positional Embedding
   ├── Dropout
   ├── N x TransformerBlock
   │     ├── LayerNorm
   │     ├── MultiHeadAttention (causal mask)
   │     ├── Dropout + Residual
   │     ├── LayerNorm
   │     ├── FeedForward (GELU activation)
   │     └── Dropout + Residual
   ├── Final LayerNorm
   └── Linear Output Head (weight tied with token embedding)
   │
   ▼
AdamW Optimizer  ──►  Cross Entropy Loss  ──►  Training Loop
   │
   ▼
Checkpoint saved to checkpoints/
   │
   ▼
inference.py     ──►  Generate text from prompt
inference_kv.py  ──►  Generate text with KV cache (2x faster)
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add training data
Place any `.txt` file inside `data/` and name it `train.txt`. The model was originally trained on a short story (~5k tokens) — any plain text corpus works.

### 3. Configure
Edit `src/config.py` to set model size and training hyperparameters:

```python
vocab_size      = 50257   # GPT-2 vocabulary size
context_length  = 256     # Sequence length (tokens)
emb_dim         = 768     # Embedding dimension
n_heads         = 12      # Attention heads
n_layers        = 12      # Transformer blocks
drop_rate       = 0.1     # Dropout probability
qkv_bias        = False

batch_size      = 2
stride          = 128
train_ratio     = 0.90
epoch           = 10
# scheduler
use_scheduler = True
max_lr        = 0.0004
min_lr        = 1e-5
warmup_steps  = 430

# gradient clipping
max_grad_norm = 1.0

#positional embedding
use_rope = True
```

### 4. Train
```bash
python main.py
```

Training prints loss at every `eval_freq` steps and generates a sample after each epoch. Checkpoint is saved to `checkpoints/model_and_optimizer.pth` at the end.

### 5. Generate text
```bash
python inference.py
```

### 6. Generate text with KV cache
```bash
python inference_kv.py
```

---

## Model Architecture

### MultiHeadAttention
Implements scaled dot-product causal self-attention. The causal mask is registered as a buffer (upper triangular), ensuring each token can only attend to past tokens. Q, K, V projections are separate linear layers with an optional bias controlled by `qkv_bias`.

### TransformerBlock
Each block follows the Pre-LayerNorm (Pre-LN) pattern — normalization is applied before attention and before the feed-forward sublayer, with residual connections around both.

### FeedForward
Two linear layers with a 4x expansion in the hidden dimension, using a custom GELU activation (tanh approximation matching GPT-2's original implementation).

### GPTModel
Combines token embeddings and learned positional embeddings, passes through N transformer blocks, applies a final layer norm, and projects to vocabulary logits via a linear head. The output head shares weights with the token embedding layer (weight tying).

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.0004 |
| Weight decay | 0.1 |
| Loss function | Cross Entropy |
| Evaluation | Every 5 steps on both train and val |
| Generation during training | After every epoch (greedy decoding) |

---

## Inference Details

The `generate()` function supports:
- **Greedy decoding** — picks the highest probability token at each step (default)
- **Temperature scaling** — controls randomness of output (`temperature > 0`)
- **Top-k sampling** — restricts sampling to the top-k most likely tokens
- **Early stopping** — stops if `eos_id` token is generated

---

## Implemented features on basic llm architecture:
- **KV cache**: The network needs only last hidden state to predict next token. the last hidden state depends only on last token's query vector and all key and value vectors. Thus to keep on predicting tokens, the same previous key and value vectors are utilised, only query vector changes. Thus they can be preserved in a cache.
- **Learning rate scheduler**: A fixed lr is too cautious early and too aggressive late. Thus there is need of big steps to learn fast in early training and small steps to learn slowly in late training. In short lr scheduler controls how big each step is over time
warmup phase:    lr grows from 0 → 0.0004 over first N steps
cosine decay:    lr gradually decreases from 0.0004 → 0 following a cosine curve
- **Gradient clipping**: During backpropagation, gradients can sometimes become very large — called **exploding gradients**. This causes the optimizer to make a massive weight update in one step, completely destabilizing training. In short gradient clipping controls how big each step can ever be at most.
- **RoPE (Rotary Positional Embeddings)**: Added Rotary Positional Embeddings as an optional alternative to learnable positional embeddings. Unlike standard positional embeddings that are added directly to token embeddings, RoPE injects positional information by rotating the **query (Q)** and **key (K)** vectors in the attention mechanism. This preserves the magnitude of token embeddings while encoding positional relationships through rotation in the embedding space. The rotation angles are derived from sinusoidal frequency functions, similar in spirit to sinusoidal positional encodings. This approach enables the model to naturally capture **relative positional relationships** between tokens and has become widely used in modern large language models.


## KV Cache

KV cache is implemented as an inference optimization in `inference_kv.py`.

```
without cache:  step N recomputes K,V for all N tokens  →  O(N²)
with cache:     step N computes K,V for 1 token only    →  O(N)
```

Measured speedup on this model: **~2x faster** for 15 token generation. The benefit grows significantly with longer sequences and larger models.

The cache is a list of 12 dictionaries (one per transformer block), each storing K and V tensors in RAM. It lives only for the duration of the generation call — nothing is persisted to disk.

---

## Dependencies

```
torch
tiktoken
```

---

## Notes

- This is a from-scratch implementation for learning purposes — no pretrained weights are used
- The architecture closely follows GPT-2 small but is trained on a tiny corpus, so generated text is limited in quality
- The goal is to understand every component of a transformer LLM by building it line by line

---

## Future Prospects

Planned additions to this repository in order of priority:

- Learning rate scheduler (cosine + warmup), gradient clipping, and loss curve visualization to complete the current training pipeline
- RoPE, RMSNorm, and Grouped Query Attention to align the architecture with modern LLMs like LLaMA and Mistral
- Fine tuning on a larger real world corpus to produce meaningful generation quality
- Mixture of Experts (MoE) as a replacement for the dense feed-forward layer
- Knowledge distillation, RAG, and reasoning model techniques (RLHF, GRPO) as longer term explorations


## Acknowledgements

This project follows and implements concepts from the book
"Build a Large Language Model (From Scratch)" by Sebastian Raschka.

The goal of this repository is to reproduce, experiment with, and deepen
understanding of LLM architectures by implementing them step-by-step.

## Related Writing
- [Positional Embeddings in LLMs — Sinusoidal, Learned and RoPE](https://medium.com/p/bfd88cedd4c4?postPublishedType=initial)