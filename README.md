# build-llm-from-scratch
Building a Large Language Model from scratch: implementing tokenization, transformer architecture, training pipeline, optimization techniques, and LoRA fine-tuning.

# Build LLM From Scratch

A GPT-2 style autoregressive language model implemented from scratch in PyTorch. The project covers the full pipeline — tokenization, dataset preparation, model architecture, training, text generation, and LoRA-based instruction fine-tuning — with no use of HuggingFace or any high-level ML framework.

---

## Project Structure

```
build_llm_from_scratch/
│
├── data/
│   └── train.txt                        # Raw text corpus for pretraining
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
│   └── model_and_optimizer.pth          # Saved after pretraining
│
├── fine_tuning/
│   ├── __init__.py
│   ├── config_lora.py                   # LoRA hyperparameters (rank, lr, epochs)
│   │
│   ├── data/
│   │   └── GPTeacher.json               # Instruction fine-tuning dataset (Alpaca format)
│   │
│   ├── lora/
│   │   ├── __init__.py
│   │   ├── lora_layer.py                # LoraLayer class
│   │   └── inject.py                    # inject_lora, merge_lora functions
│   │
│   ├── dataset_preparation/
│   │   ├── __init__.py
│   │   ├── format.py                    # format_example (instruction/input/response template)
│   │   └── alpaca_dataset.py            # AlpacaDataset class, collate_fn
│   │
│   ├── checkpoints/
│   │   ├── lora_adapter.pt              # Saved LoRA weights (A and B matrices only)
│   │   └── merged_model.pth             # Final merged model (LoRA baked into base weights)
│   │
│   ├── train.py                         # LoRA fine-tuning training loop
│   └── merge_lora.py                    # Merge LoRA adapter into base model weights
│
├── main.py                              # Pretrain the model
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
GPTModel (Pretraining — self-supervised, next token prediction)
   │
   ▼
Checkpoint saved to checkpoints/
   │
   ▼
LoRA Fine-tuning (SFT on instruction/response pairs)
   ├── Freeze all pretrained weights
   ├── Inject A and B matrices alongside W_query and W_value
   ├── Train only A and B (~0.18% of total parameters)
   └── Save lora_adapter.pt
   │
   ▼
Merge LoRA weights into base model  ──►  merged_model.pth
   │
   ▼
Inference with merged model
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train base model
```bash
python main.py
```

### 3. Fine-tune with LoRA
```bash
python fine_tuning/train.py
```

### 4. Merge LoRA weights into base model
```bash
python fine_tuning/merge_lora.py
```

### 5. Generate text
```bash
python inference.py
```

---

## Model Architecture

### MultiHeadAttention
Scaled dot-product causal self-attention. Causal mask registered as a buffer. Q, K, V projections are separate linear layers with optional bias controlled by `qkv_bias`.

### TransformerBlock
Pre-LayerNorm pattern — normalization applied before attention and feed-forward sublayers, with residual connections around both.

### FeedForward
Two linear layers with 4x hidden dimension expansion, using GELU activation (tanh approximation matching GPT-2's original implementation).

### GPTModel
Token embeddings + positional embeddings → N transformer blocks → final LayerNorm → linear output head. Output head shares weights with token embedding layer (weight tying).

---

## Implemented Features

- **KV Cache** — caches K and V vectors during generation, avoids recomputing for previous tokens. ~2x faster for 15 token generation, benefit grows with sequence length.
- **RoPE** — injects positional information by rotating Q and K vectors. No learned parameters, generalizes to longer sequences, encodes relative positions naturally.
- **LR Scheduler** — warmup phase (lr grows 0 → max_lr) followed by cosine decay (max_lr → min_lr). Prevents unstable early updates and overshooting late in training.
- **Gradient Clipping** — clips gradient norm to prevent exploding gradients from destabilizing training.
- **LoRA Fine-tuning** — see below.

---

## LoRA Fine-tuning

LoRA (Low Rank Adaptation) fine-tunes the pretrained model on instruction/response data without updating the original weights.

### How It Works

Instead of updating the full W_query and W_value matrices during fine-tuning, two small matrices A and B are injected alongside them:

```
output = W(x) + B(A(x))
```

- W is frozen — pretrained values never change
- A projects input DOWN to a small bottleneck (rank r)
- B projects back UP to original dimension
- Only A and B are trained

B is initialized to zeros so the LoRA branch contributes nothing at the start, preserving the pretrained model's behavior exactly.

### Parameter Efficiency

| | Parameters |
|---|---|
| Full fine-tuning (all weights) | 162,518,016 |
| LoRA fine-tuning (A and B only) | 294,912 |
| Percentage trained | 0.18% |

### Dataset Format

Uses Alpaca format — JSON file with `instruction`, `input`, and `output` fields. Formatted into a single sequence before tokenization:

```
### Instruction:
<task description>

### Input:
<optional context>

### Response:
<model output>
```

### Prompting the Fine-tuned Model

Use the same instruction format at inference:

```python
prompt = """### Instruction:
Explain what gravity is.

### Response:"""
```

### Merging

After fine-tuning, LoRA weights can be merged directly into the base model:

```
W_final = W + B @ A
```

The merged model has identical structure to the original — no LoRA machinery needed at inference.

---

## Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Pretraining lr | 0.0004 |
| LoRA fine-tuning lr | 1e-4 |
| Loss function | Cross Entropy |
| LoRA rank | 8 |
| LoRA target layers | W_query, W_value |

---

## Dependencies

```
torch
tiktoken
```

## Future Prospects

- RMSNorm and Grouped Query Attention to align architecture with LLaMA/Mistral
- DPO (Direct Preference Optimization) as a simpler alternative to RLHF for preference-based alignment
- Mixture of Experts (MoE) as a replacement for the dense feed-forward layer
- Quantization for memory-efficient inference

---

## Acknowledgements

This project follows and implements concepts from the book
"Build a Large Language Model (From Scratch)" by Sebastian Raschka.

The goal is to reproduce, experiment with, and deepen understanding of LLM architectures by implementing them step-by-step.

## Related Writing
- [Positional Embeddings in LLMs — Sinusoidal, Learned and RoPE](https://medium.com/p/bfd88cedd4c4?postPublishedType=initial)