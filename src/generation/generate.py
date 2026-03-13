import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Original generate function — no KV cache, works as before."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_with_cache(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    KV cache enabled generation.
    First step processes the full prompt and builds the cache.
    Subsequent steps only process one new token at a time using cached K, V.
    """
    cache = None

    # step 1 — prefill: process entire prompt at once, build cache
    with torch.no_grad():
        logits, cache = model(idx, cache=cache)

    # get next token from last position of prompt
    logits = logits[:, -1, :]

    if top_k is not None:
        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

    if temperature > 0.0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
    else:
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

    if idx_next == eos_id:
        return idx

    idx = torch.cat((idx, idx_next), dim=1)

    # step 2 — decode: generate remaining tokens one at a time using cache
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            # only pass the latest token, cache holds the rest
            logits, cache = model(idx_next, cache=cache)

        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx