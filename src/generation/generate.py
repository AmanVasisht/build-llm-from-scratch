import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def sample_next_token(logits, temperature, top_k, top_p):
    logits = logits[:, -1, :]

    if temperature > 0.0:
        logits = logits / temperature

    if top_k is not None:
        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(
            logits < min_val,
            torch.tensor(float("-inf")).to(logits.device),
            logits
        )

    probs = torch.softmax(logits, dim=-1)

    if top_p is not None:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        tokens_to_remove = cumulative_probs - sorted_probs > top_p
        sorted_probs[tokens_to_remove] = 0.0
        sampled_pos = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices[sampled_pos]
    elif temperature > 0.0:
        return torch.multinomial(probs, num_samples=1)
    else:
        return torch.argmax(logits, dim=-1, keepdim=True)

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, top_p=None, eos_id=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits, _ = model(idx_cond)

        idx_next = sample_next_token(logits, temperature, top_k, top_p)

        if eos_id is not None and idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_with_cache(model, idx, max_new_tokens, context_size,
                        temperature=0.0, top_k=None, top_p=None, eos_id=None):
    cache = None

    # phase 1 — prefill
    with torch.no_grad():
        logits, cache = model(idx, cache=cache)

    idx_next = sample_next_token(logits, temperature, top_k, top_p)

    if eos_id is not None and idx_next == eos_id:
        return idx

    idx = torch.cat((idx, idx_next), dim=1)

    # phase 2 — decode loop
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            logits, cache = model(idx_next, cache=cache)

        idx_next = sample_next_token(logits, temperature, top_k, top_p)

        if eos_id is not None and idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx