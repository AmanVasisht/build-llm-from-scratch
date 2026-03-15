import torch
import math


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits, _ = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def get_lr(step, warmup_steps, total_steps, max_lr, min_lr=1e-5):
    """
    Cosine learning rate schedule with linear warmup.
    - Linearly increases lr from min_lr to max_lr over warmup_steps
    - Cosine decays lr from max_lr to min_lr over remaining steps
    """
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * (step / warmup_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay


def generate_and_print_sample(model, tokenizer, device, start_context):
    from src.generation.generate import text_to_token_ids, token_ids_to_text, generate
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, save_model=True,
                       use_scheduler=True, max_lr=0.0004, min_lr=1e-5, warmup_steps=20,
                       max_grad_norm=1.0):
    """
    Training loop with:
      - cosine LR schedule with linear warmup (use_scheduler=True)
      - gradient clipping (max_grad_norm=1.0)
    """
    train_losses, val_losses, track_tokens_seen, lr_history = [], [], [], []
    tokens_seen, global_step = 0, -1

    total_steps = num_epochs * len(train_loader)

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # update lr at every step based on schedule
            if use_scheduler:
                lr = get_lr(global_step, warmup_steps, total_steps, max_lr, min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                lr_history.append(lr)

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # clip gradients before optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                      f"LR {current_lr:.6f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    if save_model is True:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, "checkpoints/model_and_optimizer.pth")

    return train_losses, val_losses, track_tokens_seen, lr_history