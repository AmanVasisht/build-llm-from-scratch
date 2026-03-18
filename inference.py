import torch
import tiktoken
from src.model.architecture import GPTModel
from src import config
from src.generation.generate import token_ids_to_text, text_to_token_ids, generate
import time
# ------------------------------
# 1. Load tokenizer (adjust as needed)
# ------------------------------
tokenizer = tiktoken.get_encoding("gpt2")  # Ensure vocab size matches model

# ------------------------------
# 2. Initialize model
# ------------------------------
model = GPTModel(config)

# ------------------------------
# 3. Load checkpoint
# ------------------------------
checkpoint = torch.load("checkpoints/model_and_optimizer.pth", map_location=torch.device("cpu"))  # or "cuda" if using GPU
model.load_state_dict(checkpoint["model_state_dict"])

# ------------------------------
# 4. Set model to evaluation mode
# ------------------------------
model.eval()

# ------------------------------
# 5. Prepare input prompt
# ------------------------------
# input_text = "Once upon a time"
# token_ids = tokenizer.encode(input_text, allowed_special={"<|endoftext|>"})
#   # shape: (1, context_length)
# input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
# ------------------------------
# 6. Generate tokens autoregressively
# ------------------------------

print(model)
max_new_tokens = 15  # Number of tokens to generate
x = time.time()
token_ids = generate(
    model=model,
    idx=text_to_token_ids("I think he is", tokenizer),
    max_new_tokens=15,
    context_size=config.context_length,
    top_k=25,
    temperature=1.4
)
y = time.time()
print("Time required to generate without cache",y-x)
# ------------------------------
# 7. Decode generated tokens
# ------------------------------
# output_text = tokenizer.decode(generated[0].tolist())
output_text = token_ids_to_text(token_ids, tokenizer)
print("Generated Text:\n", output_text)
