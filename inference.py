import torch
import tiktoken
from src.model.architecture import GPTModel
from src import config
from src.generation.generate import token_ids_to_text, text_to_token_ids, generate

tokenizer = tiktoken.get_encoding("gpt2")  # Ensure vocab size matches model trained vocab size
model = GPTModel(config)

checkpoint = torch.load("checkpoints/model_and_optimizer.pth", map_location=torch.device("cpu"))  # or "cuda" if using GPU
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

max_new_tokens = 20  # Number of tokens to generate

token_ids = generate(
    model=model,
    idx=text_to_token_ids("I think it is", tokenizer),
    max_new_tokens=15,
    context_size=config.context_length,
    top_k=25,
    temperature=1.4
)

output_text = token_ids_to_text(token_ids, tokenizer)
print("Generated Text:\n", output_text)
