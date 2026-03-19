import sys
import os
sys.path.insert(0, os.getcwd())
import torch
import tiktoken
from src import config
from src.model.architecture import GPTModel
from src.generation.generate import token_ids_to_text, text_to_token_ids, generate_with_cache
import time

enc = tiktoken.get_encoding("gpt2")

gpt = GPTModel(config)

# load merged model instead of original
checkpoint = torch.load("fine_tuning/checkpoints/merged_model.pth",
                        map_location=torch.device("cpu"))
gpt.load_state_dict(checkpoint["model_state_dict"])
gpt.eval()

# use instruction format matching what model was trained on
prompt = """### Instruction:
Explain what gravity is.

### Response:"""

x = time.time()
token_ids = generate_with_cache(
    model=gpt,
    idx=text_to_token_ids(prompt, enc),
    max_new_tokens=100,
    context_size=config.context_length,
    top_k=25,
    temperature=1.4
)
y = time.time()
print("Time taken:", y - x)
print("Generated Text:\n", token_ids_to_text(token_ids, enc))
