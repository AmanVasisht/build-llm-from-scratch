import torch
import tiktoken
from src import config
from src.model.architecture import GPTModel
from src.generation.generate import token_ids_to_text, text_to_token_ids, generate_with_cache
import time
# load tokenizer
enc = tiktoken.get_encoding("gpt2")

# initialise model
gpt = GPTModel(config)

# load checkpoint
checkpoint = torch.load("checkpoints/model_and_optimizer.pth", map_location=torch.device("cpu"))
gpt.load_state_dict(checkpoint["model_state_dict"])
gpt.eval()

# generate with kv cache
x = time.time()
token_ids = generate_with_cache(
    model=gpt,
    idx=text_to_token_ids("Where did the author find the couple?", enc),
    max_new_tokens=15,
    context_size=config.context_length,
    top_k=25,
    temperature=1.4
)
y = time.time()
print("time taken to generate with cache is",y-x)
print("Generated Text:\n", token_ids_to_text(token_ids, enc))