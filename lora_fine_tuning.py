import torch.nn as nn
class LoraLayer(nn.Module):
    def __init__(self, original_layer, rank):
        super().__init__()
        d = original_layer.weight.shape[1]
        self.original = original_layer      # frozen W
        self.lora_A = nn.Linear(d, rank, bias=False)
        self.lora_B = nn.Linear(rank, d, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x))


def inject_lora(model, rank=8):
    for layer in model.transformer_blocks:
        # replace W_q and W_v with LoraLayer objects
        layer.attn.W_q = LoraLayer(layer.attn.W_q, rank)
        layer.attn.W_v = LoraLayer(layer.attn.W_v, rank)