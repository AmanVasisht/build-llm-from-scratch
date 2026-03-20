from lora.lora_layer import LoraLayer
def inject_lora(model, rank=8, alpha=None):
    for layer in model.trf_blocks:
        layer.att.W_query = LoraLayer(layer.att.W_query, rank, alpha)
        layer.att.W_value = LoraLayer(layer.att.W_value, rank, alpha)