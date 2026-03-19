from lora.lora_layer import LoraLayer
def inject_lora(model, rank=8):
    for layer in model.trf_blocks:
        layer.att.W_query = LoraLayer(layer.att.W_query, rank)
        layer.att.W_value = LoraLayer(layer.att.W_value, rank)