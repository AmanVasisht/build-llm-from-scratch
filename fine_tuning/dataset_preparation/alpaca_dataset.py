import torch
from dataset_preparation.format import format_example
from torch.utils.data import Dataset

class AlpacaDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        for example in data:
            if not example["output"]:
                continue
            text = format_example(example)
            self.examples.append(text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        tokens = self.tokenizer.encode(text)
        tokens = tokens[:self.max_len]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids


def collate_fn(batch):
    input_ids, target_ids = zip(*batch)
    max_len = max(x.size(0) for x in input_ids)
    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(input_ids, target_ids):
        pad_len = max_len - inp.size(0)
        padded_inputs.append(
            torch.cat([inp, torch.full((pad_len,), 0, dtype=torch.long)])
        )
        padded_targets.append(
            torch.cat([tgt, torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return torch.stack(padded_inputs), torch.stack(padded_targets)
