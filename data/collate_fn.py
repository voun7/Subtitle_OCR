import torch


class Collator:
    def __init__(self, tensor_keys: list) -> None:
        self.tensor_keys = tensor_keys

    def collate_fn(self, batch: list) -> dict:
        tensor_batch = {}
        for sample in batch:
            for key, value in sample.items():
                if key in self.tensor_keys:
                    value = torch.from_numpy(value)
                tensor_batch.setdefault(key, []).append(value)
        for key, value in tensor_batch.items():
            if key in self.tensor_keys:
                tensor_batch[key] = torch.stack(value)
        return tensor_batch
