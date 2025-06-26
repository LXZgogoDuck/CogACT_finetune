import torch
import numpy as np
from transformers import AutoTokenizer

class FinetuneCollator:
    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def to_tensor(self, x):
        # Ensure input is a torch.Tensor
        return torch.from_numpy(x) if isinstance(x, np.ndarray) else x

    def __call__(self, batch):
        # Primary image: (B, T, H, W, C) or (B, T, C, H, W)
        image_primary = [self.to_tensor(item['observation']['image_primary']) for item in batch]
        image_primary = torch.stack(image_primary)
        if image_primary.dim() == 5 and image_primary.shape[-1] in {1, 3}:  # Convert HWC → CHW
            image_primary = image_primary.permute(0, 1, 4, 2, 3).contiguous()

        # Optional wrist image
        wrist_sample = batch[0]['observation'].get('image_wrist')
        if wrist_sample is not None:
            image_wrist = torch.stack([
                self.to_tensor(item['observation']['image_wrist']) for item in batch
            ])
            if image_wrist.shape[-1] in {1, 3}:
                image_wrist = image_wrist.permute(0, 1, 4, 2, 3).contiguous()
        else:
            image_wrist = None

        # Proprioception and actions
        proprio = torch.stack([self.to_tensor(item['observation']['proprio']) for item in batch])  # (B, T, D)
        actions = torch.stack([self.to_tensor(item['action']) for item in batch])                  # (B, T, D)

        # Tokenize instructions
        instructions = [item['task']['language_instruction'] for item in batch]
        tokenized = self.tokenizer(instructions, return_tensors='pt', padding=True, truncation=True)

        # Prepare image dict
        pixel_values = {
            "dino": image_primary.float(),
            "siglip": image_primary.float(),
        }
        if image_wrist is not None:
            pixel_values["wrist"] = image_wrist.float()

        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "pixel_values": pixel_values,
            "proprio": proprio,
            "actions": actions,
            "action_masks": torch.ones_like(actions[..., 0]),  # dummy mask
            "labels": actions.clone(),                         # same as actions for supervised
        }
