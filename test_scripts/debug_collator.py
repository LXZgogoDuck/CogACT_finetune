import torch
from types import SimpleNamespace
from torch.utils.data import DataLoader
from scripts.finetune_collator import FinetuneCollator
from training.dataset_finetune import FinetuneDataset

def debug_finetune_batch():
    # === Setup Config ===
    config = SimpleNamespace(
        proprio_type='poseulerg',
        action_type='delta',
        wrist_key='wrist_image',
        image_key='primary_image_crop',
        force_regenerate_meta=False,
        plot_hist=False,
        action_chunk_size=8,
        proprio_hist_size=0,
        proprio_chunk_size=8,
        data_path="",  # e.g., "datasets/finetune_data"
        mock_data=True,
    )

    # === Instantiate Dataset + Collator ===
    dataset = FinetuneDataset(config=config, train=True)
    collator = FinetuneCollator(tokenizer_name="bert-base-uncased")

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator, num_workers=0)

    # === Get One Batch ===
    batch = next(iter(dataloader))

    # === Print Shape Summary ===
    print("input_ids:", batch["input_ids"].shape)
    print("attention_mask:", batch["attention_mask"].shape)
    print("pixel_values:", batch["pixel_values"].shape)
    print("actions:", batch["actions"].shape)
    print("labels:", batch["labels"].shape)
    print("action_masks:", batch["action_masks"].shape)

    # === Check for NaNs or weird values ===
    assert not torch.isnan(batch["pixel_values"]).any(), "NaN in images!"
    assert not torch.isnan(batch["actions"]).any(), "NaN in actions!"

    print("\n✅ Batch looks good!")

if __name__ == "__main__":
    debug_finetune_batch()
