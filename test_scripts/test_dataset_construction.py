from training.dataset_finetune import FinetuneDataset
from types import SimpleNamespace
import torch

config = SimpleNamespace(
    proprio_type='poseulerg',
    action_type='delta',
    wrist_key='wrist_image',
    image_key='primary_image_crop',
    force_regenerate_meta=False,
    plot_hist=True,
    action_chunk_size=15,
    proprio_hist_size=0,
    proprio_chunk_size=8,
    data_path="/tmp/mock_finetune",
    mock_data=True,
)

dataset = FinetuneDataset(config=config, train=True)

# Print some sample data
sample = next(iter(dataset))
print("Keys:", sample.keys())
print("Observation keys:", sample["observation"].keys())
print("Image primary shape:", sample["observation"]["image_primary"].shape)
print("Proprio shape:", sample["observation"]["proprio"].shape)
print("Action shape:", sample["action"].shape)
print("Instruction:", sample["task"]["language_instruction"])
