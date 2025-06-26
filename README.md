# 🔧 CogACT Finetune Pipeline (Mock Data)

This project sets up a mock fine-tuning pipeline for CogACT by modifying the dataset interface and training logic to accept randomly generated data in the correct format.


## Modifications

### File: `training/dataset_finetune.py`

modify the original `FinetuneDataset` to generate mock data

#### Key Changes

- Introduced config flags:
  ```python
  config.mock_data = True
  config.mock_data_num_trajs = 64
- When `mock_data=True`, the dataset:
  - Skips loading metadata/statistics files. 
  - Randomly generates proprioception, joint states, and action sequences. 
  - Fakes visual input with dummy white images. 
  - Randomly selects from predefined language instructions.
  
_Output format:_

```python
{
    'observation': {
        'image_primary': Tensor [chunk_size, H, W, C],
        'image_wrist': Tensor [chunk_size, H, W, C] or None,
        'proprio': Tensor [chunk_size, D]
    },
    'action': Tensor [chunk_size, D],
    'task': {
        'language_instruction': str
    }
}
```
### File: `training/collator_finetune.py`
Defines the FinetuneCollator class, prepares batches from individual samples.

- Converts image format [H, W, C] → [C, H, W] and stacks into (B, T, C, H, W)
- Handles optional wrist camera images
- Stacks proprioception and action vectors
- Constructs final input dictionary compatible with CogACT

_Output format:_
```python
{
    "input_ids": LongTensor [B, L],           # tokenized instructions
    "attention_mask": LongTensor [B, L],     
    "pixel_values": {
        "dino": FloatTensor [B, T, C, H, W],  # main camera images
        "siglip": FloatTensor [B, T, C, H, W],
        "wrist": FloatTensor [B, T, C, H, W], # optional wrist camera
    },
    "proprio": FloatTensor [B, T, D],       
    "actions": FloatTensor [B, T, D],         # ground truth action sequence
    "action_masks": FloatTensor [B, T],       # dummy mask (all 1s)
    "labels": FloatTensor [B, T, D],          # training target = actions
}
```

### File: `training/vla.py`:
add new experiment config: Exp_Custom_Finetune

### File: `scripts/train.py`
#### Key Changes:
- When vla_id == "custom-finetune", uses FinetuneDataset with mock data.

- Minimal Dataset Config via SimpleNamespace:
  ```python
  config = SimpleNamespace(
    proprio_type='poseulerg',
    action_type='delta',
    wrist_key='wrist_image',
    image_key='primary_image_crop',
    ...
    mock_data=True,)
  ```
- Uses FinetuneCollator for batching and tokenization.

## Results
![result](assets/result.png)
```
| >> [*] FSDP Full-Shard Strategy =>> Finalized Training Setup:                                                           fsdp.py:288
                                   |-> Global (Effective) Batch Size = 16                                                                                      
                                   |-> Per-Device Batch Size = 1                                                                                               
                                   |-> Distributed World Size = 4                                                                                              
                                   |-> Gradient Accumulation Steps = 4                                                                                         
                                                                                                                                                               
                                   |-> LLM Backbone FSDP Gradient Checkpointing = True                                                                         
                                   |-> Use FSDP Mixed Precision = True                                                                                         
                                           |-> Parameter Precision = torch.bfloat16                                                                            
                                           |-> Reduction Precision = torch.float32                                                                             
                                           |-> Buffer Precision = torch.float32                                                                                
                                                                                                                                                               
                                   |-> Default AdamW LR = 0.0001                                                                                               
                                   |-> AdamW Weight Decay = 0.01                                                                                               
                                   |-> LR Scheduler Type = linear-warmup+cosine-decay                                                                          
                                   |-> LR Scheduler Warmup Steps (Ratio) = 5 (0.05)                                                                            
                                   |-> Dataset Size = 32 Examples                                                                                              
                                   |-> Max Steps = 100     
```
Problem: 

torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 172.00 MiB. GPU 3 has a total capacity of 21.95 GiB of which 154.12 MiB is free. 
Process 82739 has 21.79 GiB memory in use. Of the allocated memory 21.04 GiB is allocated by PyTorch, and 118.33 MiB is reserved by PyTorch but unallocated. 
If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  

Trial: 

reduce batch_size_per_device, reduce number of trajectories generated -> still OOM

##  Set Up

```bash
git clone https://github.com/LXZgogoDuck/CogACT_finetune.git
cd CogACT_finetune
```
```bash
docker run --gpus all --rm --network host --ipc=host \
  -v /users/vis24xl/xuanzhuo:/workspace \
  -v /data/auriga:/mnt/auriga \
  --entrypoint "" \
  -it cog /bin/bash
```
```bash
conda init
source ~/.bashrc
conda activate cogact
```
```bash
export HF_TOKEN=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

torchrun --standalone --nproc-per-node=4 scripts/train.py \
  --vla.type custom-finetune \
  --pretrained_checkpoint CogACT/CogACT-Base \
  --hf_token $HF_TOKEN \
  --is_resume false
```