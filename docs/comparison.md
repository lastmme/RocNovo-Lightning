# Comparison with the Original Implementation

This repository builds upon the original repository([RocNovo](https://github.com/lastmme/RocNovo)).

## Shared Configurations
To maintain the integrity of the feature extraction logic, we kept the following core settings identical to the original repository:

- The model architecture and dimensional settings remain exactly the same.
- Both implementations utilize a two-stage training paradigm: CLIP pre-training followed by *de novo* post training.
- The training epochs is strictly maintained at 10 for both the CLIP and *de novo* models.
- The learning rate warmup strategies and the specific number of warmup steps are identical.

## Core Differences
While the foundational structure remains the same, our implementation introduces the following key comparisons:

- **Mixed Precision Training:** While the original repository trains the models entirely in standard FP32 precision, this repository fully adopts BF16 mixed precision training.
- **Realistic Data Augmentation (CLIP):** To introduce extreme noise during the CLIP model training, the original repository may randomly return a noise spectrum peak tensor [0, 1]. In contrast, we replaced this with mass spectrometry data augmentation strategies.
- **Temperature Scaling (CLIP):** The original repository uses a fixed temperature value of 0.05 for contrastive loss. In this repository, we implemented temperature scaling as a learnable parameter, initializing it at 0.05.
- **Differentiable Distributed Communication (CLIP):** During multi-GPU training for the CLIP model, the original repository aggregates tensors without preserving the computation graph. We implemented differentiable distributed communication (using `torch.distributed.nn.all_gather` with gradients attached), allowing gradients to seamlessly flow back across all devices during contrastive learning.
- **Scaled Batch Sizes:** The originasitory uses a batch size of 200 for the CLIP model and 64 for the *de novo* model. This repository increases these to 300 and 192, respectively.
- **Inference Speedup:** For the inference stage, we fully implemented Key-Value (KV) Cache support during the autoregressive decoding process, accelerating the *de novo* sequencing generation speed.