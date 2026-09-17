# Miniature LoRA Implementation & Experiment

A complete, from-scratch PyTorch implementation of [Low-Rank Adaptation (LoRA)](https://arxiv.org/abs/2106.09685) applied to a custom Transformer model. 

This repository demonstrates the core mechanism of LoRA by pretraining a small GPT model on a generic text corpus and then fine-tuning it on a shifted domain using two methods: Full Fine-Tuning and LoRA.

## Files
* **`lora.py`**: The core implementation. Contains the `LoRALinear` layer which wraps frozen `nn.Linear` weights and adds trainable low-rank matrices ($A$ and $B$). Also includes `inject_lora()` for dynamic model patching.
* **`model.py`**: A character-level GPT decoder-only model. Features separate $Q, K, V$ projections to allow precise LoRA targeting.
* **`train.py`**: The experiment script. Pretrains the base model, then runs a head-to-head comparison between full fine-tuning and LoRA fine-tuning on a new text domain.

## Setup & Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the experiment:
   ```bash
   python train.py
   ```

## Results & Takeaway
The output script directly compares the final loss and trainable parameter counts of both adaptation methods. It successfully demonstrates the real-world engineering trade-off of LoRA: achieving convergence on a new domain while updating only ~1-3% of the parameters required by full fine-tuning.