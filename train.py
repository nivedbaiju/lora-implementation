"""
LoRA reimplementation demo.

Setup mirrors the paper's actual use case: take a model that already has
useful weights, freeze it, and adapt it to a new domain by training only
small low-rank matrices instead of all parameters.

Since we don't download a real pretrained LM here, phase 1 trains a small
GPT from scratch on a generic corpus to play the role of "the pretrained
model" that phases 2a/2b then adapt to a different, shifted domain.

Run: python train.py
"""

import torch
import torch.nn.functional as F

from model import MiniGPT
from lora import inject_lora, freeze_all, trainable_params

torch.manual_seed(0)

# ---------------------------------------------------------------------
# Toy corpora. PRETRAIN_TEXT plays the role of the generic pretraining
# corpus; FINETUNE_TEXT is a different domain/vocabulary the model has
# not seen, so adapting to it is a meaningful test of both methods.
# ---------------------------------------------------------------------
PRETRAIN_TEXT = (
    "the transformer processes tokens using self attention layers. "
    "each layer mixes information across the sequence and refines the "
    "representation of every token before passing it to the next block. "
) * 40

FINETUNE_TEXT = (
    "the chef added garlic and onions to the pan and let them soften "
    "slowly while the rice cooked gently in the broth on low heat. "
) * 40

chars = sorted(set(PRETRAIN_TEXT + FINETUNE_TEXT))
stoi = {c: i for i, c in enumerate(chars)}
vocab_size = len(chars)


def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)


BLOCK_SIZE = 64


def get_batch(data, batch_size=32):
    ix = torch.randint(0, len(data) - BLOCK_SIZE - 1, (batch_size,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


def run_steps(model, data, steps, params, lr=3e-4, log_every=50, tag=""):
    opt = torch.optim.AdamW(params, lr=lr)
    last_loss = None
    for step in range(steps):
        x, y = get_batch(data)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()
        if step % log_every == 0 or step == steps - 1:
            print(f"[{tag}] step {step:4d}  loss {last_loss:.4f}")
    return last_loss


def build_model():
    return MiniGPT(vocab_size, dim=128, n_heads=4, n_layers=4, block_size=BLOCK_SIZE)


def main():
    pretrain_data = encode(PRETRAIN_TEXT)
    finetune_data = encode(FINETUNE_TEXT)

    # ---- phase 1: pretrain a base model on the generic corpus --------
    model = build_model()
    print("=== phase 1: pretraining base model ===")
    run_steps(model, pretrain_data, steps=300, params=model.parameters(), tag="pretrain")

    _, total = trainable_params(model)
    print(f"\nbase model total params: {total:,}\n")

    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ---- phase 2a: full fine-tuning baseline on the shifted domain ---
    full_model = build_model()
    full_model.load_state_dict(base_state)
    print("=== phase 2a: full fine-tuning on new domain ===")
    full_loss = run_steps(full_model, finetune_data, steps=150,
                           params=full_model.parameters(), tag="full-ft")
    full_trainable, _ = trainable_params(full_model)

    # ---- phase 2b: LoRA fine-tuning on the shifted domain ------------
    lora_model = build_model()
    lora_model.load_state_dict(base_state)
    freeze_all(lora_model)
    lora_model, n_injected = inject_lora(lora_model, target_names=("q_proj", "v_proj"),
                                          r=4, alpha=8)
    lora_params = [p for p in lora_model.parameters() if p.requires_grad]

    print(f"\ninjected LoRA adapters into {n_injected} linear layers\n")
    print("=== phase 2b: LoRA fine-tuning on new domain ===")
    lora_loss = run_steps(lora_model, finetune_data, steps=150,
                           params=lora_params,lr=3e-2, tag="lora-ft")
    lora_trainable, _ = trainable_params(lora_model)

    # ---- summary -------------------------------------------------------
    print("\n=== summary ===")
    print(f"full fine-tuning : {full_trainable:,} trainable params, final loss {full_loss:.4f}")
    print(f"LoRA fine-tuning : {lora_trainable:,} trainable params "
          f"({100 * lora_trainable / full_trainable:.2f}% of full), final loss {lora_loss:.4f}")


if __name__ == "__main__":
    main()
