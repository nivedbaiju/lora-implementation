import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a frozen base weight and a trainable
    low-rank update, as in Hu et al. 2021 (LoRA):

        h = W_0 x + (alpha / r) * B A x

    W_0 is frozen. A is initialized with Kaiming uniform, B is initialized
    to zero so the adapter starts as a no-op (output == base output).
    """

    def __init__(self, base_linear: nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def extra_repr(self):
        return f"r={self.r}, alpha={self.scaling * self.r:.1f}"


def inject_lora(model: nn.Module, target_names=("q_proj", "v_proj"), r: int = 4,
                 alpha: int = 8, dropout: float = 0.0):
    """
    Walks the model and replaces every nn.Linear child whose attribute name
    contains one of `target_names` with a LoRALinear wrapper. Everything
    else in the model stays frozen implicitly, since we only mark the
    injected layers' A/B as trainable (base weights are frozen inside
    LoRALinear, and every other parameter should be frozen by the caller
    before calling this, e.g. via `for p in model.parameters(): p.requires_grad_(False)`).

    Returns (model, num_layers_injected).
    """
    count = 0
    for module in model.modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in child_name for t in target_names):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                count += 1
    return model, count


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


def trainable_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
