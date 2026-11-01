"""Binary focal modulation with the existing positive-class cost retained."""
import torch
import torch.nn.functional as F


class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, gamma, pos_weight):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weighted = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=logits.new_tensor(self.pos_weight))
        return ((1 - torch.exp(-bce)).pow(self.gamma) * weighted).mean()
