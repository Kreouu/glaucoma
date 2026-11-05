"""Positive BCE retained; easy-negative loss modulated without clipping."""
import torch
import torch.nn.functional as F

class NegativeModulatedBCE(torch.nn.Module):
    def __init__(self,gamma,pos_weight):
        super().__init__();self.gamma=gamma;self.pos_weight=pos_weight
    def forward(self,logits,targets):
        z,y=logits.float(),targets.float()
        return (self.pos_weight*y*F.softplus(-z)+(1-y)*z.sigmoid().pow(self.gamma)*F.softplus(z)).mean()
