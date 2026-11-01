"""Change batch composition while retaining the expected weighted-BCE objective."""
import torch
from torch.utils.data import WeightedRandomSampler


def positive_resampling(labels, generator):
    labels=torch.as_tensor(labels,dtype=torch.long)
    assert set(labels.tolist())=={0,1}
    weights=torch.where(labels==1,3.,1.).double()
    # Z * E_{q(i)=w_i/sum(w)} BCE_i = mean_i(w_i * BCE_i).
    scale=float(weights.mean())
    sampler=WeightedRandomSampler(weights,len(labels),replacement=True,generator=generator)
    return sampler,scale
