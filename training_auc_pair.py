"""Class-present cyclic batches and an explicit pairwise AUC surrogate."""
import torch
import torch.nn.functional as F


class ClassPresentBatches:
    def __init__(self,labels,batch_size,generator):
        self.labels=torch.as_tensor(labels,dtype=torch.long);self.batch_size=batch_size;self.generator=generator
        assert set(self.labels.tolist())=={0,1} and batch_size>=2
        self.sizes=[batch_size]*(len(labels)//batch_size);tail=len(labels)%batch_size
        if tail==1 and self.sizes:self.sizes[-1]-=1;tail+=1
        if tail:self.sizes.append(tail)
    def __len__(self):return len(self.sizes)
    def __iter__(self):
        pools=[torch.where(self.labels==c)[0] for c in (0,1)];orders=[[],[]];cursor=[0,0]
        def take(c,n):
            values=[]
            while len(values)<n:
                if cursor[c]==len(orders[c]):
                    orders[c]=pools[c][torch.randperm(len(pools[c]),generator=self.generator)].tolist();cursor[c]=0
                count=min(n-len(values),len(orders[c])-cursor[c]);values+=orders[c][cursor[c]:cursor[c]+count];cursor[c]+=count
            return values
        for size in self.sizes:
            positives=min(2,size-1);batch=take(1,positives)+take(0,size-positives)
            yield [batch[i] for i in torch.randperm(size,generator=self.generator).tolist()]


def auc_pair_loss(logits,labels,train_prior):
    logits=logits.float();positive=logits[labels==1];negative=logits[labels==0]
    assert positive.numel() and negative.numel()
    rank=F.relu(1-positive[:,None]+negative[None,:]).square().mean()
    # Conditional averages retain the old expected weighted-BCE3 objective.
    auxiliary=3*train_prior*F.softplus(-positive).mean()+(1-train_prior)*F.softplus(negative).mean()
    return rank+.1*auxiliary
