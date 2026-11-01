import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from training_head_lr import classifier_lr_groups


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=torch.nn.Linear(4,3);self.classifier=torch.nn.Linear(3,1)
    def get_classifier(self):return self.classifier


class HeadLRTest(unittest.TestCase):
    def test_partition_and_first_adamw_update_ratio(self):
        model=Model().double();groups=classifier_lr_groups(model,1e-4,10)
        ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        old={id(p):p.detach().clone() for p in model.parameters()}
        optimizer=torch.optim.AdamW(groups,weight_decay=0)
        for p in model.parameters():p.grad=torch.ones_like(p)
        optimizer.step()
        for group in optimizer.param_groups:
            for p in group['params']:
                expected=group['lr']/(1+optimizer.defaults['eps'])
                self.assertTrue(torch.allclose(old[id(p)]-p,torch.full_like(p,expected),atol=1e-12,rtol=0))
        for g in optimizer.param_groups:g['lr']=5e-5*g['lr_scale']
        self.assertEqual([g['lr'] for g in optimizer.param_groups],[5e-5,5e-4])


if __name__=='__main__':unittest.main()
