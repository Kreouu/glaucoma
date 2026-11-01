import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from training_auc_pair import ClassPresentBatches,auc_pair_loss


class AUCPairTest(unittest.TestCase):
    def test_explicit_pairs_and_gradients(self):
        x=torch.tensor([-.5,.1,.4,-.2],requires_grad=True);y=torch.tensor([1.,0.,0.,1.]);p=.032
        loss=auc_pair_loss(x,y,p)
        rank=sum(torch.relu(1-x[i]+x[j]).square() for i in (0,3) for j in (1,2))/4
        aux=3*p*torch.nn.functional.softplus(-x[[0,3]]).mean()+(1-p)*torch.nn.functional.softplus(x[[1,2]]).mean()
        self.assertTrue(torch.allclose(loss,rank+.1*aux));loss.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue((x.grad[[0,3]]<0).all());self.assertTrue((x.grad[[1,2]]>0).all())

    def test_sampler_seed_coverage_tail_and_classes(self):
        for n in (17,19,64,51396):
            labels=[1]*max(2,round(n*.032))+[0]*(n-max(2,round(n*.032)))
            a=ClassPresentBatches(labels,16,torch.Generator().manual_seed(42));b=ClassPresentBatches(labels,16,torch.Generator().manual_seed(42));rows=list(a)
            self.assertEqual(rows,list(b));self.assertEqual(sum(map(len,rows)),n);self.assertEqual(len(rows),len(a));self.assertTrue(all(2<=len(r)<=16 and {labels[i] for i in r}=={0,1} for r in rows))
            negatives=[i for r in rows for i in r if labels[i]==0]
            self.assertEqual(len(negatives),len(set(negatives)))


if __name__=='__main__':unittest.main()
