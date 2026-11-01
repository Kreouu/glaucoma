import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from training_sampling import positive_resampling


class SamplingTest(unittest.TestCase):
    def test_expected_loss_and_gradient_match_weighted_bce(self):
        y=torch.tensor([0.,0.,0.,1.,1.],dtype=torch.double)
        x=torch.tensor([-4.,.2,3.,-.5,2.],dtype=torch.double,requires_grad=True)
        sampler,scale=positive_resampling(y,torch.Generator().manual_seed(42))
        q=sampler.weights/sampler.weights.sum()
        actual=scale*(q*torch.nn.functional.binary_cross_entropy_with_logits(x,y,reduction='none')).sum()
        expected=torch.nn.functional.binary_cross_entropy_with_logits(x,y,pos_weight=torch.tensor(3.),reduction='mean')
        self.assertTrue(torch.allclose(actual,expected,atol=1e-12,rtol=0))
        a=torch.autograd.grad(actual,x,retain_graph=True)[0];b=torch.autograd.grad(expected,x)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=0))

    def test_draw_count_seed_and_positive_mass(self):
        y=[0]*49739+[1]*1657
        a,scale=positive_resampling(y,torch.Generator().manual_seed(42));b,_=positive_resampling(y,torch.Generator().manual_seed(42))
        first=list(a);self.assertEqual(first,list(b));self.assertEqual(len(first),51396)
        q=3*1657/(49739+3*1657);observed=sum(y[i] for i in first)/len(first)
        self.assertLess(abs(observed-q),.005);self.assertAlmostEqual(scale,(49739+3*1657)/51396)
        self.assertTrue(a.replacement)


if __name__=='__main__':unittest.main()
