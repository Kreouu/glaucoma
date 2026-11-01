import unittest
import torch
from torchvision.ops import sigmoid_focal_loss
from training_focal import BinaryFocalLoss


class FocalTest(unittest.TestCase):
    def test_zero_gamma_matches_weighted_bce_and_gradients(self):
        x=torch.tensor([-4.,-.2,.5,3.],requires_grad=True);y=torch.tensor([0.,1.,0.,1.])
        actual=BinaryFocalLoss(0,3)(x,y)
        expected=torch.nn.functional.binary_cross_entropy_with_logits(x,y,pos_weight=torch.tensor(3.))
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(expected,x)[0])

    def test_matches_official_focal_with_preserved_class_cost(self):
        x=torch.tensor([-80.,-4.,-.2,.5,3.,80.],requires_grad=True);y=torch.tensor([1.,0.,1.,0.,1.,0.])
        actual=BinaryFocalLoss(2,3)(x,y)
        expected=4*sigmoid_focal_loss(x,y,alpha=.75,gamma=2,reduction='mean')
        torch.testing.assert_close(actual,expected)
        actual.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertLess(float(x.grad[0]),0);self.assertGreater(float(x.grad[-1]),0)


if __name__=='__main__':unittest.main()
