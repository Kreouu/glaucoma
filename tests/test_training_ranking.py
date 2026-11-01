import unittest
import torch
from training_ranking import pairwise_logistic_loss


class RankingTest(unittest.TestCase):
    def test_order_shift_and_gradient_directions(self):
        x=torch.tensor([2.,1.,-1.,0.],requires_grad=True);y=torch.tensor([1.,1.,0.,0.])
        loss=pairwise_logistic_loss(x,y)
        expected=torch.stack([torch.nn.functional.softplus(x[n]-x[p]) for n in (2,3) for p in (0,1)]).mean()
        torch.testing.assert_close(loss,expected)
        torch.testing.assert_close(loss,pairwise_logistic_loss(x+5,y))
        self.assertLess(float(loss.detach()),float(pairwise_logistic_loss(-x,y).detach()))
        loss.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertTrue((x.grad[:2]<0).all());self.assertTrue((x.grad[2:]>0).all())

    def test_single_class_batch_is_differentiable_zero(self):
        for label in (0.,1.):
            x=torch.tensor([-80.,80.],requires_grad=True);y=torch.full((2,),label)
            loss=pairwise_logistic_loss(x,y);self.assertEqual(float(loss.detach()),0)
            loss.backward();torch.testing.assert_close(x.grad,torch.zeros_like(x))


if __name__=='__main__':unittest.main()
