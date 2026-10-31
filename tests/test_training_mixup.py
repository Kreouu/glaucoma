import unittest
from unittest.mock import patch
import numpy as np
import torch
from training_mixup import mix_batch


class MixupTest(unittest.TestCase):
    def test_disabled_preserves_objects_and_rng(self):
        images=torch.randn(4,3,2,2);labels=torch.tensor([0.,0.,1.,0.])
        torch_state=torch.random.get_rng_state();numpy_state=np.random.get_state()
        x,y=mix_batch(images,labels,0)
        self.assertIs(x,images);self.assertIs(y,labels)
        self.assertTrue(torch.equal(torch_state,torch.random.get_rng_state()))
        after=np.random.get_state()
        self.assertEqual(numpy_state[0],after[0]);np.testing.assert_array_equal(numpy_state[1],after[1]);self.assertEqual(numpy_state[2:],after[2:])

    def test_soft_weighted_bce_matches_mixed_hard_label_losses(self):
        images=torch.tensor([1.,3.,5.,7.]).reshape(4,1,1,1);labels=torch.tensor([0.,1.,0.,0.]);order=torch.tensor([1,0,3,2]);weight=.3
        with patch('training_mixup.np.random.beta',return_value=weight),patch('training_mixup.torch.randperm',return_value=order):
            x,y=mix_batch(images,labels,.2)
        torch.testing.assert_close(x,weight*images+(1-weight)*images[order])
        self.assertAlmostEqual(float(y.sum()),float(labels.sum()))
        logits=torch.tensor([-.4,.2,1.2,-2.],requires_grad=True)
        loss=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.),reduction='none')
        actual=loss(logits,y);expected=weight*loss(logits,labels)+(1-weight)*loss(logits,labels[order])
        torch.testing.assert_close(actual,expected)
        actual.mean().backward();self.assertTrue(torch.isfinite(logits.grad).all())


if __name__=='__main__':unittest.main()
