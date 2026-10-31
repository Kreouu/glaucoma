import math,unittest
import torch
from knn_inference import weighted_knn_logit


class KNNVote(unittest.TestCase):
    def test_top_two_weighted_vote_matches_closed_form(self):
        score=torch.tensor([1.,0.,-1.]);labels=torch.tensor([1.,0.,0.])
        actual=weighted_knn_logit(score,labels,2,1.,1/3,.001,3).sigmoid().item()
        p=.999*math.e/(math.e+1)+.001/3
        expected=3*p/(1+2*p)
        self.assertAlmostEqual(actual,expected,places=6)

    def test_single_class_neighborhood_stays_finite(self):
        for value in (0.,1.):
            result=weighted_knn_logit(torch.tensor([1.,.9]),torch.full((2,),value),2,.07,.03,.001,3)
            self.assertTrue(torch.isfinite(result))
            self.assertGreater(result.sigmoid().item(),0)
            self.assertLess(result.sigmoid().item(),1)


if __name__=='__main__':unittest.main()
