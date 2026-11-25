import unittest
import torch
from research_audit.ordinal_iqa import OrdinalIQA, ordinal_loss
from research_audit.compact_iqa import CompactIQA


class OrdinalIQATest(unittest.TestCase):
    def test_second_backbone_score_contract(self):
        model=CompactIQA(backbone='resnet18').eval()
        scores=model(torch.rand(2,3,64,64))
        self.assertEqual(tuple(scores.shape),(2,1))
        self.assertTrue(((scores>=0)&(scores<=4)).all())
        scores.sum().backward()
        self.assertTrue(torch.isfinite(model.head[-1].weight.grad).all())

    def test_cdf_loss_has_correct_extreme_costs(self):
        p=torch.zeros(2,21); p[:,0]=1
        self.assertEqual(float(ordinal_loss(p,torch.zeros(2))),0.)
        self.assertEqual(float(ordinal_loss(p,torch.full((2,),4.))),1.)

    def test_interpolated_mos_target_and_gradient(self):
        p=torch.zeros(1,21); p[0,10:12]=.5
        self.assertLess(float(ordinal_loss(p,torch.tensor([2.1]))),1e-10)
        logits=torch.randn(3,21,requires_grad=True)
        loss=ordinal_loss(logits.softmax(-1),torch.tensor([0.,2.1,4.]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0.)

    def test_expectation_bounds_and_backpropagation(self):
        model=OrdinalIQA().eval()
        scores=model(torch.rand(2,3,64,64))
        self.assertEqual(tuple(scores.shape),(2,1))
        self.assertTrue(((scores>=0)&(scores<=4)).all())
        scores.sum().backward()
        self.assertTrue(torch.isfinite(model.head[-1].weight.grad).all())


if __name__=='__main__': unittest.main()
