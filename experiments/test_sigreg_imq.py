"""CPU checks for the IMQ ablation; no pretrained assets or training runs needed.

Run: python experiments/test_sigreg_imq.py
SciPy is used only as a finite-dimensional reference in these checks.
"""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import hyp1f1
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "imgnet"))
from methods.sigreg import IMQGaussianField, SlicedGaussianField
from methods.tbsm import RepresentationScatteringField


class IMQFieldTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_matches_negative_potential_gradient(self):
        # Multiple tokens catch accidental use of channels instead of full d.
        h = torch.randn(3, 4, 7, dtype=torch.float64, requires_grad=True)
        r, s = torch.randn(2, 3, 4, 7, dtype=torch.float64)
        ell, scale, lam = .7, 3.5, .6
        d = h[0].numel()

        def kernel(reference):
            return (1 + (reference - h).flatten(1).square().sum(1) / (d * ell**2)).rsqrt()

        potential = scale * (-kernel(r) + lam * kernel(s)).sum()
        expected = -torch.autograd.grad(potential, h)[0]
        actual = IMQGaussianField(ell, scale)(h, r, s, lam)
        torch.testing.assert_close(actual, expected.float(), rtol=2e-6, atol=1e-8)
        self.assertFalse(actual.requires_grad)
        self.assertIsNone(h.grad)

    def test_zero_distance_cancellation_and_lambda(self):
        h, r = torch.randn(2, 3, 4, 7)
        field = IMQGaussianField(scale=2)
        torch.testing.assert_close(field(h, r, r), torch.zeros_like(h), rtol=0, atol=0)
        torch.testing.assert_close(field(h, h, h), torch.zeros_like(h), rtol=0, atol=0)
        attraction = field(h, r, h, lambda_weight=0)
        repulsion = field(h, h, r)
        torch.testing.assert_close(attraction, -repulsion)
        self.assertTrue(((attraction * (r - h)).flatten(1).sum(1) > 0).all())
        torch.testing.assert_close(field(h, r, r, .25), .75 * attraction)

    def test_bf16_autocast_rng_and_replay(self):
        h, r, s = torch.randn(3, 2, 4, 7).bfloat16()
        field = IMQGaussianField(scale=43052.067763388004)
        before = torch.get_rng_state().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = field(h, r, s)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, field(h.float(), r.float(), s.float()), rtol=0, atol=0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        restored = IMQGaussianField(scale=field.scale)
        restored.load_state_dict(copy.deepcopy(field.state_dict()))
        torch.testing.assert_close(actual, restored(h, r, s), rtol=0, atol=0)

    def test_high_dimensional_exact_direction_average(self):
        # Compare against the finite-d sphere integral, not another IMQ formula.
        # This is a pointwise approximation check, not a uniform error bound.
        d = 256 * 768
        q = np.array([.1, .5, 1., 2., 4.])
        h = torch.zeros(len(q), 256, 768)
        r = h.clone()
        r[:, 0, 0] = torch.from_numpy(np.sqrt(d * q)).float()
        actual = IMQGaussianField()(h, r, h)[:, 0, 0].double().numpy()
        radius = r[:, 0, 0].double().numpy()
        exact = radius / d * hyp1f1(1.5, d / 2 + 1, -radius**2 / 2)
        np.testing.assert_allclose(actual, exact, rtol=4e-6, atol=0)

    def test_tbsm_dispatch_target_mixing_and_gradients(self):
        class Representor(torch.nn.Module):
            feat_dim = 8

            def extract_featmap(self, x):
                return x

        field = RepresentationScatteringField(
            Representor(), lambda_weight=.8, rho=.9, num_classes=3,
            tracker_config=dict(hidden_dim=16, time_dim=8, depth=1, num_heads=2),
            field_config=dict(type="sigreg_imq", bandwidth=.7, scale=2),
        )
        self.assertIsInstance(field.sigreg, IMQGaussianField)
        h, r, s, tracked = torch.randn(4, 2, 4, 8)
        target, label = field._frozen_target(r, h, s, tracked)
        expected = field.sigreg(h, r, s, lambda_weight=.8)
        torch.testing.assert_close(label, expected)
        torch.testing.assert_close(target, h + .1 * expected + .9 * tracked)
        self.assertFalse(target.requires_grad)

        h = torch.randn(2, 8, 2, 2, requires_grad=True)
        r, s = torch.randn(2, 2, 8, 2, 2)
        g_loss, t_loss, _ = field.regression_losses(
            r, h, s, torch.tensor([0, 1]), torch.tensor([1., .5])
        )
        g_loss.backward()
        self.assertIsNotNone(h.grad)
        self.assertTrue(torch.isfinite(h.grad).all() and h.grad.norm() > 0)
        self.assertTrue(all(p.grad is None for p in field.tracker.parameters()))
        t_loss.backward()
        grads = [p.grad for p in field.tracker.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.square().sum().item() for g in grads), 0)

    def test_ablation_config_changes_only_field_and_output(self):
        base = ROOT / "experiments/configs/imagenet_sit_l_siglip256"
        sampled = yaml.safe_load((base / "sigreg.yaml").read_text())
        averaged = yaml.safe_load((base / "sigreg_imq.yaml").read_text())
        self.assertNotEqual(sampled["experiment"].pop("save_dir"), averaged["experiment"].pop("save_dir"))
        a = sampled["method"]["params"].pop("field")
        b = averaged["method"]["params"].pop("field")
        self.assertEqual(sampled, averaged)
        self.assertEqual(b, dict(type="sigreg_imq", bandwidth=a["bandwidth"], scale=a["scale"]))
        # Old field dispatch/state is retained for existing sampled checkpoints.
        dummy = torch.nn.Module()
        dummy.feat_dim = 8
        original = RepresentationScatteringField(dummy, 1., 0., 3, field_config=a)
        self.assertIsInstance(original.sigreg, SlicedGaussianField)
        self.assertIn("sigreg.draw_index", original.state_dict())

    def test_invalid_config(self):
        for v in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                IMQGaussianField(bandwidth=v)
            with self.assertRaises(ValueError):
                IMQGaussianField(scale=v)
        with self.assertRaises(TypeError):
            IMQGaussianField(directions=2048)
        with self.assertRaises(ValueError):
            IMQGaussianField()(torch.zeros(2, 3), torch.zeros(2, 4), torch.zeros(2, 3))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
