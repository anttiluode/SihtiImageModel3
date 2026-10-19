import unittest
import numpy as np
from sihti3 import LearnedMedium, patchify, unpatchify, sieve, gaussian_fill
from data import train_set, held_out, damage


class MediumTests(unittest.TestCase):
    def setUp(self):
        self.train = train_set(80)
        self.model = LearnedMedium.fit(self.train, rank=12, codes=32, seed=2026)
        self.target = held_out(1)[0]
        self.obs, self.mask = damage(self.target, 0)

    def test_sieve_lossless(self):
        bands = sieve(self.target)
        np.testing.assert_allclose(sum(bands), self.target, atol=1e-12)

    def test_patch_overlap_has_no_seams_for_constant(self):
        v = np.ones((32, 32)) * .37
        np.testing.assert_allclose(unpatchify(patchify(v), v.shape), v, atol=1e-10)

    def test_hidden_pixels_cannot_enter_observation(self):
        obs2 = self.obs.copy()
        obs2[~self.mask] = 10
        one = self.model.run(self.obs, self.mask, steps=3)
        two = self.model.run(obs2, self.mask, steps=3)
        np.testing.assert_allclose(one["image"], two["image"], atol=1e-10)

    def test_observed_pixels_stay_unchanged(self):
        for ping in (False, True):
            result = self.model.run(self.obs, self.mask, steps=4, ping=ping)
            np.testing.assert_allclose(result["image"][self.mask], self.obs[self.mask], atol=1e-12)

    def test_coarse_feedback_changed_by_fine_structure(self):
        x = gaussian_fill(self.obs, self.mask)
        x2 = x.copy()
        x2[~self.mask] = 1-x2[~self.mask]
        delta = np.max(np.abs(self.model.global_projection(x)-self.model.global_projection(x2)))
        self.assertGreater(delta, 0)

    def test_all_arms_finite(self):
        for feedback in (False, True):
            result = self.model.run(self.obs, self.mask, steps=3, feedback=feedback, ping=True)
            self.assertTrue(np.isfinite(result["image"]).all())
            self.assertEqual(result["frames"].shape, (4, 32, 32))


if __name__ == "__main__":
    unittest.main()
