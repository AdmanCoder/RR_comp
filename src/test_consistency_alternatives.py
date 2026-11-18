import unittest
import numpy as np
from consistency_alternatives import variation_components

class AlternativeConsistencyTests(unittest.TestCase):
    def test_known_noise_is_removed_from_variance(self):
        r=variation_components([-1,1],[1,1],[1,1])
        self.assertAlmostEqual(r['observed_variance'],1.)
        self.assertAlmostEqual(r['estimated_noise'],.5)
        self.assertAlmostEqual(r['noise_adjusted_sd'],1.)
    def test_overestimated_noise_clips_without_imaginary_sd(self):
        r=variation_components([-.1,.1],[1,1],[10,10])
        self.assertEqual(r['noise_adjusted_sd'],0.)
    def test_downside_does_not_penalize_upside(self):
        r=variation_components([.1,1.],[1,1],[0,0])
        self.assertEqual(r['downside'],0.)
        self.assertGreater(r['observed_variance'],0.)
    def test_consistently_negative_has_downside(self):
        r=variation_components([-.2,-.2],[1,1],[0,0])
        self.assertAlmostEqual(r['downside'],.2)
        self.assertEqual(r['observed_variance'],0.)
    def test_single_appearance_cannot_estimate_latent_variation(self):
        r=variation_components([.2],[1],[0])
        self.assertTrue(np.isnan(r['noise_adjusted_sd']))

if __name__=='__main__':unittest.main()
