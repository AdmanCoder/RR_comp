import unittest
import numpy as np
import pandas as pd
from consistency_validation import future_profiles

class ConsistencyValidationTests(unittest.TestCase):
    def sample(self, rates):
        return pd.DataFrame(dict(striker_id=[1]*len(rates),match_id=list(range(len(rates))),bat_pressure=[100]*len(rates),contribution=rates))
    def test_equal_mean_different_variation(self):
        stable=future_profiles(self.sample([.2,.2]),'bat').iloc[0]
        variable=future_profiles(self.sample([-.2,.6]),'bat').iloc[0]
        self.assertAlmostEqual(stable.future_mean,variable.future_mean)
        self.assertEqual(stable.future_sd,0)
        self.assertGreater(variable.future_sd,stable.future_sd)
        self.assertGreater(stable.future_reliability,variable.future_reliability)
        self.assertEqual(variable.future_below_expected_share,.5)
    def test_consistently_poor_not_reliable_quality(self):
        weak=future_profiles(self.sample([-.2,-.2]),'bat').iloc[0]
        self.assertEqual(weak.future_sd,0)
        self.assertLess(weak.future_reliability,0)
        self.assertEqual(weak.future_below_expected_share,1)
    def test_weighted_matches(self):
        d=self.sample([1.,-1.]);d.loc[1,'bat_pressure']=50
        result=future_profiles(d,'bat').iloc[0]
        self.assertAlmostEqual(result.future_mean,(1-.125)/(1+.125))
        self.assertAlmostEqual(result.future_below_expected_share,.125/1.125)

if __name__=='__main__': unittest.main()
