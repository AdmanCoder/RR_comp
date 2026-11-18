import unittest
import numpy as np
import pandas as pd
from audit import normalize_name, resolve_name
from rating import pressure, recency_weight, innings_rates, assert_ready

class IntegrityTests(unittest.TestCase):
    def test_name_ambiguity_is_not_merged(self):
        ids=pd.DataFrame({'player_id':[1,2],'player_name':['A. Khan','A Khan']})
        self.assertEqual(resolve_name('A Khan',ids)['status'],'unresolved')
    def test_normalization(self):
        self.assertEqual(normalize_name(' José  Smith '),'josesmith')
    def test_fuzzy_never_automatically_joins(self):
        ids=pd.DataFrame({'player_id':[1,2],'player_name':['Ravindra Jadeja','Jos Buttler']})
        self.assertEqual(resolve_name('Ravindra Jadej',ids)['status'],'review_fuzzy')
    def test_pressure_bounds_and_damping(self):
        self.assertLessEqual(float(pressure(1,1,1,knockout=True)),100)
        self.assertLess(float(pressure(.5,1,.5,feasibility=0)),float(pressure(.5,1,.5)))
    def test_recency(self):
        self.assertAlmostEqual(float(recency_weight(730.5)),.7)
    def test_high_pressure_negative_counts_against_player(self):
        d=pd.DataFrame({'player_id':[1,1],'match_id':[1,1],'innings':[1,1],
                        'pressure':[100,20],'contribution':[-1,1]})
        self.assertLess(innings_rates(d).rate.iloc[0],0)
    def test_final_output_gate(self):
        with self.assertRaises(ValueError):assert_ready({'score_reconstruction_verified':False})

if __name__=='__main__': unittest.main()
