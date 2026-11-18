import unittest
import numpy as np
import pandas as pd
from optimize_expectations import build_features,feature_sets

class ExpectationFeatureTests(unittest.TestCase):
    def setUp(self):
        self.b=pd.DataFrame(dict(match_id=np.repeat([1,2,3],6),innings=1,
            match_date=np.repeat(['2024-01-01','2024-01-01','2025-01-01'],6),
            wides=0,noballs=0,balls_left=np.tile(np.arange(120,114,-1),3),
            needed=np.nan,score=np.tile([0,1,3,3,7,8],3),wickets=0,
            knock_out=0,first_innings_par=np.tile(np.arange(6)*1.5,3),
            striker_id=1,bowler_id=2,venue_id=3,runs=np.tile([1,2,0,4,1,0],3),
            total_runs=np.tile([1,2,0,4,1,0],3),is_wicket=False,
            dismissed_player_id=np.nan,batting_type='RHB',bowling_type='Pace',pace_or_spin='Pace'))
    def test_current_and_future_events(self):
        original=build_features(self.b)
        columns=feature_sets('bat')['predictive']
        for i in [0,3,6,9,12,17]:
            changed=self.b.copy()
            changed.loc[i:,['runs','total_runs']]=99
            changed.loc[i:,'is_wicket']=True
            changed.loc[i:,'dismissed_player_id']=1
            result=build_features(changed)
            pd.testing.assert_frame_equal(original.loc[:i,columns],result.loc[:i,columns])
    def test_same_date_history_excluded(self):
        a=build_features(self.b);b=self.b.copy();b.loc[:5,'runs']=99
        c=build_features(b)
        history=[x for x in a if '_past_' in x]
        pd.testing.assert_frame_equal(a.loc[6:11,history],c.loc[6:11,history])
        self.assertNotEqual(a.loc[12,'striker_id_past_runs'],c.loc[12,'striker_id_past_runs'])
    def test_free_hit_uses_prior_events_and_survives_wide(self):
        b=self.b.copy();b.loc[0,'noballs']=1;b.loc[1,'wides']=1
        a=build_features(b)
        self.assertEqual(a.free_hit.iloc[:4].tolist(),[0,1,1,0])
        self.assertEqual(a.free_hit.iloc[6],0)
    def test_no_outcomes_or_identity_in_allowlists(self):
        forbidden={'runs','total_runs','is_wicket','stroke','bowl_speed','striker_id','bowler_id','batting_order','result'}
        for role in ['bat','bowl']:
            for cols in feature_sets(role).values():self.assertFalse(forbidden&set(cols))

if __name__=='__main__':unittest.main()
