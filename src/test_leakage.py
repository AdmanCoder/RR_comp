import unittest
import pandas as pd
from features import prepare_features,assert_split

class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.d=pd.DataFrame({'match_id':[1]*4+[2]*4,'match_date':['2024-01-01']*4+['2025-01-01']*4,
            'innings':[1]*8,'overid':[1]*8,'over_ball_num':[1,2,3,4]*2,
            'runs':[0,4,1,6,2,0,4,0],'is_wicket':[False,False,True,False]*2,
            'match_format':['T20I']*8,'knock_out':[0]*8,'venue_country':['India']*8,
            'batting_type':['RHB']*8,'bowling_type':['Pace']*8})
    def test_current_and_future_outcomes_cannot_change_current_features(self):
        _,before=prepare_features(self.d)
        for i in range(len(self.d)):
            mutated=self.d.copy();mutated.loc[i:,'runs']=99;mutated.loc[i:,'is_wicket']=True
            _,after=prepare_features(mutated)
            pd.testing.assert_frame_equal(before.iloc[:i+1],after.iloc[:i+1])
    def test_future_match_removal_preserves_past_features(self):
        _,full=prepare_features(self.d);_,past=prepare_features(self.d.iloc[:4])
        pd.testing.assert_frame_equal(full.iloc[:4],past)
    def test_post_outcome_columns_never_enter_features(self):
        d=self.d.assign(result='win',stroke='six',bowl_speed=150,win_team_id=1,player_id=123)
        _,x=prepare_features(d)
        self.assertFalse(set(['runs','is_wicket','result','stroke','bowl_speed','win_team_id','player_id'])&set(x.columns))
    def test_state_resets_between_matches(self):
        _,x=prepare_features(self.d)
        self.assertEqual(x.loc[4,'prior_wickets'],0)
        self.assertEqual(x.loc[4,'prior_recorded_bat_runs'],0)
    def test_split_rejects_shared_match(self):
        b,_=prepare_features(self.d)
        with self.assertRaises(AssertionError):assert_split(b,b.index<2,b.index>=2)
    def test_split_is_chronological(self):
        b,_=prepare_features(self.d)
        assert_split(b,b.match_id.eq(1),b.match_id.eq(2))
        with self.assertRaises(AssertionError):assert_split(b,b.match_id.eq(2),b.match_id.eq(1))

if __name__=='__main__':unittest.main()
