import unittest
import pandas as pd
from extras import infer_events,preball_states,design

class ExtrasTests(unittest.TestCase):
    def setUp(self):
        self.b=pd.DataFrame({'match_id':[1]*6,'innings':[1,1,1,2,2,2],
            'overid':[1]*6,'over_ball_num':[1,2,3,1,2,3],
            'innings_ball_num':[1,2,3,1,2,3],
            'striker_id':[10,11,11,20,20,21],
            'non_striker_id':[11,10,10,21,21,20],
            'runs':[0,0,4,0,1,0],'is_wicket':[False]*6,
            'extras_type':['legbyes','wide','None','other','None','None'],
            'extras_2_runs':[0]*6,'extras1_runs':[3,2,6,5,6,6]})
    def test_codes_never_summed(self):
        e=infer_events(self.b)
        self.assertEqual(e.loc[2,'known_total'],4)
        self.assertEqual(e.loc[3,'known_total'],1)
        self.assertFalse(e.loc[3,'legal_delivery'])
    def test_parity(self):
        e=infer_events(self.b)
        self.assertEqual(e.loc[0,'group'],'bye_odd')
        self.assertEqual(e.loc[1,'group'],'wide_even')
    def test_last_event_does_not_use_next_innings(self):
        e=infer_events(self.b)
        self.assertEqual(e.loc[2,'running_parity'],-1)
    def test_no_current_or_future_outcomes_in_preball_state(self):
        before=preball_states(self.b)
        for i in range(len(self.b)):
            d=self.b.copy()
            d.loc[i:,'runs']=6;d.loc[i:,'is_wicket']=True
            d.loc[i:,'extras_type']='wide';d.loc[i:,'extras_2_runs']=4
            # Current identities are known before delivery; future identities are not.
            d.loc[i+1:,'striker_id']=999;d.loc[i+1:,'non_striker_id']=888
            after=preball_states(d)
            pd.testing.assert_frame_equal(before.iloc[:i+1],after.iloc[:i+1])
    def test_no_result_backfill(self):
        a=preball_states(self.b.assign(result='won by 1 run'))
        b=preball_states(self.b.assign(result='won by 99 runs'))
        pd.testing.assert_frame_equal(a,b)
    def test_dls_text_excluded_even_when_fields_blank(self):
        e=infer_events(self.b)
        m=pd.DataFrame({'match_id':[1],'match_date':['2023-01-01'],
             'result':['India Won by 2 Runs (DLS method)'],'revised_target':[''],
             'first_inn_revised_overs':[''],'second_inn_revised_overs':['']})
        self.assertTrue(design(e,m).empty)

if __name__=='__main__':unittest.main()
