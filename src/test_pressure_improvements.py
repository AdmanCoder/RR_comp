import unittest
import numpy as np
import pandas as pd
from pressure_improvements import improved_states
from ipl_pressure import score_states
from ipl_ratings import rank


def states(left,needed=0,innings=1,wickets=0):
    left=np.asarray(left)
    return pd.DataFrame(dict(balls_left=left,needed=needed,innings=innings,wickets=wickets,
        score=1.5*(120-left),first_innings_par=1.5*(120-left),knock_out=False,bat_personal=0.,bowl_personal=0.))

class ImprovementsTests(unittest.TestCase):
    def test_powerplay_above_middle_at_par(self):
        d=improved_states(states([120,96,85,84,60]))
        for role in ['bat','bowl']:
            p=d[role+'_pressure']
            self.assertGreater(p.iloc[0],p.iloc[-1])
            self.assertGreater(p.iloc[2],p.iloc[3])
            self.assertEqual(p.iloc[3],p.iloc[4])
    def test_regular_chase_batting_order_unchanged(self):
        d=states([6]*4,needed=[6,12,18,24],innings=2,wickets=5)
        a=improved_states(d)
        self.assertTrue((np.diff(a.bat_pressure)>0).all())
        old=score_states(d.copy(),d.bat_personal,d.bowl_personal)
        np.testing.assert_allclose(a.bat_pressure,old.bat_pressure)
    def test_bowler_last_ball_one_more_demanding_than_six(self):
        d=improved_states(states([1,1],needed=[1,6],innings=2,wickets=5))
        self.assertGreater(d.bowl_pressure.iloc[0],d.bowl_pressure.iloc[1])
    def test_low_required_rate_not_zero_bowler_urgency(self):
        d=states([6],needed=1,innings=2,wickets=5)
        old=score_states(d.copy(),d.bat_personal,d.bowl_personal)
        self.assertGreater(improved_states(d).bowl_pressure.iloc[0],old.bowl_pressure.iloc[0])
    def test_completed_chase_zero(self):
        d=improved_states(states([50,6,1],needed=0,innings=2))
        self.assertTrue(d[['bat_pressure','bowl_pressure']].eq(0).all().all())
    def test_extremes_do_not_gain_premium(self):
        d=states([100,12,6],needed=[500,56,31],innings=2)
        a=improved_states(d)
        old=score_states(d.copy(),d.bat_personal,d.bowl_personal)
        np.testing.assert_allclose(a[['bat_pressure','bowl_pressure']],old[['bat_pressure','bowl_pressure']])
    def test_evidence_shrinks_concentrated_positive_mean(self):
        rows=[]
        for m in range(8):
            for j in range(10):
                rows.append(dict(striker_id=1,striker_name='A',match_id=m,match_date='2025-01-01',bat_pressure=100 if m==0 else 30,contribution=.2,countable=True))
        d=pd.DataFrame(rows)
        a=rank(d,'bat',20,pressure_power=3)
        b=rank(d,'bat',20,pressure_power=3,evidence_adjusted=True)
        self.assertLess(b.rating.iloc[0],a.rating.iloc[0])
        self.assertLess(b.shrinkage_evidence.iloc[0],2)
        self.assertEqual(a['mean'].iloc[0],b['mean'].iloc[0])

if __name__=='__main__':unittest.main()
