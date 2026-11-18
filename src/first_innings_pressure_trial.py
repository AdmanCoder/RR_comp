"""Isolate first-innings scoreboard changes on matched scenarios."""
import numpy as np
import pandas as pd
from audit import OUT
from ipl_pressure import score_states

def main():
    states = pd.DataFrame([
        dict(innings=1, balls_left=left, wickets=wickets, score=par+gap,
             first_innings_par=par, needed=np.nan, knock_out=False, gap=gap)
        for left, par in [(60, 90), (12, 162)]
        for wickets in [2, 8, 9] for gap in [-30, 0, 30]
    ])
    neutral = pd.Series(0., index=states.index)
    d = score_states(states, neutral, neutral)
    scale = .6 * (132-d.balls_left)
    ahead = (d.gap/scale).clip(0,1)
    behind = (-d.gap/scale).clip(0,1)
    d['bat_cushion'] = d.bat_pressure - 32.2*d.bat_resources*.25*ahead
    middle = ((d.balls_left-30)/30).clip(0,1) * ((120-d.balls_left)/36).clip(0,1)
    d['bowl_cushion'] = d.bowl_pressure - 32.2*d.bowl_resources*.25*behind + 3*middle*(1-behind)
    late = ((60-d.balls_left)/60).clip(0,1)
    resources = 1-(1-late)*(1-d.wickets/9)
    d['bat_with_last_wicket'] = d.bat_cushion + 32.2*(resources-d.bat_resources)*(1-.25*ahead)
    for _, g in d.groupby(['balls_left','wickets']):
        assert (np.diff(g.bat_with_last_wicket)<0).all()
        assert (np.diff(g.bowl_cushion)>0).all()
    for _, g in d.groupby(['balls_left','gap']):
        assert g.set_index('wickets').bat_with_last_wicket.loc[9] > g.set_index('wickets').bat_with_last_wicket.loc[8]
    d.to_csv(OUT/'combined_replay/first_innings_pressure_trial.csv',index=False)
    print(d[['balls_left','wickets','gap','bat_pressure','bat_cushion','bat_with_last_wicket','bowl_pressure','bowl_cushion']].round(1).to_string(index=False))

if __name__ == '__main__':
    main()
