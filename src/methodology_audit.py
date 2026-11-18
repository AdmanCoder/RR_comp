"""Read-only diagnostics of the cubed ratings; never selects weights by player names."""
from pathlib import Path
import numpy as np
import pandas as pd
from ipl_pressure import score_states
from rating import recency_weight

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/methodology_audit'
BASE = ROOT/'outputs/model_optimization/cubed_pressure/final'

def main():
    OUT.mkdir(exist_ok=True)
    data = pd.read_pickle(ROOT/'outputs/model_optimization/features.pkl')
    profiles, sensitivity = [], []
    for role, player in [('bat','striker_id'),('bowl','bowler_id')]:
        board = pd.read_csv(BASE/f'{role}_pressure_rankings.csv')
        d = data[data.match_date.str[:4].astype(int).ge(2023)].copy()
        if role == 'bat':
            d = d[d.wides.eq(0)].copy()
        dates = pd.to_datetime(d.match_date)
        d['weight'] = (d[role+'_pressure']/100)**3 * recency_weight((dates.max()-dates).dt.days)
        d['phase_audit'] = np.select([d.balls_left.gt(84),d.balls_left.gt(30)],['powerplay','middle'],default='death')
        for pid,g in d.groupby(player):
            weights=g.weight
            if weights.sum()<=0: continue
            matches=g.groupby('match_id').weight.sum().sort_values(ascending=False)
            phases=g.groupby('phase_audit').weight.sum()/weights.sum()
            row=dict(role=role,player_id=pid,events=len(g),effective_events=weights.sum()**2/(weights**2).sum(),
                effective_matches=matches.sum()**2/(matches**2).sum(),
                largest_match_share=matches.iloc[0]/matches.sum(),top_three_match_share=matches.head(3).sum()/matches.sum(),
                **{phase+'_weight_share':phases.get(phase,0) for phase in ['powerplay','middle','death']})
            profiles.append(row)
        for penalty in [0,.075,.15,.225,.30]:
            b=board.copy()
            b['audit_rating']=b['mean']*b.pressure_matches/(b.pressure_matches+8)-penalty*b.match_sd
            b['audit_rank']=b.audit_rating.rank(method='min',ascending=False).astype(int)
            b['penalty_strength']=penalty
            b['role']=role
            sensitivity.append(b)
    profiles=pd.DataFrame(profiles)
    boards=pd.concat([pd.read_csv(BASE/f'{r}_pressure_rankings.csv').assign(role=r) for r in ['bat','bowl']])
    profiles=boards.merge(profiles,on=['role','player_id'],validate='one_to_one')
    profiles['mean_rank']=profiles.groupby('role')['mean'].rank(ascending=False,method='min')
    profiles.to_csv(OUT/'evidence_profiles.csv',index=False)
    pd.concat(sensitivity).to_csv(OUT/'consistency_penalty_sensitivity.csv',index=False)
    rows=[]
    for innings in [1,2]:
        for left in [120,96,85,84,60,30,12,6,1]:
            for wickets in [0,5,9]:
                for needed in ([0] if innings==1 else sorted(set([1,left,2*left,3*left,4*left]))):
                    rows.append(dict(innings=innings,balls_left=left,wickets=wickets,score=1.5*(120-left),
                        first_innings_par=1.5*(120-left),needed=needed,knock_out=False))
    cases=pd.DataFrame(rows)
    result=score_states(cases.copy(),pd.Series(0.,index=cases.index),pd.Series(0.,index=cases.index))
    result.to_csv(OUT/'pressure_scenario_grid.csv',index=False)
    assert np.isfinite(result[['bat_pressure','bowl_pressure']]).all().all()
    assert result[['bat_pressure','bowl_pressure']].ge(0).all().all()
    assert result[['bat_pressure','bowl_pressure']].le(100).all().all()
    print(profiles.groupby('role').head(10)[['role','rank','player','effective_events','effective_matches','top_three_match_share','powerplay_weight_share','middle_weight_share','death_weight_share']].round(3).to_string(index=False))
    print('\nFirst innings, par score, no personal deficit, zero wickets:')
    print(result[(result.innings==1)&(result.wickets==0)][['balls_left','bat_pressure','bowl_pressure']].round(2).to_string(index=False))
    print('\nChase, five wickets lost:')
    print(result[(result.innings==2)&(result.wickets==5)&result.balls_left.isin([6,12])][['balls_left','needed','bat_pressure','bowl_pressure']].round(2).to_string(index=False))

if __name__=='__main__': main()
