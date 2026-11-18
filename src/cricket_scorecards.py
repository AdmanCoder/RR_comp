"""Cricket outcome scorecards for current contenders."""
import numpy as np
import pandas as pd
from audit import OUT
from ipl_pressure import KEYS

def main():
    source=OUT/'combined_replay'
    dest=OUT/'submission'
    dest.mkdir(exist_ok=True)
    trial=pd.read_csv(source/'combined_pressure_trial.csv.gz')
    ref=pd.read_pickle(source/'features.pkl')
    ref=ref[ref.match_date.str.startswith('2022')].merge(
        trial[KEYS+['bat_candidate','bowl_candidate']],on=KEYS,validate='one_to_one')
    for role,pid,name in [('bat','striker_id','striker_name'),('bowl','bowler_id','bowler_name')]:
        d=pd.read_pickle(source/f'{role}_scored_deliveries.pkl').merge(
            trial[KEYS+[role+'_candidate']],on=KEYS,validate='one_to_one')
        legal=ref.wides.eq(0)&(True if role=='bat' else ref.noballs.eq(0))
        cutoff=ref.loc[legal,role+'_candidate'].quantile(.75)
        high=d[d[role+'_candidate'].ge(cutoff)].copy()
        high['legal']=high.wides.eq(0)&high.noballs.eq(0)
        high['scoring_balls']=high.wides.eq(0) if role=='bat' else high.legal
        g=high.groupby([pid,name,'match_id','match_date','source_format']).agg(
            runs=('y_runs','sum'),balls=('scoring_balls','sum'),wickets=('y_wicket','sum'),
            expected_runs=('expected_runs','sum'),expected_wickets=('expected_wicket','sum'),
            adjusted_mean=('contribution','mean')).reset_index()
        g['runs_above_expectation']=(g.runs-g.expected_runs)*(1 if role=='bat' else -1)
        g['wickets_above_expectation']=(g.expected_wickets-g.wickets)*(1 if role=='bat' else -1)
        g.to_csv(dest/f'{role}_pressure_appearances.csv',index=False)
        rows=[]
        for player,x in g.groupby(pid):
            substantial=x[x.balls.ge(6)]
            rows.append(dict(player_id=player,player=x[name].iloc[0],pressure_runs=x.runs.sum(),
                pressure_balls=x.balls.sum(),pressure_wickets=x.wickets.sum(),
                pressure_rate=(100 if role=='bat' else 6)*x.runs.sum()/max(1,x.balls.sum()),
                pressure_matches=len(x),six_ball_appearances=len(substantial),
                positive_six_ball_appearances=int(substantial.adjusted_mean.gt(0).sum()),
                median_runs_per_six_ball_appearance=substantial.runs.median(),
                median_balls_per_six_ball_appearance=substantial.balls.median(),
                runs_above_expectation=x.runs_above_expectation.sum(),
                wickets_above_expectation=x.wickets_above_expectation.sum()))
        summary=pd.DataFrame(rows)
        board=pd.read_csv(source/f'{role}_pressure_trial_rankings.csv')
        summary=board[['rank','player_id']].merge(summary,on='player_id')
        summary.to_csv(dest/f'{role}_cricket_scorecards.csv',index=False)
        names='Cummins|Vijay|Shahrukh|Pooran|Rinku|Suryakumar|Shreyas|Klaasen|Naman|Ashutosh|Dube|Sudharsan'
        show=summary[summary.player.str.contains(names)] if role=='bat' else summary.head(12)
        print(role)
        print(show[['rank','player','pressure_runs','pressure_balls','pressure_wickets','pressure_rate',
                    'six_ball_appearances','positive_six_ball_appearances','runs_above_expectation']].round(1).to_string(index=False))

if __name__=='__main__':
    main()
