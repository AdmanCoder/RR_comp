"""Continuous pressure-weighted total contribution across all substantial appearances."""
import numpy as np
import pandas as pd
from audit import OUT
from ipl_pressure import KEYS
from rating import recency_weight
from submission_selection import score

def main():
    dest=OUT/'reachability_trial'
    dest.mkdir(exist_ok=True)
    trial=pd.read_csv(OUT/'combined_replay/combined_pressure_trial.csv.gz',usecols=KEYS+['bat_candidate','bowl_candidate'])
    ref=pd.read_pickle(OUT/'combined_replay/features.pkl')
    ref=ref[ref.match_date.str.startswith('2022')]
    from final_ipl_model import WICKETS
    for role,model,pid,name in [('bat','ebm','striker_id','striker_name'),('bowl','warm_cat','bowler_id','bowler_name')]:
        d=pd.read_pickle(OUT/f'combined_replay/{role}_scored_deliveries.pkl').merge(trial,on=KEYS,validate='one_to_one')
        # Apply the existing reachability factor to the whole pressure score.
        # This experimental variant removes residual resource pressure in unreachable chases.
        d[role+'_candidate'] *= np.where(d.innings.eq(2), d.chase_extreme_factor, 1.)
        predictions=pd.concat(pd.read_pickle(OUT/f'v1_candidate_comparison/{role}_{y}_predictions.pkl') for y in range(2023,2027))
        d=d.merge(predictions[KEYS+[model+'_runs',model+'_wicket']],on=KEYS,validate='one_to_one')
        reference=ref[ref.wides.eq(0)] if role=='bat' else ref
        runs=reference.runs if role=='bat' else reference.runs+reference.wides+reference.noballs
        wickets=reference.is_wicket&(reference.dismissed_player_id.eq(reference.striker_id) if role=='bat' else reference.dismissal_type.isin(WICKETS))
        a,sign=(.7,1) if role=='bat' else (.8,-1)
        residual=sign*(a*(d.y_runs-d[model+'_runs'])/runs.std()+(1-a)*(d[model+'_wicket']-d.y_wicket)/wickets.astype(int).std())
        for power in [2,3]:
            weight=(d[role+'_candidate']/100)**power
            temp=d.assign(value=residual*weight/12,exposure=weight*d.countable)
            apps=temp.groupby([pid,name,'match_id','match_date','source_format']).agg(
                balls=('countable','sum'),adjusted_mean=('value','sum'),exposure=('exposure','sum')).reset_index()
            dates=pd.to_datetime(apps.match_date)
            apps['weight']=recency_weight((dates.max()-dates).dt.days)
            apps=apps[apps.balls.ge(6)]
            rows=[]
            for player,g in apps.groupby(pid):
                if len(g)<10:continue
                mean,floor,rating=score(g,ipl_multiplier=2)
                rows.append(dict(player_id=player,player=g[name].iloc[0],matches=len(g),
                                 positive=int(g.adjusted_mean.gt(0).sum()),mean=mean,floor=floor,
                                 rating=rating,pressure_equivalent_balls=g.exposure.sum()))
            board=pd.DataFrame(rows).sort_values(['rating','player_id'],ascending=[False,True])
            board.insert(0,'rank',range(1,len(board)+1))
            board.to_csv(dest/f'{role}_power{power}.csv',index=False)
            apps.to_csv(dest/f'{role}_power{power}_matches.csv',index=False)
            print(role,power,board.head(10)[['rank','player','matches','positive']].to_string(index=False),flush=True)

if __name__=='__main__':main()
