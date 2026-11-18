"""Sensitivity to blending actual pressure performance with contextual residuals."""
import numpy as np
import pandas as pd
from audit import OUT
from final_ipl_model import WICKETS
from ipl_pressure import KEYS
from rating import recency_weight
from submission_selection import score

def main():
    dest=OUT/'absolute_adjusted_trial'
    dest.mkdir(exist_ok=True)
    trial=pd.read_csv(OUT/'combined_replay/combined_pressure_trial.csv.gz',usecols=KEYS+['bat_candidate','bowl_candidate'])
    raw=pd.read_pickle(OUT/'combined_replay/features.pkl')
    reference=raw[raw.match_date.str.startswith('2022')].merge(trial,on=KEYS,validate='one_to_one')
    for role,model,pid in [('bat','ebm','striker_id'),('bowl','warm_cat','bowler_id')]:
        bat=role=='bat'
        ref=reference[reference.wides.eq(0)] if bat else reference
        yr=ref.runs if bat else ref.runs+ref.wides+ref.noballs
        yw=(ref.is_wicket&(ref.dismissed_player_id.eq(ref.striker_id) if bat else ref.dismissal_type.isin(WICKETS))).astype(int)
        a,sign=(.7,1) if bat else (.8,-1)
        cutoff=reference.loc[reference.wides.eq(0)&(bat|reference.noballs.eq(0)),role+'_candidate'].quantile(.75)
        d=pd.read_pickle(OUT/f'combined_replay/{role}_scored_deliveries.pkl').merge(trial,on=KEYS,validate='one_to_one')
        d=d[d[role+'_candidate'].ge(cutoff)].copy()
        # Fixed 2022 centering keeps absolute contribution on the same outcome scales.
        d['absolute']=sign*(a*(d.y_runs-yr.mean())/yr.std()-(1-a)*(d.y_wicket-yw.mean())/yw.std())
        absolute=d.groupby([pid,'match_id']).absolute.mean()
        apps=pd.read_csv(OUT/f'v1_candidate_comparison/{role}_{model}_appearances.csv').join(absolute,on=[pid,'match_id'])
        base=pd.read_csv(OUT/f'v1_candidate_comparison/{role}_{model}_rankings.csv')
        apps=apps[apps.balls.ge(6)&apps[pid].isin(base.player_id)]
        for share in [0,.25,.5]:
            current=apps.copy()
            current['adjusted_mean']=(1-share)*current.adjusted_mean+share*current.absolute
            rows=[]
            for player,g in current.groupby(pid):
                mean,floor,rating=score(g,ipl_multiplier=2)
                positive=int(g.adjusted_mean.gt(0).sum())
                rows.append(dict(player_id=player,mean=mean,floor=floor,rating=rating,matches=len(g),
                                 positive=positive,repeat_credit=positive**2/len(g)))
            board=pd.DataFrame(rows).merge(base[['player_id','player','rank']].rename(columns={'rank':'old_rank'}),on='player_id')
            board['score']=.8*board.rating.rank(pct=True)+.2*board.repeat_credit.rank(pct=True)
            board=board.sort_values(['score','rating','player_id'],ascending=[False,False,True])
            board.insert(0,'rank',range(1,len(board)+1))
            if share==0:
                assert board.player_id.tolist()==base.player_id.tolist()
                np.testing.assert_allclose(board.score,base.score)
            board.to_csv(dest/f'{role}_actual_share{share:g}.csv',index=False)
            print(role,share,board.head(10)[['rank','player','matches','positive']].to_string(index=False),flush=True)

if __name__=='__main__':main()
