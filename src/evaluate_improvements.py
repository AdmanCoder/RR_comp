"""Isolated pressure/evidence experiments using frozen chronological ML predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ipl_ratings import rank
from pressure_improvements import improved_states
from rating import recency_weight

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/methodology_improvements'

def stability(d,board,role,cutoff,evidence):
    pid='striker_id' if role=='bat' else 'bowler_id'
    dates=pd.to_datetime(d.match_date)
    d=d.copy()
    d['w']=(d[role+'_pressure']/100)**3*recency_weight((dates.max()-dates).dt.days)
    d['t']=d.w*d.contribution
    d['h']=d[role+'_pressure'].ge(cutoff)&d.countable
    g=d.groupby([pid,'match_id']).agg(w=('w','sum'),t=('t','sum'),h=('h','sum')).reset_index()
    g=g[g.w>0]
    g['s']=g.t**2/g.w
    g['w2']=g.w**2
    g['game']=g.h.gt(0).astype(int)
    matrices=[g.pivot(index=pid,columns='match_id',values=v).reindex(index=board.player_id).fillna(0).to_numpy() for v in ['w','t','s','w2','game']]
    # Fixed eligible candidate set; recompute evidence after each match deletion.
    w,t,s,w2,games=[m.sum(axis=1)[:,None]-m for m in matrices]
    mu=np.divide(t,w,out=np.zeros_like(t),where=w>0)
    sd=np.sqrt(np.maximum(0,np.divide(s,w,out=np.zeros_like(s),where=w>0)-mu**2))
    eff=np.divide(w*w,w2,out=np.zeros_like(w),where=w2>0)
    n=np.minimum(games,eff) if evidence else games
    score=mu*n/(n+8)-.15*sd
    ranks=np.argsort(np.argsort(-score,axis=0),axis=0)+1
    out=board[['player_id','player']].copy()
    out['rank']=np.arange(1,len(out)+1)
    out['worst_deletion_rank']=ranks.max(axis=1)
    out['best_deletion_rank']=ranks.min(axis=1)
    return out

def main():
    OUT.mkdir(exist_ok=True)
    raw=pd.read_pickle(ROOT/'outputs/model_optimization/features.pkl')
    ref=raw[raw.match_date.str.startswith('2022')].copy()
    summaries=[]; future=[]; sensitivities=[]; future_details=[]
    for role in ['bat','bowl']:
        d=pd.read_pickle(ROOT/f'outputs/methodology_audit/replay/{role}_scored_deliveries.pkl')
        pc=role+'_pressure'
        legal=ref.wides.eq(0) if role=='bat' else ref.wides.eq(0)&ref.noballs.eq(0)
        old_cut=ref.loc[legal,pc].quantile(.75)
        updated=improved_states(d)
        new_cut=improved_states(ref).loc[legal,pc].quantile(.75)
        variants=[('current',d,old_cut,False),('pressure_only',updated,new_cut,False),('evidence_only',d,old_cut,True),('combined',updated,new_cut,True)]
        current=rank(d,role,old_cut,pressure_power=3)
        for label,sample,cutoff,evidence in variants:
            b=rank(sample,role,cutoff,pressure_power=3,evidence_adjusted=evidence)
            b.insert(0,'rank',np.arange(1,len(b)+1))
            b.to_csv(OUT/f'{role}_{label}_rankings.csv',index=False)
            deletion=stability(sample,b,role,cutoff,evidence)
            deletion.to_csv(OUT/f'{role}_{label}_deletion.csv',index=False)
            summaries.append(dict(role=role,variant=label,cutoff=cutoff,eligible=len(b),top10_overlap=len(set(b.head(10).player_id)&set(current.head(10).player_id)),median_top10_worst_drop=float((deletion.worst_deletion_rank-deletion['rank']).head(10).median())))
            for year in [2024,2025,2026]:
                train=sample[sample.match_date.str[:4].astype(int)<year]
                # Frozen original pressure target for all methods; no candidate-specific target.
                test=d[d.match_date.str.startswith(str(year))].copy()
                pid='striker_id' if role=='bat' else 'bowler_id'
                test['w']=(test[pc]/100)**3
                test['t']=test.w*test.contribution
                target=test.groupby(pid).agg(w=('w','sum'),t=('t','sum'),n=('w','size'))
                target['future_mean']=target.t/target.w
                target=target[(target.n>=30)&(target.w>0)]
                past=rank(train,role,cutoff,pressure_power=3,evidence_adjusted=evidence)
                joined=past.merge(target,on=None,left_on='player_id',right_index=True)
                future_details.append(joined[['player_id','rating','future_mean']].assign(role=role,variant=label,year=year))
                future.append(dict(role=role,variant=label,year=year,players=len(joined),spearman=joined.rating.corr(joined.future_mean,method='spearman'),top10_future_mean=joined.head(10).future_mean.mean()))
        for points in [3.,6.,9.]:
            sample=improved_states(d,points)
            cutoff=improved_states(ref,points).loc[legal,pc].quantile(.75)
            for penalty in [0,.075,.15]:
                b=rank(sample,role,cutoff,penalty=penalty,pressure_power=3,evidence_adjusted=True)
                b['rank']=np.arange(1,len(b)+1)
                sensitivities.append(b.assign(role=role,powerplay_points=points,penalty=penalty))
    pd.DataFrame(summaries).to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(future).to_csv(OUT/'later_season_checks.csv',index=False)
    details=pd.concat(future_details)
    fair=[]
    for (role,year),g in details.groupby(['role','year']):
        common=set.intersection(*(set(v.player_id) for _,v in g.groupby('variant')))
        for label,v in g.groupby('variant'):
            v=v[v.player_id.isin(common)].sort_values('rating',ascending=False)
            fair.append(dict(role=role,year=year,variant=label,common_players=len(v),
                spearman=v.rating.corr(v.future_mean,method='spearman'),
                top10_future_mean=v.head(10).future_mean.mean() if len(v)>=10 else np.nan))
    pd.DataFrame(fair).to_csv(OUT/'common_cohort_later_season_checks.csv',index=False)
    details.to_csv(OUT/'later_season_player_details.csv',index=False)
    pd.concat(sensitivities).to_csv(OUT/'weight_sensitivity.csv',index=False)
    grid=pd.read_csv(ROOT/'outputs/methodology_audit/pressure_scenario_grid.csv')
    revised=improved_states(grid)
    for role in ['bat','bowl']: revised[role+'_old_pressure']=grid[role+'_pressure']
    revised.to_csv(OUT/'scenario_comparison.csv',index=False)
    (OUT/'manifest.json').write_text(json.dumps(dict(status='Development candidates, not promoted',ml='Frozen earlier-season predictions; pressure-dependent ML components still use original pressure',weight_power=3,shrinkage_prior_matches=8,consistency_penalty=.15,eligibility='60 high-pressure countable events and 8 matches; threshold from 2022 only',validation='2024–2026 reused development periods. Fixed original-pressure future mean target. No untouched holdout.',deletion='Fixed candidate eligibility; evidence recomputed after deletion.'),indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(future).round(4).to_string(index=False))

if __name__=='__main__':main()
