"""Match-cluster sensitivity, conditional on fixed models, pressure and eligibility.
Bootstrap inclusion frequencies are stability diagnostics, not posterior probabilities.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from rating import recency_weight

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/methodology_audit'

def main():
    results=[]
    for role,player in [('bat','striker_id'),('bowl','bowler_id')]:
        d=pd.read_pickle(OUT/f'replay/{role}_scored_deliveries.pkl')
        board=pd.read_csv(OUT/f'replay/{role}_pressure_rankings.csv')
        original=pd.read_csv(ROOT/f'outputs/model_optimization/cubed_pressure/final/{role}_pressure_rankings.csv')
        assert board.player_id.tolist()==original.player_id.tolist(), 'Replay ranking changed; investigate before comparing.'
        assert np.allclose(board.rating,original.rating,atol=1e-8)
        dates=pd.to_datetime(d.match_date)
        d['w']=(d[role+'_pressure']/100)**3*recency_weight((dates.max()-dates).dt.days)
        d['total']=d.w*d.contribution
        g=d.groupby([player,'match_id']).agg(w=('w','sum'),total=('total','sum')).reset_index()
        g=g[g.w.gt(0)]
        g['second']=g.total**2/g.w
        pids=board.player_id.tolist()
        mids=d.match_id.unique()
        matrices=[g.pivot(index=player,columns='match_id',values=v).reindex(index=pids,columns=mids).fillna(0).to_numpy() for v in ['w','total','second']]
        weight,total,second=matrices
        shrink=(board.pressure_matches/(board.pressure_matches+8)).to_numpy()
        def scores(w,t,s):
            mean=np.divide(t,w,out=np.zeros_like(t),where=w>0)
            variance=np.maximum(0,np.divide(s,w,out=np.zeros_like(s),where=w>0)-mean**2)
            return mean*shrink[:,None]-.15*np.sqrt(variance)
        # Resample entire matches jointly across players, keeping comparisons paired.
        rng=np.random.default_rng(1703)
        counts=rng.multinomial(len(mids),np.full(len(mids),1/len(mids)),size=2000).T
        sampled=scores(weight@counts,total@counts,second@counts)
        ranks=np.argsort(np.argsort(-sampled,axis=0),axis=0)+1
        deletion=scores(weight.sum(axis=1)[:,None]-weight,total.sum(axis=1)[:,None]-total,second.sum(axis=1)[:,None]-second)
        deletion_ranks=np.argsort(np.argsort(-deletion,axis=0),axis=0)+1
        b=board.copy()
        b['role']=role
        b['bootstrap_rank_p10']=np.quantile(ranks,.1,axis=1)
        b['bootstrap_rank_p90']=np.quantile(ranks,.9,axis=1)
        b['bootstrap_top10_frequency']=(ranks<=10).mean(axis=1)
        b['best_rank_deleting_one_match']=deletion_ranks.min(axis=1)
        b['worst_rank_deleting_one_match']=deletion_ranks.max(axis=1)
        b['largest_downward_change_deleting_one_match']=deletion_ranks.max(axis=1)-b['rank']
        results.append(b)
    result=pd.concat(results)
    result.to_csv(OUT/'match_cluster_rank_stability.csv',index=False)
    print(result[(result['rank']<=10)|result.player.str.contains('Arshdeep|Prasidh')][['role','rank','player','bootstrap_rank_p10','bootstrap_rank_p90','bootstrap_top10_frequency','best_rank_deleting_one_match','worst_rank_deleting_one_match']].round(3).to_string(index=False))

if __name__=='__main__': main()
