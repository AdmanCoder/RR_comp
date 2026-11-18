"""Controlled combined-data candidate replay with frozen V1 aggregation."""
import json
import hashlib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from audit import OUT
from final_ipl_model import DEST, prepare_role, WICKETS
from ipl_pressure import KEYS
from ipl_ratings import FEATURES
from optimize_expectations import feature_sets
from pressure_ml_optimization import cat_predictions
from submission_selection import score
from rating import recency_weight

DESTINATION = OUT/'v1_candidate_comparison'


def leaderboard(d, expected, role, cutoff, scales):
    d=d.copy()
    a=.7 if role=='bat' else .8
    sign=1 if role=='bat' else -1
    d['contribution']=a*sign*(d.y_runs-expected[:,0])/scales[0]+(1-a)*sign*(expected[:,1]-d.y_wicket)/scales[1]
    d['expected_runs']=expected[:,0];d['expected_wickets']=expected[:,1]
    pid,name=('striker_id','striker_name') if role=='bat' else ('bowler_id','bowler_name')
    high=d[d[role+'_candidate'].ge(cutoff)]
    g=high.groupby([pid,name,'match_id','match_date','source_format']).agg(
        balls=('countable','sum'),runs=('y_runs','sum'),wickets=('y_wicket','sum'),
        expected_runs=('expected_runs','sum'),expected_wickets=('expected_wickets','sum'),
        adjusted_mean=('contribution','mean')).reset_index()
    dates=pd.to_datetime(g.match_date)
    g['weight']=np.sqrt(g.balls)*recency_weight((dates.max()-dates).dt.days)
    rows=[]
    for player,x in g[g.balls.ge(6)].groupby(pid):
        if len(x)<10: continue
        mean,floor,rating=score(x,ipl_multiplier=2)
        positive=int(x.adjusted_mean.gt(0).sum())
        rows.append(dict(player_id=player,player=x[name].iloc[0],matches=len(x),positive=positive,
            pressure_balls=int(x.balls.sum()),mean=mean,floor=floor,rating=rating,repeat_credit=positive**2/len(x)))
    board=pd.DataFrame(rows)
    board['score']=.8*board.rating.rank(pct=True)+.2*board.repeat_credit.rank(pct=True)
    board=board.sort_values(['score','rating','player_id'],ascending=[False,False,True]).reset_index(drop=True)
    board.insert(0,'rank',np.arange(1,len(board)+1))
    return board,g


def main():
    dest=DESTINATION;dest.mkdir(exist_ok=True)
    frozen=OUT/'submission/model_v1'
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.glob('*.csv')}
    raw=pd.read_pickle(OUT/'combined_replay/features.pkl')
    trial=pd.read_csv(OUT/'combined_replay/combined_pressure_trial.csv.gz',usecols=KEYS+['bat_candidate','bowl_candidate'])
    seed=json.loads((DEST/'warm_history_seeds.json').read_text())
    metrics=[];selection=[];tops=[];checks=[]
    for role in ['bat','bowl']:
        bat=role=='bat'
        d=prepare_role(raw,seed,role)
        if bat:d=d[d.wides.eq(0)].copy()
        d=d.merge(trial,on=KEYS,validate='one_to_one')
        d['countable']=d.wides.eq(0)&(bat|d.noballs.eq(0))
        d['y_runs']=d.runs if bat else d.charged_runs
        d['y_wicket']=(d.striker_out if bat else d.is_wicket&d.dismissal_type.isin(WICKETS)).astype(int)
        years=d.match_date.str[:4].astype(int)
        ref=d[years.eq(2022)]
        cutoff=ref.loc[ref.countable,role+'_candidate'].quantile(.75)
        training_cutoff=ref.loc[ref.countable,role+'_pressure'].quantile(.75)
        scales=ref[['y_runs','y_wicket']].std().to_numpy()
        saved=pd.read_pickle(OUT/f'combined_replay/{role}_scored_deliveries.pkl')
        saved=saved[KEYS+['expected_runs','expected_wicket']]
        parts=[];history=[]
        for year in sorted(years.unique())[1:]:
            train=d[years.lt(year)];test=d[years.eq(year)].copy()
            assert train.match_date.max()<test.match_date.min()
            assert not set(train.match_id)&set(test.match_id)
            cache=dest/f'{role}_{year}_predictions.pkl'
            if cache.exists():
                exported=pd.read_pickle(cache)
                test=test.merge(exported,on=KEYS,validate='one_to_one')
                candidates={n:test[[n+'_runs',n+'_wicket']].to_numpy() for n in ['ebm','warm_cat','pressure_cat','blended','selected_v1']}
            else:
                kw=dict(interactions=2,outer_bags=1,validation_size=0,max_rounds=300,max_bins=32,min_samples_leaf=80,n_jobs=2,random_state=17)
                r=ExplainableBoostingRegressor(**kw).fit(train[FEATURES],train.y_runs)
                w=ExplainableBoostingClassifier(**kw).fit(train[FEATURES],train.y_wicket)
                original=np.column_stack([r.predict(test[FEATURES]),w.predict_proba(test[FEATURES])[:,1]])
                core=feature_sets(role)['opponent']
                regular=cat_predictions(train,test,core,None)
                focused=cat_predictions(train,test,core+[role+'_pressure'],np.where(train[role+'_pressure'].ge(training_cutoff),3.,1.))
                selected=test[KEYS].merge(saved,on=KEYS,validate='one_to_one',how='left')[['expected_runs','expected_wicket']].to_numpy()
                candidates=dict(ebm=original,warm_cat=regular,pressure_cat=focused,blended=(original+regular+focused)/3,selected_v1=selected)
                exported=test[KEYS].copy()
                for n,pred in candidates.items():exported[[n+'_runs',n+'_wicket']]=pred
                exported.to_pickle(cache)
            names=['ebm','warm_cat','pressure_cat','blended']
            choice=['ebm','ebm'] if not history else [min(names,key=lambda n:np.mean([h[n][k] for h in history])) for k in range(2)]
            candidates['chronological_choice']=np.column_stack([candidates[n][:,k] for k,n in enumerate(choice)])
            selection.append(dict(role=role,year=int(year),runs=choice[0],wickets=choice[1],train_end=train.match_date.max()))
            truth=test[['y_runs','y_wicket']].to_numpy();high=test[role+'_candidate'].ge(cutoff).to_numpy()
            loss={}
            for n,pred in candidates.items():
                assert np.isfinite(pred).all() and ((pred[:,1]>=0)&(pred[:,1]<=1)).all()
                test[[n+'_runs',n+'_wicket']]=pred
                err=(truth-pred)**2;loss[n]=err[high].mean(axis=0)
                for subset,mask in [('all',np.ones(len(test),bool)),('high',high)]:
                    metrics.append(dict(role=role,year=int(year),model=n,subset=subset,events=int(mask.sum()),runs_mse=err[mask,0].mean(),wicket_brier=err[mask,1].mean()))
            history.append(loss);parts.append(test)
            print(role,year,'complete; choice',choice,flush=True)
        all_data=pd.concat(parts,ignore_index=True)
        for n in candidates:
            board,apps=leaderboard(all_data,all_data[[n+'_runs',n+'_wicket']].to_numpy(),role,cutoff,scales)
            board.to_csv(dest/f'{role}_{n}_rankings.csv',index=False)
            apps.to_csv(dest/f'{role}_{n}_appearances.csv',index=False)
            tops.append(board.head(10).assign(role=role,model=n))
            if n=='selected_v1':
                baseline=pd.read_csv(frozen/f'final_{role}_rankings.csv')
                assert board.player_id.tolist()==baseline.player_id.tolist()
                np.testing.assert_allclose(board[['mean','floor','rating','score']],baseline[['mean','floor','rating','score']],atol=1e-12)
                checks.append(dict(role=role,eligible=len(board),v1_exact_rank_match=True,cutoff=cutoff))
        print(role,'V1 reproduction passed',flush=True)
    pd.concat(tops).to_csv(dest/'all_top_tens.csv',index=False)
    pd.DataFrame(metrics).to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(selection).to_csv(dest/'chronological_selection.csv',index=False)
    (dest/'validation.json').write_text(json.dumps(dict(checks=checks,frozen_hashes=hashes),indent=2))
    for p,h in hashes.items():assert hashlib.sha256(open(p,'rb').read()).hexdigest()==h
    tables=[]
    for role in ['bat','bowl']:
        table=pd.concat([x for x in tops if x.role.iloc[0]==role]).pivot(index='rank',columns='model',values='player')
        lines=['| Rank | '+' | '.join(table.columns)+' |', '|---|'+'---|'*len(table.columns)]
        lines += ['| '+str(i)+' | '+' | '.join(row)+' |' for i,row in table.iterrows()]
        tables.append(role.upper()+'\n\n'+'\n'.join(lines))
    (dest/'TOP_TENS.md').write_text('\n\n'.join(tables))
    print('Comparison complete',flush=True)

if __name__=='__main__':main()
