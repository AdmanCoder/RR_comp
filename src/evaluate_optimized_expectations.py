"""Select on 2024/25, report reused 2026 separately, with match-cluster intervals."""
import json
import numpy as np
import pandas as pd
from optimize_expectations import DEST,feature_sets,fit_predict,metrics,WICKETS


def main():
    b=pd.read_pickle(DEST/'features.pkl');v=pd.read_csv(DEST/'validation_metrics.csv')
    ab=pd.read_csv(DEST/'feature_ablation.csv').rename(columns={'variant':'model'})
    v=pd.concat([v,ab,pd.read_csv(DEST/'warm_history_metrics.csv')],ignore_index=True)
    results=[];selection={};all_predictions=[];intervals=[]
    rng=np.random.default_rng(17)
    for role in ['bat','bowl']:
        d=b[b.wides.eq(0)].copy() if role=='bat' else b.copy()
        d['y_runs']=d.runs if role=='bat' else d.runs+d.wides+d.noballs
        d['y_wicket']=(d.is_wicket&(d.dismissed_player_id.eq(d.striker_id) if role=='bat' else d.dismissal_type.isin(WICKETS))).astype(int)
        year=pd.to_datetime(d.match_date).dt.year;train=d[year.lt(2026)];test=d[year.eq(2026)]
        assert pd.to_datetime(train.match_date).max()<pd.to_datetime(test.match_date).min()
        part=v[v.role.eq(role)&~v.model.str.startswith('predictive')&~v.model.isin(['constant','bucket_baseline'])]
        mean=part.groupby('model')[['mse','brier']].mean()
        chosen={'runs':mean.mse.idxmin(),'wickets':mean.brier.idxmin()};selection[role]=chosen
        cache={}
        def predict(name):
            if name in cache:return cache[name]
            if name=='warm_opponent_cat':
                from warm_history_experiment import apply_warm
                seed=json.loads((DEST/'warm_history_seeds.json').read_text())
                value=fit_predict(apply_warm(train,seed),apply_warm(test,seed),feature_sets(role)['opponent'],'cat')
                cache[name]=value;return value
            if name.startswith('drop_') or name=='run_distribution':
                features=feature_sets(role)['opponent']
                drops={'drop_knockout':['knock_out'],
                       'drop_recent':[f for f in features if f.startswith('recent_')],
                       'drop_styles':['batting_type','bowling_type','pace_or_spin'],
                       'drop_history':[f for f in features if '_past_' in f]}
                if name=='run_distribution':
                    from catboost import CatBoostClassifier
                    from optimize_expectations import prepare
                    x,z,cats=prepare(train,test,features,'cat')
                    model=CatBoostClassifier(iterations=350,depth=5,learning_rate=.04,l2_leaf_reg=15,thread_count=2,random_seed=17,verbose=False,allow_writing_files=False,cat_features=cats,loss_function='MultiClass')
                    model.fit(x,train.y_runs.astype(int))
                    value=(model.predict_proba(z)@model.classes_.astype(float),np.full(len(test),np.nan))
                else:
                    value=fit_predict(train,test,[f for f in features if f not in drops[name]],'cat')
                cache[name]=value;return value
            fs,family=name.rsplit('_',1)
            if family=='ensemble':
                a=predict(fs+'_cat');c=predict(fs+'_xgb');value=((a[0]+c[0])/2,(a[1]+c[1])/2)
            else:value=fit_predict(train,test,feature_sets(role)[fs],family)
            cache[name]=value;return value
        baseline=(np.full(len(test),train.y_runs.mean()),np.full(len(test),train.y_wicket.mean()))
        original=predict('original_ebm')
        optimized=(predict(chosen['runs'])[0],predict(chosen['wickets'])[1])
        for name,(r,w) in [('constant',baseline),('original_ebm',original),('selected',optimized)]:
            results.append(dict(role=role,model=name,n=len(test),**metrics(test,r,w)))
            all_predictions.append(pd.DataFrame(dict(role=role,model=name,match_id=test.match_id.to_numpy(),y_runs=test.y_runs.to_numpy(),y_wicket=test.y_wicket.to_numpy(),pred_runs=r,pred_wicket=w)))
        for measure,y,base,pred in [('mae',test.y_runs.to_numpy(),baseline[0],optimized[0]),('mse',test.y_runs.to_numpy(),baseline[0],optimized[0]),('brier',test.y_wicket.to_numpy(),baseline[1],optimized[1])]:
            a=np.abs(y-base) if measure=='mae' else (y-base)**2
            c=np.abs(y-pred) if measure=='mae' else (y-pred)**2
            table=pd.DataFrame({'match_id':test.match_id.to_numpy(),'a':a,'c':c,'n':1}).groupby('match_id').sum().to_numpy()
            indices=rng.integers(0,len(table),size=(1000,len(table)))
            sums=table[indices].sum(axis=1)
            gain=100*(1-sums[:,1]/sums[:,0])
            intervals.append(dict(role=role,metric=measure,gain_pct=100*(1-c.mean()/a.mean()),low95=np.quantile(gain,.025),high95=np.quantile(gain,.975)))
        print(role,chosen,results[-1],flush=True)
    pd.DataFrame(results).to_csv(DEST/'reused_2026_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(DEST/'gain_intervals.csv',index=False)
    pd.concat(all_predictions).to_csv(DEST/'reused_2026_predictions.csv.gz',index=False)
    (DEST/'selected_heads.json').write_text(json.dumps(selection,indent=2))

if __name__=='__main__':main()
