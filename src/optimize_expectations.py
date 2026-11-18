"""Chronological feature/model ablations. Never writes player ratings."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss
from audit import OUT
from ipl_pressure import run, prior_sum
from ipl_ratings import FEATURES, WICKETS
DEST=OUT/'model_optimization'


def build_features(b):
    b=b.copy()
    groups=[b.match_id,b.innings]
    legal=(b.wides.eq(0)&b.noballs.eq(0)).astype(int)
    b['played']=120-b.balls_left
    b['required_rpo']=6*b.needed/b.balls_left
    b['current_rpo']=6*b.score/(b.played+1)
    b['phase']=np.select([b.played.lt(36),b.played.lt(90)],[0,1],default=2)
    b['legal_ball_in_over']=b.played%6
    b['par_deficit']=b.first_innings_par-b.score
    b['batter_faced']=prior_sum(b.wides.eq(0).astype(int),[b.match_id,b.innings,b.striker_id])
    b['batter_runs']=prior_sum(b.runs,[b.match_id,b.innings,b.striker_id])
    b['bowler_balls']=prior_sum(legal,[b.match_id,b.innings,b.bowler_id])
    b['bowler_charged']=prior_sum(b.runs+b.wides+b.noballs,[b.match_id,b.innings,b.bowler_id])
    b['batter_rate']=(b.batter_runs+12*1.5)/(b.batter_faced+12)
    b['bowler_rate']=(b.bowler_charged+12*1.5)/(b.bowler_balls+12)
    for window in [6,12]:
        b[f'recent_runs_{window}']=b.groupby(['match_id','innings']).total_runs.transform(lambda s:s.shift().rolling(window,min_periods=1).sum()).fillna(0)
        b[f'recent_wickets_{window}']=b.groupby(['match_id','innings']).is_wicket.transform(lambda s:s.astype(int).shift().rolling(window,min_periods=1).sum()).fillna(0)
    # A no-ball makes the next delivery a free hit; a wide does not consume it.
    free=[]
    for _,g in b.groupby(['match_id','innings'],sort=False):
        active=False
        for row in g.itertuples():
            free.append(int(active))
            if row.noballs>0:active=True
            elif row.wides==0:active=False
    b['free_hit']=free
    # Empirical history uses earlier dates only, never today's other match.
    outcomes=pd.DataFrame({'r':b.runs,'w':(b.is_wicket&b.dismissed_player_id.eq(b.striker_id)).astype(int), 'n':1},index=b.index)
    for entity in ['striker_id','bowler_id','venue_id']:
        table=pd.concat([b[[entity,'match_date']],outcomes],axis=1)
        daily=table.groupby([entity,'match_date'])[['r','w','n']].sum().sort_index()
        prior=daily.groupby(level=0).cumsum()-daily
        lookup=prior.reindex(pd.MultiIndex.from_frame(b[[entity,'match_date']])).reset_index(drop=True)
        strength=120 if entity!='venue_id' else 600
        b[entity+'_past_runs']=(lookup.r.to_numpy()+strength*1.5)/(lookup.n.to_numpy()+strength)
        b[entity+'_past_wickets']=(lookup.w.to_numpy()+strength*.05)/(lookup.n.to_numpy()+strength)
        b[entity+'_past_n']=lookup.n.to_numpy()
    # Stable style attributes only. No current delivery speed, shot or outcome fields.
    for column in ['batting_type','bowling_type','pace_or_spin']:
        b[column]=b[column].fillna('unknown').astype(str)
    return b


def feature_sets(role):
    context=FEATURES+['required_rpo','current_rpo','phase','legal_ball_in_over','par_deficit',
        'batter_faced','bowler_balls','recent_runs_6','recent_runs_12','recent_wickets_6',
        'recent_wickets_12','free_hit','batting_type','bowling_type','pace_or_spin',
        'venue_id_past_runs','venue_id_past_wickets','venue_id_past_n']
    opponent='bowler_id' if role=='bat' else 'striker_id'
    adjusted=context+[opponent+'_past_runs',opponent+'_past_wickets',opponent+'_past_n']
    own='striker_id' if role=='bat' else 'bowler_id'
    predictive=adjusted+[own+'_past_runs',own+'_past_wickets',own+'_past_n','batter_rate','bowler_rate']
    return {'original':FEATURES,'context':context,'opponent':adjusted,'predictive':predictive}


def prepare(train,test,features,family):
    x=train[features].copy();z=test[features].copy()
    categorical=list(x.select_dtypes(include='object').columns)
    if family=='cat':
        return x.fillna(-999),z.fillna(-999),categorical
    for c in categorical:
        # Train-only category mapping; unseen test styles get -1.
        mapping={v:i for i,v in enumerate(sorted(x[c].unique()))}
        x[c]=x[c].map(mapping);z[c]=z[c].map(mapping).fillna(-1)
    return x.fillna(-999),z.fillna(-999),[]


def fit_predict(train,test,features,family):
    x,z,cats=prepare(train,test,features,family)
    if family=='cat':
        kw=dict(iterations=350,depth=5,learning_rate=.04,l2_leaf_reg=15,thread_count=2,random_seed=17,verbose=False,allow_writing_files=False,cat_features=cats)
        r=CatBoostRegressor(loss_function='RMSE',**kw);w=CatBoostClassifier(loss_function='Logloss',**kw)
    elif family=='xgb':
        kw=dict(n_estimators=350,max_depth=3,learning_rate=.035,min_child_weight=80,reg_lambda=20,subsample=.85,colsample_bytree=.9,n_jobs=2,random_state=17)
        r=XGBRegressor(objective='reg:squarederror',**kw);w=XGBClassifier(objective='binary:logistic',**kw)
    else:
        kw=dict(interactions=2,outer_bags=1,validation_size=0,max_rounds=300,max_bins=32,min_samples_leaf=80,n_jobs=2,random_state=17)
        r=ExplainableBoostingRegressor(**kw);w=ExplainableBoostingClassifier(**kw)
    r.fit(x,train.y_runs);w.fit(x,train.y_wicket)
    return np.maximum(0,r.predict(z)),np.clip(w.predict_proba(z)[:,1],1e-6,1-1e-6)


def metrics(t,r,w):
    return dict(mae=mean_absolute_error(t.y_runs,r),mse=mean_squared_error(t.y_runs,r),
                brier=brier_score_loss(t.y_wicket,w),logloss=log_loss(t.y_wicket,w,labels=[0,1]))


def main():
    DEST.mkdir(exist_ok=True)
    base=run(report=False);b=build_features(base)
    rows=[]; predictions=[]
    configurations=[('original','ebm'),('context','ebm'),('context','cat'),('context','xgb'),('opponent','cat'),('opponent','xgb'),('predictive','cat')]
    for role in ['bat','bowl']:
        d=b[b.wides.eq(0)].copy() if role=='bat' else b.copy()
        d['y_runs']=d.runs if role=='bat' else d.runs+d.wides+d.noballs
        d['y_wicket']=(d.is_wicket & (d.dismissed_player_id.eq(d.striker_id) if role=='bat' else d.dismissal_type.isin(WICKETS))).astype(int)
        years=pd.to_datetime(d.match_date).dt.year
        sets=feature_sets(role)
        for year in [2024,2025]:
            train=d[years.lt(year)];test=d[years.eq(year)]
            assert pd.to_datetime(train.match_date).max()<pd.to_datetime(test.match_date).min()
            assert not set(train.match_id)&set(test.match_id)
            cached={}
            base_r=np.full(len(test),train.y_runs.mean());base_w=np.full(len(test),train.y_wicket.mean())
            variants=[('constant',base_r,base_w)]
            # Stronger simple benchmark: phase/innings/wicket-count group mean, shrunk.
            keys=['innings','phase','wickets'];table=train.groupby(keys).agg(r=('y_runs','sum'),w=('y_wicket','sum'),n=('y_runs','size'))
            tab=table.reindex(pd.MultiIndex.from_frame(test[keys])).fillna(0)
            variants.append(('bucket_baseline',(tab.r.to_numpy()+100*train.y_runs.mean())/(tab.n.to_numpy()+100),(tab.w.to_numpy()+100*train.y_wicket.mean())/(tab.n.to_numpy()+100)))
            for fs,family in configurations:
                name=fs+'_'+family
                r,w=fit_predict(train,test,sets[fs],family);cached[name]=(r,w);variants.append((name,r,w))
                print(role,year,name,metrics(test,r,w),flush=True)
            for fs in ['context','opponent']:
                a=cached[fs+'_cat'];c=cached[fs+'_xgb']
                variants.append((fs+'_ensemble',(a[0]+c[0])/2,(a[1]+c[1])/2))
            for name,r,w in variants:
                rows.append(dict(role=role,year=year,model=name,n=len(test),**metrics(test,r,w)))
                predictions.append(pd.DataFrame(dict(role=role,year=year,model=name,match_id=test.match_id.to_numpy(),y_runs=test.y_runs.to_numpy(),y_wicket=test.y_wicket.to_numpy(),pred_runs=r,pred_wicket=w)))
            pd.DataFrame(rows).to_csv(DEST/'validation_metrics.csv',index=False)
    pd.concat(predictions).to_csv(DEST/'validation_predictions.csv.gz',index=False)
    (DEST/'feature_manifest.json').write_text(json.dumps({r:feature_sets(r) for r in ['bat','bowl']},indent=2))
    # Selection uses equal-fold relative MSE and Brier, not the previously seen 2026.
    scores=pd.DataFrame(rows)
    choices={}
    for role in ['bat','bowl']:
        part=scores[scores.role.eq(role)].copy();baseline=part[part.model.eq('constant')].set_index('year')
        part['objective']=.5*(part.mse/part.year.map(baseline.mse)+part.brier/part.year.map(baseline.brier))
        # Own ability is a forecasting diagnostic, not the common ranking benchmark.
        eligible=part[~part.model.str.startswith('predictive') & ~part.model.isin(['constant','bucket_baseline'])]
        choices[role]=eligible.groupby('model').objective.mean().idxmin()
    (DEST/'selected.json').write_text(json.dumps(choices,indent=2))
    print('SELECTED',choices,flush=True)
    b.to_pickle(DEST/'features.pkl')

if __name__=='__main__':main()
