"""Small chronological model comparison on verified, pre-ball features only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss, brier_score_loss
from audit import run, OUT
from features import prepare_features, assert_split, NUMERIC, CATEGORICAL

def main():
    _, b = run()
    b,X=prepare_features(b)
    # No current delivery stroke, speed, style, result, IDs, personal ability or inferred extras.
    numeric=NUMERIC
    categorical=CATEGORICAL
    features=numeric+categorical
    # Striker dismissal is deliberately narrower than all batter dismissals. Non-striker run-outs
    # require separate exposure attribution and are not silently charged to the striker.
    dismissed=pd.to_numeric(b.dismissed_player_id,errors='coerce')
    targets={'bat_runs':b.runs.astype(float),
             'striker_dismissal':(b.is_wicket & dismissed.eq(b.striker_id)).astype(int),
             'bowler_wicket':(b.is_wicket & b.dismissal_type.isin(['Caught','Bowled','LBW','Stump','Hit Wicket'])).astype(int)}
    folds=[('dev_2024',b.year.le(2023),b.year.eq(2024)),('dev_2025',b.year.le(2024),b.year.eq(2025))]
    for _,tr,va in folds:
        assert_split(b,tr,va)
    def model(name,classification):
        if name=='baseline':
            prep=ColumnTransformer([('num',make_pipeline(SimpleImputer(),StandardScaler()),numeric),('cat',OneHotEncoder(handle_unknown='ignore'),categorical)])
            return make_pipeline(prep,LogisticRegression(C=0.3,max_iter=250) if classification else Ridge(alpha=50,solver='lsqr'))
        if name=='ebm':
            from interpret.glassbox import ExplainableBoostingRegressor,ExplainableBoostingClassifier
            cls=ExplainableBoostingClassifier if classification else ExplainableBoostingRegressor
            # No random validation across deliveries; fixed conservative boosting budget.
            return cls(interactions=5,outer_bags=1,validation_size=0,max_rounds=600,max_bins=32,min_samples_leaf=100,n_jobs=2,random_state=17)
        if name=='catboost':
            from catboost import CatBoostClassifier,CatBoostRegressor
            cls=CatBoostClassifier if classification else CatBoostRegressor
            return cls(iterations=250,depth=5,learning_rate=0.05,l2_leaf_reg=8,cat_features=categorical,thread_count=2,verbose=False,allow_writing_files=False,random_seed=17)
        from xgboost import XGBClassifier,XGBRegressor
        prep=ColumnTransformer([('num',SimpleImputer(),numeric),('cat',OneHotEncoder(handle_unknown='ignore'),categorical)])
        cls=XGBClassifier if classification else XGBRegressor
        return make_pipeline(prep,cls(n_estimators=250,max_depth=4,learning_rate=.05,min_child_weight=30,subsample=.8,colsample_bytree=.9,reg_lambda=10,n_jobs=2,random_state=17,tree_method='hist'))
    records=[]; failures=[]; predictions=[]
    names=['baseline','ebm','catboost','xgboost']
    for target,y in targets.items():
        classification=target!='bat_runs'
        for name in names:
            for fold,tr,va in folds:
                start=time.time()
                try:
                    estimator=model(name,classification);estimator.fit(X.loc[tr],y.loc[tr])
                    p=estimator.predict_proba(X.loc[va])[:,1] if classification else estimator.predict(X.loc[va])
                    record={'target':target,'model':name,'fold':fold,'train_matches':b.loc[tr,'match_id'].nunique(),'validation_matches':b.loc[va,'match_id'].nunique(),'n':int(va.sum()),'seconds':round(time.time()-start,2)}
                    if classification:
                        record.update(loss=log_loss(y.loc[va],p),brier=brier_score_loss(y.loc[va],p))
                    else:record.update(loss=mean_squared_error(y.loc[va],p),rmse=float(np.sqrt(mean_squared_error(y.loc[va],p))))
                    record.update(actual_mean=float(y.loc[va].mean()),predicted_mean=float(np.mean(p)))
                    records.append(record); print(json.dumps(record),flush=True)
                except Exception as exc:
                    failures.append({'target':target,'model':name,'fold':fold,'error':str(exc)});print(failures[-1],flush=True);break
                pd.DataFrame(records).to_csv(OUT/'model_validation.csv',index=False)
        available=pd.DataFrame(records); scores=available[available.target.eq(target)].groupby('model').filter(lambda q:len(q)==2).groupby('model').loss.mean()
        best=scores.idxmin()
        # Prefer interpretability if EBM is within 1% of best mean development loss.
        selected='ebm' if 'ebm' in scores and scores['ebm'] <= scores.min()*1.01 else best
        tr=b.year.le(2025);te=b.year.eq(2026)
        assert_split(b,tr,te)
        estimator=model(selected,classification);estimator.fit(X.loc[tr],y.loc[tr])
        p=estimator.predict_proba(X.loc[te])[:,1] if classification else estimator.predict(X.loc[te])
        loss=log_loss(y.loc[te],p) if classification else mean_squared_error(y.loc[te],p)
        predictions.append({'target':target,'selected_model':selected,'test_year':2026,'test_loss':float(loss),'actual_mean':float(y.loc[te].mean()),'predicted_mean':float(np.mean(p)),'status':'diagnostic baseline; score-dependent context unavailable'})
        (OUT/'model_test.json').write_text(json.dumps(predictions,indent=2))
    (OUT/'model_failures.json').write_text(json.dumps(failures,indent=2))
    (OUT/'feature_manifest.json').write_text(json.dumps({'features':features,'exclusions':'No unverified score reconstruction, player identity, current-delivery attributes, post-match outcomes or revised target timing.'},indent=2))

if __name__=='__main__':main()
