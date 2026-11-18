"""Development-only feature removal checks, plus a distributional run model."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from optimize_expectations import DEST,build_features,feature_sets,fit_predict,metrics,WICKETS,run,prepare

def main():
    b=build_features(run(report=False));rows=[]
    for role in ['bat','bowl']:
        d=b[b.wides.eq(0)].copy() if role=='bat' else b.copy()
        d['y_runs']=d.runs if role=='bat' else d.runs+d.wides+d.noballs
        d['y_wicket']=(d.is_wicket&(d.dismissed_player_id.eq(d.striker_id) if role=='bat' else d.dismissal_type.isin(WICKETS))).astype(int)
        years=pd.to_datetime(d.match_date).dt.year
        features=feature_sets(role)['opponent']
        groups={'drop_knockout':['knock_out'],
                'drop_recent':[f for f in features if f.startswith('recent_')],
                'drop_styles':['batting_type','bowling_type','pace_or_spin'],
                'drop_history':[f for f in features if '_past_' in f]}
        for year in [2024,2025]:
            train=d[years.lt(year)];test=d[years.eq(year)]
            for label,drop in groups.items():
                r,w=fit_predict(train,test,[f for f in features if f not in drop],'cat')
                row=dict(role=role,year=year,variant=label,**metrics(test,r,w));rows.append(row);print(row,flush=True)
            x,z,cats=prepare(train,test,features,'cat')
            model=CatBoostClassifier(iterations=350,depth=5,learning_rate=.04,l2_leaf_reg=15,thread_count=2,random_seed=17,verbose=False,allow_writing_files=False,cat_features=cats,loss_function='MultiClass')
            model.fit(x,train.y_runs.astype(int))
            probabilities=model.predict_proba(z)
            expectation=probabilities@model.classes_.astype(float)
            # Report runs only: no replacement wicket model was fitted here.
            rows.append(dict(role=role,year=year,variant='run_distribution',mae=np.mean(abs(test.y_runs-expectation)),mse=np.mean((test.y_runs-expectation)**2)))
            pd.DataFrame(rows).to_csv(DEST/'feature_ablation.csv',index=False)
if __name__=='__main__':main()
