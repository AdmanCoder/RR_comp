"""Evaluate regularized wicket probability models, including an interpretable spline."""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from optimize_expectations import DEST, WICKETS, feature_sets, prepare
from pressure_ml_matchups import extra_context
from warm_history_experiment import apply_warm


def main():
    raw = pd.read_pickle(DEST/'features.pkl')
    raw = apply_warm(raw,json.loads((DEST/'warm_history_seeds.json').read_text()))
    rows = []
    for role in ['bat','bowl']:
        data = extra_context(raw,role)
        bat = role=='bat'
        if bat:
            data = data[data.wides.eq(0)].copy()
        data['target'] = (data.striker_out if bat else
            (data.is_wicket&data.dismissal_type.isin(WICKETS))).astype(int)
        pressure = role+'_pressure'
        years = pd.to_datetime(data.match_date).dt.year
        countable = data.wides.eq(0)&(bat|data.noballs.eq(0))
        cutoff = data.loc[years.eq(2022)&countable,pressure].quantile(.75)
        numerical = ['balls_left','wickets','required_rpo','current_rpo','batter_faced',
                     'phase_opponent_wickets','opponent_style_wickets','stand_events']
        categorical = ['innings','phase','free_hit','pace_or_spin','batting_type']
        fields = feature_sets(role)['opponent'] + ['phase_opponent_wickets','opponent_style_wickets']
        for year in [2024,2025,2026]:
            train,test = data[years.lt(year)],data[years.eq(year)]
            assert train.match_date.max()<test.match_date.min()
            assert not set(train.match_id)&set(test.match_id)
            transform = ColumnTransformer([
                ('splines', make_pipeline(SimpleImputer(strategy='constant',fill_value=-1),
                    SplineTransformer(n_knots=5,degree=2,knots='quantile')),numerical),
                ('category',OneHotEncoder(handle_unknown='ignore'),categorical)])
            smooth = make_pipeline(transform,LogisticRegression(C=.3,max_iter=500))
            x,z,_ = prepare(train,test,fields,'xgb')
            tree = XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.035,
                min_child_weight=30,reg_lambda=25,max_delta_step=1,n_jobs=2,random_state=17)
            with threadpool_limits(limits=2):
                smooth.fit(train,train.target)
                tree.fit(x,train.target)
                a = smooth.predict_proba(test)[:,1]
                b = tree.predict_proba(z)[:,1]
            for name, pred in [('constant',np.full(len(test),train.target.mean())),
                                ('spline_logistic',a),('regularized_xgb',b),('blend',(a+b)/2)]:
                for subset,mask in [('all',np.ones(len(test),dtype=bool)),('high',test[pressure].ge(cutoff))]:
                    outcome = test.target[mask]
                    rows.append(dict(role=role,year=year,subset=subset,model=name,
                        events=len(outcome),wickets=outcome.sum(),brier=brier_score_loss(outcome,pred[mask]),
                        logloss=log_loss(outcome,pred[mask],labels=[0,1]),
                        average_precision=average_precision_score(outcome,pred[mask])))
            print(role,year,flush=True)
    pd.DataFrame(rows).to_csv(DEST/'rare_wicket_metrics.csv',index=False)

if __name__=='__main__':
    main()
