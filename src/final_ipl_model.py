"""Fit, replay and load the selected IPL expected-performance models."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from xgboost import XGBClassifier

from ipl_pressure import KEYS
from optimize_expectations import DEST, FEATURES, WICKETS, feature_sets, metrics
from pressure_ml_matchups import EXTRA, extra_context
from warm_history_experiment import apply_warm

OUTPUT = DEST/'final_ipl_model'
SPLINE_NUM = ['balls_left','wickets','required_rpo','current_rpo','batter_faced',
              'phase_opponent_wickets','opponent_style_wickets','stand_events']
SPLINE_CAT = ['innings','phase','free_hit','pace_or_spin','batting_type']


def fit_component(train, fields, kind, weights=None):
    x = train[fields].copy()
    mappings = {}
    if kind in ('xgb','ebm'):
        for column in x.select_dtypes(include='object'):
            mappings[column] = {value:i for i,value in enumerate(sorted(x[column].unique()))}
            x[column] = x[column].map(mappings[column])
    if kind != 'spline':
        x = x.fillna(-999)
    if kind in ('runs','cat'):
        args = dict(iterations=350,depth=5,learning_rate=.04,l2_leaf_reg=15,
                    verbose=False,thread_count=2,random_seed=17,allow_writing_files=False,
                    cat_features=list(x.select_dtypes(include='object').columns))
        model = CatBoostRegressor(**args) if kind=='runs' else CatBoostClassifier(**args)
    elif kind == 'ebm':
        model = ExplainableBoostingClassifier(interactions=2,outer_bags=1,validation_size=0,
            max_rounds=300,max_bins=32,min_samples_leaf=80,n_jobs=2,random_state=17)
    elif kind == 'xgb':
        model = XGBClassifier(n_estimators=300,max_depth=3,learning_rate=.035,
            min_child_weight=30,reg_lambda=25,max_delta_step=1,n_jobs=2,random_state=17)
    else:
        transform = ColumnTransformer([
            ('splines',make_pipeline(SimpleImputer(strategy='constant',fill_value=-1),
             SplineTransformer(n_knots=5,degree=2,knots='quantile')),SPLINE_NUM),
            ('categories',OneHotEncoder(handle_unknown='ignore'),SPLINE_CAT)])
        model = make_pipeline(transform,LogisticRegression(C=.3,max_iter=500))
    args = {} if weights is None else {'sample_weight':weights}
    model.fit(x,train.y_runs if kind=='runs' else train.y_wicket,**args)
    return dict(model=model,fields=fields,kind=kind,mappings=mappings)


def component_predictions(component, frame):
    x = frame[component['fields']].copy()
    for field, mapping in component['mappings'].items():
        x[field] = x[field].map(mapping).fillna(-1)
    if component['kind']!='spline':
        x = x.fillna(-999)
    model = component['model']
    if component['kind']=='runs':
        return np.maximum(0,model.predict(x))
    return model.predict_proba(x)[:,1]


def fit_role(train, role, cutoff):
    core = feature_sets(role)['opponent']
    if role=='bat':
        weights = np.where(train.bat_pressure.ge(cutoff),3.,1.)
        wicket = [fit_component(train,FEATURES,'ebm'),fit_component(train,core,'cat'),
                  fit_component(train,core+['bat_pressure'],'cat',weights)]
        runs = fit_component(train,core,'runs')
    else:
        wicket = [fit_component(train,SPLINE_NUM+SPLINE_CAT,'spline'),
                  fit_component(train,core+['phase_opponent_wickets','opponent_style_wickets'],'xgb')]
        runs = fit_component(train,core+EXTRA,'runs')
    return dict(runs=runs,wicket=wicket,cutoff=cutoff,trained_through=train.match_date.max())


def predict(bundle, prepared, role):
    if prepared.match_date.le(bundle['trained_through']).any():
        raise ValueError('Prediction dates must follow the model training cutoff.')
    runs = component_predictions(bundle['runs'],prepared)
    wicket = np.mean([component_predictions(m,prepared) for m in bundle['wicket']],axis=0)
    if role=='bowl':
        wicket[prepared.free_hit.eq(1)] = 0
    return np.column_stack([runs,wicket])


def prepare_role(raw, seed, role):
    return extra_context(apply_warm(raw,seed),role)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,help='Prepared base feature pickle from the IPL feature builder')
    parser.add_argument('--role',choices=['bat','bowl'])
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if any([args.features,args.role,args.output]):
        if not all([args.features,args.role,args.output]):
            parser.error('--features, --role and --output are required together')
        artifact = joblib.load(OUTPUT/f'{args.role}.joblib')
        data = prepare_role(pd.read_pickle(args.features),artifact['seed'],args.role)
        data = data[data.match_date.gt(artifact['trained_through'])]
        if args.role=='bat':
            data = data[data.wides.eq(0)]
        if data.empty:
            parser.error('No eligible events dated after the model training cutoff')
        result = data[KEYS+['match_date']].copy()
        result[['expected_runs','expected_wicket']] = predict(artifact,data,args.role)
        result.to_csv(args.output,index=False)
        return
    OUTPUT.mkdir(exist_ok=True)
    raw = pd.read_pickle(DEST/'features.pkl')
    seed = json.loads((DEST/'warm_history_seeds.json').read_text())
    summary = []
    for role in ['bat','bowl']:
        data = prepare_role(raw,seed,role)
        bat = role=='bat'
        if bat:
            data = data[data.wides.eq(0)].copy()
        data['y_runs'] = data.runs if bat else data.charged_runs
        data['y_wicket'] = (data.striker_out if bat else
            (data.is_wicket&data.dismissal_type.isin(WICKETS))).astype(int)
        year = data.match_date.str[:4].astype(int)
        countable = data.wides.eq(0)&(bat|data.noballs.eq(0))
        cutoff = data.loc[year.eq(2022)&countable,role+'_pressure'].quantile(.75)
        train,test = data[year.lt(2026)],data[year.eq(2026)]
        assert not set(train.match_id)&set(test.match_id)
        replay = fit_role(train,role,cutoff)
        estimates = predict(replay,test,role)
        exported = test[KEYS+['match_date','y_runs','y_wicket',role+'_pressure','free_hit']].copy()
        exported[['expected_runs','expected_wicket']] = estimates
        exported.to_csv(OUTPUT/f'{role}_2026_predictions.csv',index=False)
        for subset, mask in [('all',np.ones(len(test),dtype=bool)),
                             ('high',test[role+'_pressure'].ge(cutoff))]:
            actual = test.loc[mask]
            scores = metrics(actual,estimates[mask,0],estimates[mask,1])
            base = metrics(actual,np.full(len(actual),train.y_runs.mean()),
                           np.full(len(actual),train.y_wicket.mean()))
            summary.append(dict(role=role,subset=subset,events=len(actual),**scores,
                mse_gain=100*(1-scores['mse']/base['mse']),
                brier_gain=100*(1-scores['brier']/base['brier'])))
        production = fit_role(data,role,cutoff)
        production['seed'] = seed
        path = OUTPUT/f'{role}.joblib'
        joblib.dump(production,path,compress=3)
        loaded = joblib.load(path)
        probe = data.tail(20).copy()
        probe['match_date'] = '2099-01-01'
        np.testing.assert_allclose(predict(production,probe,role),predict(loaded,probe,role))
        print(role,production['trained_through'],flush=True)
    pd.DataFrame(summary).to_csv(OUTPUT/'reused_2026_metrics.csv',index=False)
    print(pd.DataFrame(summary).round(4).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
