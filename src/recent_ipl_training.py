"""Test adaptation to the rising scoring environment with date-weighted fits."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from optimize_expectations import DEST, WICKETS, feature_sets, metrics, prepare


def main():
    data = pd.read_pickle(DEST / 'features.pkl')
    dates = pd.to_datetime(data.match_date)
    rows = []
    for role in ['bat','bowl']:
        sample = data[data.wides.eq(0)].copy() if role == 'bat' else data.copy()
        sample['y_runs'] = sample.runs if role == 'bat' else sample.runs+sample.wides+sample.noballs
        credited = (sample.dismissed_player_id.eq(sample.striker_id) if role == 'bat'
                    else sample.dismissal_type.isin(WICKETS))
        sample['y_wicket'] = (sample.is_wicket & credited).astype(int)
        year = dates.loc[sample.index].dt.year
        for evaluation in [2024,2025,2026]:
            train, test = sample[year.lt(evaluation)], sample[year.eq(evaluation)]
            assert train.match_date.max() < test.match_date.min()
            assert not set(train.match_id) & set(test.match_id)
            features = feature_sets(role)['opponent']
            x, z, _ = prepare(train,test,features,'hist')
            age = (dates.loc[train.index].max() - dates.loc[train.index]).dt.days / 365.25
            for half_life in [.75, 1.5, np.inf]:
                weights = np.exp2(-age/half_life)
                params = dict(max_iter=180,max_leaf_nodes=12,min_samples_leaf=100,
                              l2_regularization=10,learning_rate=.05,early_stopping=False)
                with threadpool_limits(limits=2):
                    runs = HistGradientBoostingRegressor(**params).fit(
                        x,train.y_runs,sample_weight=weights).predict(z)
                    wickets = HistGradientBoostingClassifier(**params).fit(
                        x,train.y_wicket,sample_weight=weights).predict_proba(z)[:,1]
                rows.append(dict(role=role,year=evaluation,half_life=half_life,
                                 **metrics(test,np.maximum(runs,0),wickets)))
            print(role,evaluation,flush=True)
    pd.DataFrame(rows).to_csv(DEST/'recent_ipl_training_metrics.csv',index=False)

if __name__ == '__main__':
    main()
