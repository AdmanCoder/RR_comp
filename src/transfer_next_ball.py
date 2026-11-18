"""Evaluate ideas from v11/v16 on chronological IPL folds."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from optimize_expectations import DEST, WICKETS, feature_sets, metrics, prepare


def add_context(frame, role):
    data = frame.copy()
    innings = [data.match_id, data.innings]
    first = np.minimum(data.striker_id, data.non_striker_id)
    second = np.maximum(data.striker_id, data.non_striker_id)
    partnership = [*innings, first, second]
    data['stand_runs'] = data.total_runs.groupby(partnership).cumsum() - data.total_runs
    data['stand_events'] = data.groupby(partnership).cumcount()
    dots = data.total_runs.eq(0).astype(int)
    data['previous_dots'] = dots.groupby(innings).transform(
        lambda values: values.shift().rolling(6, min_periods=1).sum()).fillna(0)
    opponent = 'bowler_id' if role == 'bat' else 'striker_id'
    history_keys = [opponent, 'phase']
    values = data.assign(history_dismissal=(
        data.is_wicket & data.dismissed_player_id.eq(data.striker_id)).astype(int))
    daily = values.groupby(history_keys + ['match_date']).agg(
        runs=('runs', 'sum'), wickets=('history_dismissal', 'sum'), events=('runs', 'size'))
    past = daily.groupby(level=[0, 1]).cumsum() - daily
    prior = past.reindex(pd.MultiIndex.from_frame(data[history_keys + ['match_date']]))
    data['phase_opponent_runs'] = (prior.runs.to_numpy() + 150) / (prior.events.to_numpy() + 100)
    data['phase_opponent_wickets'] = (prior.wickets.to_numpy() + 5) / (prior.events.to_numpy() + 100)
    return data


def main():
    raw = pd.read_pickle(DEST / 'features.pkl')
    previous = pd.read_csv(DEST / 'validation_predictions.csv.gz')
    later = pd.read_csv(DEST / 'reused_2026_predictions.csv.gz')
    rows = []
    for role in ['bat', 'bowl']:
        data = add_context(raw, role)
        if role == 'bat':
            data = data[data.wides.eq(0)].copy()
        data['y_runs'] = data.runs if role == 'bat' else data.runs + data.wides + data.noballs
        credited = (data.dismissed_player_id.eq(data.striker_id) if role == 'bat'
                    else data.dismissal_type.isin(WICKETS))
        data['y_wicket'] = (data.is_wicket & credited).astype(int)
        years = pd.to_datetime(data.match_date).dt.year
        columns = feature_sets(role)['opponent'] + [
            'stand_runs', 'stand_events', 'previous_dots', 'phase_opponent_runs',
            'phase_opponent_wickets', role + '_pressure']
        folds = {}
        for year in [2024, 2025, 2026]:
            train, test = data[years.lt(year)], data[years.eq(year)]
            assert not set(train.match_id) & set(test.match_id)
            assert train.match_date.max() < test.match_date.min()
            x, z, _ = prepare(train, test, columns, 'hist')
            options = dict(max_iter=180, max_leaf_nodes=12, min_samples_leaf=100,
                           l2_regularization=10, learning_rate=.05, early_stopping=False)
            with threadpool_limits(limits=2):
                runs = HistGradientBoostingRegressor(**options).fit(x, train.y_runs).predict(z)
                wickets = HistGradientBoostingClassifier(**options).fit(
                    x, train.y_wicket).predict_proba(z)[:, 1]
            predictions = later if year == 2026 else previous[previous.year.eq(year)]
            reference = predictions[predictions.role.eq(role) & predictions.model.eq('original_ebm')]
            for field in ['match_id', 'y_runs', 'y_wicket']:
                np.testing.assert_array_equal(reference[field], test[field])
            folds[year] = (test, np.maximum(0, runs), wickets,
                           reference.pred_runs.to_numpy(), reference.pred_wicket.to_numpy(),
                           train.y_runs.mean(), train.y_wicket.mean())
        weights = {}
        for target, index, loss in [('runs', 1, 'mse'), ('wickets', 2, 'brier')]:
            candidates = {}
            for alpha in [0., .25, .5, .75, 1.]:
                losses = []
                for year in [2024, 2025]:
                    test, runs, wickets, base_runs, base_wickets, *_ = folds[year]
                    values = [None, runs, wickets, base_runs, base_wickets]
                    truth = test.y_runs if index == 1 else test.y_wicket
                    losses.append(np.mean((truth - alpha*values[index] - (1-alpha)*values[index+2])**2))
                candidates[alpha] = np.mean(losses)
            weights[target] = min(candidates, key=candidates.get)
        for year, (test, runs, wickets, ebm_runs, ebm_wickets, mean_runs, mean_wickets) in folds.items():
            for name, r, w in [
                ('constant', np.full(len(test), mean_runs), np.full(len(test), mean_wickets)),
                ('original_ebm', ebm_runs, ebm_wickets),
                ('histgb_transfer', runs, wickets),
                ('blend', weights['runs']*runs+(1-weights['runs'])*ebm_runs,
                 weights['wickets']*wickets+(1-weights['wickets'])*ebm_wickets),
            ]:
                rows.append(dict(role=role, year=year, model=name, **metrics(test, r, w)))
        print(role, json.dumps(weights), flush=True)
    pd.DataFrame(rows).to_csv(DEST / 'v11_v16_transfer_metrics.csv', index=False)


if __name__ == '__main__':
    main()
