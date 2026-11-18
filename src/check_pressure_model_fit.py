"""Check selected outcome models where pressure ratings place the most weight."""
import numpy as np
import pandas as pd

from optimize_expectations import DEST


def main():
    features = pd.read_pickle(DEST / 'features.pkl')
    predictions = pd.read_csv(DEST / 'reused_2026_predictions.csv.gz')
    years = pd.to_datetime(features.match_date).dt.year
    rows = []
    for role in ['bat', 'bowl']:
        pressure = role + '_pressure'
        eligible = features.wides.eq(0) if role == 'bat' else pd.Series(True, index=features.index)
        reference = features.loc[eligible & years.eq(2022), pressure]
        test = features.loc[eligible & years.eq(2026)].reset_index(drop=True)
        thresholds = reference.quantile([.75, .9])
        groups = {
            'all': np.ones(len(test), dtype=bool),
            'high_2022_p75': test[pressure].ge(thresholds.loc[.75]),
            'very_high_2022_p90': test[pressure].ge(thresholds.loc[.9]),
        }
        actual_runs = test.runs if role == 'bat' else test.runs + test.wides + test.noballs
        for model, predictions_for_model in predictions.loc[predictions.role.eq(role)].groupby('model'):
            p = predictions_for_model.reset_index(drop=True)
            np.testing.assert_array_equal(p.match_id, test.match_id)
            np.testing.assert_array_equal(p.y_runs, actual_runs)
            run_error = p.pred_runs - p.y_runs
            wicket_error = p.pred_wicket - p.y_wicket
            for group, mask in groups.items():
                rows.append(dict(role=role, model=model, group=group,
                                 events=int(np.sum(mask)), matches=test.loc[mask,'match_id'].nunique(),
                                 run_bias=run_error[mask].mean(),
                                 runs_mse=run_error[mask].pow(2).mean(),
                                 runs_mae=run_error[mask].abs().mean(),
                                 wicket_bias_pp=100*wicket_error[mask].mean(),
                                 wicket_brier=wicket_error[mask].pow(2).mean()))
    results = pd.DataFrame(rows)
    results.to_csv(DEST / 'pressure_segment_metrics.csv', index=False)
    base = results[results.model.eq('constant')].set_index(['role','group'])
    selected = results[results.model.eq('selected')].set_index(['role','group'])
    for metric in ['runs_mse','runs_mae','wicket_brier']:
        selected[metric+'_gain_pct'] = 100*(1-selected[metric]/base[metric])
    selected.to_csv(DEST / 'pressure_segment_gains.csv')
    print(selected.round(4).to_string())


if __name__ == '__main__':
    main()
