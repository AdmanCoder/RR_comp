"""Test opponent style splits and live scoring conditions in IPL forecasts."""
import json

import numpy as np
import pandas as pd

from optimize_expectations import DEST, WICKETS, feature_sets, metrics
from pressure_ml_optimization import cat_predictions
from transfer_next_ball import add_context
from warm_history_experiment import apply_warm


def extra_context(frame, role):
    data = add_context(frame, role)
    opponent = 'bowler_id' if role == 'bat' else 'striker_id'
    style = 'batting_type' if role == 'bat' else 'pace_or_spin'
    groups = [opponent, style]
    data['striker_out'] = (data.is_wicket & data.dismissed_player_id.eq(data.striker_id)).astype(int)
    data['charged_runs'] = data.runs + data.wides + data.noballs
    runs = 'charged_runs' if role == 'bat' else 'runs'
    by_day = data.groupby(groups+['match_date']).agg(
        runs=(runs,'sum'), outs=('striker_out','sum'), balls=(runs,'size')).sort_index()
    totals = by_day.groupby(level=[0,1]).cumsum() - by_day
    prior = totals.reindex(pd.MultiIndex.from_frame(data[groups+['match_date']]))
    data['opponent_style_runs'] = (prior.runs.to_numpy() + 150)/(prior.balls.to_numpy()+100)
    data['opponent_style_wickets'] = (prior.outs.to_numpy() + 5)/(prior.balls.to_numpy()+100)
    data['opponent_style_balls'] = prior.balls.to_numpy()
    # Available once the first innings is complete. The chase target encodes it.
    data['set_target'] = data.needed + data.score
    data['chase_rate_gap'] = data.required_rpo - data.current_rpo
    # Compare scoring so far with a historical par based on earlier dates.
    data['live_scoring_gap'] = (data.score-data.first_innings_par)/(data.played+24)
    data.loc[data.innings.eq(2),'live_scoring_gap'] = np.nan
    return data


EXTRA = ['opponent_style_runs','opponent_style_wickets','opponent_style_balls',
         'phase_opponent_runs','phase_opponent_wickets','stand_runs','stand_events',
         'previous_dots','set_target','chase_rate_gap','live_scoring_gap']


def main():
    raw = pd.read_pickle(DEST/'features.pkl')
    raw = apply_warm(raw,json.loads((DEST/'warm_history_seeds.json').read_text()))
    rows = []
    for role in ['bat','bowl']:
        data = extra_context(raw,role)
        bat = role=='bat'
        if bat:
            data = data[data.wides.eq(0)].copy()
        data['y_runs'] = data.runs if bat else data.charged_runs
        data['y_wicket'] = (data.striker_out if bat else
            (data.is_wicket&data.dismissal_type.isin(WICKETS))).astype(int)
        years = pd.to_datetime(data.match_date).dt.year
        pressure = role+'_pressure'
        countable = data.wides.eq(0)&(bat|data.noballs.eq(0))
        cutoff = data.loc[years.eq(2022)&countable,pressure].quantile(.75)
        core = feature_sets(role)['opponent']
        for year in [2024,2025,2026]:
            train,test = data[years.lt(year)],data[years.eq(year)]
            assert train.match_date.max()<test.match_date.min()
            assert not set(train.match_id)&set(test.match_id)
            for model, columns in [('warm_cat',core),('richer_cat',core+EXTRA)]:
                prediction = cat_predictions(train,test,columns,None)
                for subset, mask in [('all',np.ones(len(test),dtype=bool)),
                                     ('high',test[pressure].ge(cutoff))]:
                    report = metrics(test.loc[mask],prediction[mask,0],prediction[mask,1])
                    baseline = np.mean((test.loc[mask,'y_runs']-train.y_runs.mean())**2)
                    rows.append(dict(role=role,year=year,subset=subset,model=model,
                        constant_mse=baseline,mse_gain_pct=100*(1-report['mse']/baseline),**report))
            pd.DataFrame(rows).to_csv(DEST/'matchup_and_live_context.csv',index=False)
            print(role,year,flush=True)


if __name__=='__main__':
    main()
