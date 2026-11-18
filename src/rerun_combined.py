"""Replay the existing architecture on verified IPL and T20I matches."""
import json
import numpy as np
import pandas as pd
from audit import OUT
from final_ipl_model import DEST, fit_role, predict, prepare_role, WICKETS
from ipl_pressure import KEYS, historical_par, score_states
from ipl_ratings import rank
from optimize_expectations import build_features
from rating import combine_contributions
from repeatable_pressure_rank import main as repeatable


def main():
    dest = OUT / 'combined_replay'
    dest.mkdir(exist_ok=True)
    ipl = pd.read_pickle(DEST / 'features.pkl').assign(source_format='IPL')
    t20 = pd.read_pickle(OUT / 'verified_t20i/deliveries.pkl').assign(source_format='T20I')
    assert not set(ipl.match_id) & set(t20.match_id)
    source_fields = t20.columns.intersection(ipl.columns).union(['source_format'])
    data = pd.concat([ipl[source_fields], t20[source_fields]], ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    data['first_innings_par'] = historical_par(data, venue_weight=20)
    data = build_features(data)
    bat_personal = ((1.30 * data.batter_faced - data.batter_runs)/(data.batter_faced+12)).clip(0,1)
    bowl_personal = ((data.bowler_charged - (11/6) * data.bowler_balls)/(data.bowler_balls+12)).clip(0,1)
    data = score_states(data, bat_personal, bowl_personal)
    path = dest / 'features.pkl'
    data.to_pickle(path)
    seed = json.loads((DEST/'warm_history_seeds.json').read_text())
    metrics = []
    for role in ['bat','bowl']:
        batting = role == 'bat'
        d = prepare_role(data,seed,role)
        if batting:
            d = d[d.wides.eq(0)].copy()
        d['countable'] = d.wides.eq(0) & (batting | d.noballs.eq(0))
        d['y_runs'] = d.runs if batting else d.charged_runs
        d['y_wicket'] = (d.striker_out if batting else (d.is_wicket & d.dismissal_type.isin(WICKETS))).astype(int)
        years = d.match_date.str[:4].astype(int)
        ref = d[years.eq(2022)]
        pressure = role+'_pressure'
        cutoff = ref.loc[ref.countable,pressure].quantile(.75)
        parts = []
        for year in sorted(years.unique())[1:]:
            train, test = d[years.lt(year)], d[years.eq(year)].copy()
            assert not set(train.match_id)&set(test.match_id)
            test[['expected_runs','expected_wicket']] = predict(fit_role(train,role,cutoff),test,role)
            sign = 1 if batting else -1
            test['contribution'] = combine_contributions(sign*(test.y_runs-test.expected_runs),
                sign*(test.expected_wicket-test.y_wicket), ref.y_runs.std(), ref.y_wicket.std(),
                'batting' if batting else 'bowling')
            for fmt,g in test.groupby('source_format'):
                metrics.append(dict(role=role,year=year,format=fmt,events=len(g),
                    runs_mse=((g.y_runs-g.expected_runs)**2).mean(),
                    wicket_brier=((g.y_wicket-g.expected_wicket)**2).mean(),
                    baseline_runs_mse=((g.y_runs-train.y_runs.mean())**2).mean(),
                    baseline_wicket_brier=((g.y_wicket-train.y_wicket.mean())**2).mean()))
            parts.append(test)
            print(role,year,len(test),flush=True)
        scored = pd.concat(parts)
        scored.to_pickle(dest/f'{role}_scored_deliveries.pkl')
        player = 'striker_id' if batting else 'bowler_id'
        weights = (scored[pressure]/100)**3
        total = weights.groupby([scored[player],scored.match_id,scored.innings]).transform('sum')
        for method,sample in [('cubed',scored),('appearance',scored.assign(
                rating_weight=weights/np.sqrt(total.where(total.gt(0),1))))]:
            board = rank(sample, role, cutoff, pressure_power=3)
            board.insert(0,'rank',range(1,len(board)+1))
            board.to_csv(dest/f'{role}_{method}.csv',index=False)
    pd.DataFrame(metrics).to_csv(dest/'metrics.csv',index=False)
    repeatable(dest/'repeatable',path,dest)


if __name__ == '__main__':
    main()
