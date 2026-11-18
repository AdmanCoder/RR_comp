"""Compare three pressure ratings against common later-season outcomes."""
import numpy as np
import pandas as pd

from audit import OUT
from ipl_pressure import KEYS, run
from ipl_ratings import WICKETS, rank
from rating import combine_contributions


def pooled_expectations(train, test, prior_size=150):
    """Pool sparse score states into broader historical groups."""
    expected = np.tile(train[['actual_runs', 'actual_wicket']].mean(), (len(test), 1))
    for keys in [['innings','phase'], ['innings','phase','wicket_band'],
                 ['innings','phase','wicket_band','demand_band']]:
        groups = train.groupby(keys).agg(runs=('actual_runs','sum'),
            wickets=('actual_wicket','sum'), count=('actual_runs','size'))
        lookup = groups.reindex(pd.MultiIndex.from_frame(test[keys])).fillna(0)
        expected = (lookup[['runs','wickets']].to_numpy()+prior_size*expected) / (
            lookup['count'].to_numpy()[:, None]+prior_size)
    return expected


def main(dest=OUT/'pressure_rating_comparison', pressure_power=2):
    base = run(report=False)
    dest.mkdir(parents=True,exist_ok=True)
    rows, losses = [], []
    for role in ['bat','bowl']:
        bat = role=='bat'
        data = base[base.wides.eq(0)].copy() if bat else base.copy()
        pid = 'striker_id' if bat else 'bowler_id'
        pressure = role+'_pressure'
        data['countable'] = True if bat else data.wides.eq(0)&data.noballs.eq(0)
        data['actual_runs'] = data.runs if bat else data.runs+data.wides+data.noballs
        credited = data.dismissed_player_id.eq(data.striker_id) if bat else data.dismissal_type.isin(WICKETS)
        data['actual_wicket'] = (data.is_wicket&credited).astype(int)
        data['year'] = pd.to_datetime(data.match_date).dt.year
        data['phase'] = np.select([data.balls_left.gt(84),data.balls_left.gt(30)],[0,1],default=2)
        data['wicket_band'] = pd.cut(data.wickets,[-1,2,5,10],labels=False)
        data['demand_band'] = np.digitize(6*data.needed.fillna(-1)/data.balls_left,[0,7,10,13])
        reference = data[data.year.eq(2022)]
        cutoff = reference.loc[reference.countable,pressure].quantile(.75)
        scales = reference[['actual_runs','actual_wicket']].std()
        ml = pd.read_csv(OUT/'player_evaluation_v1'/f'{role}_delivery_evaluation.csv.gz',
                         usecols=KEYS+['expected_runs','expected_wicket','train_end'])
        evaluated = []
        sign = 1 if bat else -1
        for year in [2023,2024,2025,2026]:
            train, test = data[data.year.lt(year)],data[data.year.eq(year)].copy()
            assert train.match_date.max()<test.match_date.min()
            assert not set(train.match_id)&set(test.match_id)
            test = test.merge(ml,on=KEYS,validate='one_to_one',how='left')
            assert test.expected_runs.notna().all() and (test.train_end<test.match_date).all()
            pooled = pooled_expectations(train,test)
            y = test[['actual_runs','actual_wicket']].to_numpy()
            for label, expected in [('raw',np.zeros_like(y)),('pooled',pooled),
                ('ml',test[['expected_runs','expected_wicket']].to_numpy())]:
                test[label] = combine_contributions(sign*(y[:,0]-expected[:,0]),
                    sign*(expected[:,1]-y[:,1]),*scales,'batting' if bat else 'bowling')
                if label!='raw':
                    for subset, mask in [('all',np.ones(len(test),dtype=bool)),('high',test[pressure].ge(cutoff))]:
                        error = y[mask]-expected[mask]
                        losses.append(dict(role=role,year=year,benchmark=label,subset=subset,
                            run_mse=np.mean(error[:,0]**2),wicket_brier=np.mean(error[:,1]**2)))
            evaluated.append(test)
        scored = pd.concat(evaluated)
        for year in [2024,2025,2026,2027]:
            past = scored[scored.year.lt(year)].copy()
            future = scored[scored.year.eq(year)&scored[pressure].ge(cutoff)]
            count = future.groupby(pid).countable.sum()
            outcomes = future.groupby(pid)[['raw','pooled']].mean().loc[count[count.ge(30)].index]
            for label in ['raw','pooled','ml']:
                past['contribution'] = past[label]
                board = rank(past,role,cutoff,pressure_power=pressure_power)
                if year==2027:
                    board.to_csv(dest/f'{role}_{label}_ratings.csv',index=False)
                    continue
                paired = board.set_index('player_id').join(outcomes,how='inner')
                rows.append(dict(role=role,year=year,method=label,players=len(paired),
                    future_raw_spearman=paired.rating.corr(paired.raw,method='spearman'),
                    future_pooled_spearman=paired.rating.corr(paired.pooled,method='spearman')))
    results = pd.DataFrame(rows)
    results.to_csv(dest/'later_season_validation.csv',index=False)
    pd.DataFrame(losses).to_csv(dest/'benchmark_errors.csv',index=False)
    print(results.round(3).to_string(index=False))

if __name__=='__main__':
    main()
