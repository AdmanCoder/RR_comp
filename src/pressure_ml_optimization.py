"""Compare fixed ML candidates, then select each head on prior-season pressure loss."""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

from optimize_expectations import DEST, WICKETS, feature_sets, prepare
from warm_history_experiment import apply_warm
from ipl_pressure import KEYS
from ipl_ratings import rank
from rating import combine_contributions


def cat_predictions(train, test, fields, weights):
    x, z, categories = prepare(train, test, fields, 'cat')
    params = dict(iterations=350, depth=5, learning_rate=.04, l2_leaf_reg=15,
                  verbose=False, thread_count=2, random_seed=17,
                  cat_features=categories, allow_writing_files=False)
    runs = CatBoostRegressor(**params).fit(
        x, train.y_runs, sample_weight=weights).predict(z)
    wicket = CatBoostClassifier(**params).fit(
        x, train.y_wicket, sample_weight=weights).predict_proba(z)[:, 1]
    return np.column_stack([np.maximum(0, runs), wicket])


def main(dest=DEST/'pressure_focused', pressure_power=2):
    data = pd.read_pickle(DEST/'features.pkl')
    data = apply_warm(data, json.loads((DEST/'warm_history_seeds.json').read_text()))
    dest.mkdir(parents=True,exist_ok=True)
    losses, checks, selections = [], [], []
    for role in ['bat','bowl']:
        bat = role == 'bat'
        pid = 'striker_id' if bat else 'bowler_id'
        pressure = role+'_pressure'
        sample = data[data.wides.eq(0)].copy() if bat else data.copy()
        sample['countable'] = True if bat else sample.wides.eq(0)&sample.noballs.eq(0)
        sample['y_runs'] = sample.runs if bat else sample.runs+sample.wides+sample.noballs
        credited = sample.dismissed_player_id.eq(sample.striker_id) if bat else sample.dismissal_type.isin(WICKETS)
        sample['y_wicket'] = (sample.is_wicket&credited).astype(int)
        sample['year'] = sample.match_date.str[:4].astype(int)
        target = ['y_runs','y_wicket']
        ref = sample[sample.year.eq(2022)]
        cutoff = ref.loc[ref.countable,pressure].quantile(.75)
        scales = ref[target].std()
        fields = feature_sets(role)['opponent']
        ebm = pd.read_csv(DEST.parent/'player_evaluation_v1'/f'{role}_delivery_evaluation.csv.gz',
                          usecols=KEYS+['expected_runs','expected_wicket','train_end'])
        history = []
        scored = []
        for year in range(2023,2027):
            train = sample[sample.year.lt(year)]
            test = sample[sample.year.eq(year)].merge(ebm,on=KEYS,validate='one_to_one',how='left')
            assert train.match_date.max() < test.match_date.min()
            assert not set(train.match_id)&set(test.match_id)
            assert (test.train_end<test.match_date).all()
            original = test[['expected_runs','expected_wicket']].to_numpy()
            regular = cat_predictions(train,test,fields,None)
            emphasis = np.where(train[pressure].ge(cutoff),3.,1.)
            focused = cat_predictions(train,test,fields+[pressure],emphasis)
            candidates = dict(ebm=original, warm_cat=regular, pressure_cat=focused,
                              blended=(regular+focused+original)/3)
            mask = test[pressure].ge(cutoff)
            # Choose each head by earlier-season loss on the high-pressure slice.
            choice = ['ebm','ebm'] if not history else [
                min(candidates,key=lambda name: np.mean([item[name][head] for item in history]))
                for head in range(2)]
            high_loss = {}
            truth = test[target].to_numpy()
            candidates['chronological_choice'] = np.column_stack(
                [candidates[name][:,head] for head,name in enumerate(choice)])
            selections.append(dict(role=role,year=year,runs=choice[0],wickets=choice[1]))
            constant = np.broadcast_to(train[target].mean().to_numpy(),truth.shape)
            for name, expected in {**candidates,'constant':constant}.items():
                error = (truth-expected)**2
                high_loss[name] = error[mask].mean(axis=0)
                for subset, values in [('high',error[mask]),('all',error)]:
                    losses.append(dict(role=role,year=year,model=name,subset=subset,
                                       mse=values[:,0].mean(),brier=values[:,1].mean()))
                sign = 1 if bat else -1
                if name != 'constant':
                    test[name] = combine_contributions(sign*(truth[:,0]-expected[:,0]),
                        sign*(expected[:,1]-truth[:,1]),*scales,'batting' if bat else 'bowling')
            history.append(high_loss)
            scored.append(test)
            print(role,year,choice,flush=True)
        scored = pd.concat(scored)
        for year in [2025,2026,2027]:
            past = scored[scored.year.lt(year)].copy()
            later = scored[scored.year.eq(year)&scored[pressure].ge(cutoff)]
            played = later.groupby(pid).countable.sum()
            actual = later.assign(actual=combine_contributions(
                (1 if bat else -1)*later.y_runs,(-1 if bat else 1)*later.y_wicket,
                *scales,'batting' if bat else 'bowling'))
            future = actual.groupby(pid).actual.mean().loc[played[played.ge(30)].index]
            for name in candidates:
                past['contribution'] = past[name]
                board = rank(past,role,cutoff,pressure_power=pressure_power)
                if year == 2027:
                    board.to_csv(dest/f'{role}_{name}_ratings.csv',index=False)
                else:
                    paired = board.set_index('player_id').join(future,how='inner')
                    checks.append(dict(role=role,year=year,model=name,players=len(paired),
                                       future_raw_spearman=paired.rating.corr(paired.actual,method='spearman')))
    pd.DataFrame(losses).to_csv(dest/'benchmark_errors.csv',index=False)
    pd.DataFrame(checks).to_csv(dest/'later_season.csv',index=False)
    pd.DataFrame(selections).to_csv(dest/'selection.csv',index=False)

if __name__=='__main__':
    main()
