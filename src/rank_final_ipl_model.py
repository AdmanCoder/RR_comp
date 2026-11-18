"""Rank verified IPL players using earlier-season fits of the saved model setup."""
import argparse
import json

import numpy as np

import pandas as pd

from final_ipl_model import DEST, OUTPUT, KEYS, WICKETS, prepare_role, fit_role, predict
from ipl_ratings import rank
from rating import combine_contributions


def bowling_tradeoff(evaluated, cutoff, scales):
    events = evaluated.copy()
    events['runs_saved'] = events.expected_runs - events.y_runs
    events['extra_wickets'] = events.y_wicket - events.expected_wicket
    implied = (0.4 / scales.y_wicket) / (0.6 / scales.y_runs)
    tables = []
    for value in [0, 3, 4, implied, 6, 8, 10]:
        events['contribution'] = events.runs_saved + value * events.extra_wickets
        board = rank(events, 'bowl', cutoff)
        board.insert(0, 'rank', range(1,len(board)+1))
        board.insert(0, 'runs_per_wicket', value)
        tables.append(board)
    pd.concat(tables).to_csv(OUTPUT/'bowling_wicket_value_sensitivity.csv',index=False)
    events['phase'] = np.select([events.balls_left.gt(84),events.balls_left.gt(30)],
                                ['powerplay','middle'],default='death')
    events['pressure_band'] = np.where(events.bowl_pressure.ge(cutoff),'high','other')
    events['weight'] = (events.bowl_pressure/100)**2
    events['weighted_runs_saved'] = events.weight*events.runs_saved
    events['weighted_extra_wickets'] = events.weight*events.extra_wickets
    events.groupby(['bowler_id','bowler_name','phase','pressure_band']).agg(
        events=('runs_saved','size'), legal_balls=('countable','sum'),
        actual_runs=('y_runs','sum'), expected_runs=('expected_runs','sum'),
        actual_wickets=('y_wicket','sum'), expected_wickets=('expected_wicket','sum'),
        weighted_runs_saved=('weighted_runs_saved','sum'),
        weighted_extra_wickets=('weighted_extra_wickets','sum')
    ).to_csv(OUTPUT/'bowling_runs_wickets_by_phase.csv')
    events['contribution'] = events.runs_saved + implied * events.extra_wickets
    pressure_boards = []
    for label, sample, threshold in [
        ('current',events,cutoff),
        ('high_only',events.loc[events.bowl_pressure.ge(cutoff)],cutoff),
        ('linear_weight',events.assign(bowl_pressure=np.sqrt(events.bowl_pressure/100)*100),
         np.sqrt(cutoff/100)*100),
        ('equal_weight_high',events.loc[events.bowl_pressure.ge(cutoff)].assign(bowl_pressure=100),100),
    ]:
        board = rank(sample,'bowl',threshold)
        board.insert(0,'rank',range(1,len(board)+1))
        board.insert(0,'variant',label)
        pressure_boards.append(board)
    pd.concat(pressure_boards).to_csv(OUTPUT/'bowling_pressure_weight_sensitivity.csv',index=False)
    # Keep predictions and scoring fixed. Compare changes to pressure weighting.
    reference = pd.read_pickle(DEST/'features.pkl')
    reference = reference[reference.match_date.str.startswith('2022') &
                          reference.wides.eq(0)&reference.noballs.eq(0)].copy()

    def phase_adjusted(data):
        # Six-point new-ball premium fades over the final two powerplay overs.
        # The existing extreme-chase filter suppresses the premium in dead chases.
        premium = 6*np.clip((data.balls_left-84)/12,0,1)
        return np.minimum(100,data.bowl_pressure+premium*data.chase_extreme_factor)

    def percentiles(values, anchors):
        ordered = np.sort(anchors)
        return 50*(np.searchsorted(ordered,values,side='left')+
                   np.searchsorted(ordered,values,side='right'))/len(ordered)

    boosted = phase_adjusted(events)
    boosted_ref = phase_adjusted(reference)
    boosted_cutoff = boosted_ref.quantile(.75)
    percentile = percentiles(events.bowl_pressure,reference.bowl_pressure)
    experiments = [
        ('current',events.bowl_pressure,cutoff,1.),
        ('powerplay_premium',boosted,boosted_cutoff,1.),
        ('below_cutoff_discount',events.bowl_pressure,cutoff,.25),
        ('historical_percentile',percentile,percentiles([cutoff],reference.bowl_pressure)[0],1.),
        ('premium_and_discount',boosted,boosted_cutoff,.25),
    ]
    profiles, rankings = [], []
    for label, scores, threshold, discount in experiments:
        sample = events.assign(bowl_pressure=scores)
        high = sample.bowl_pressure.ge(threshold)
        # Pass weights separately so discounting never changes the eligibility flags.
        sample['rating_weight'] = (sample.bowl_pressure/100)**2*np.where(high,1.,discount)
        board = rank(sample,'bowl',threshold)
        board.insert(0,'rank',range(1,len(board)+1))
        rankings.append(board.assign(variant=label))
        legal = sample[sample.countable].assign(high=high)
        for phase, group in legal.groupby('phase'):
            profiles.append(dict(variant=label,phase=phase,cutoff=threshold,
                high_pct=100*group.high.mean(),
                weight_pct=100*group.rating_weight.sum()/legal.rating_weight.sum()))
    pd.concat(rankings).to_csv(OUTPUT/'bowling_phase_experiment_rankings.csv',index=False)
    pd.DataFrame(profiles).to_csv(OUTPUT/'bowling_phase_experiment_exposure.csv',index=False)
    print('Reference SDs:', scales.to_dict(), 'implied runs per wicket:', implied, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bowling-tradeoff',action='store_true')
    parser.add_argument('--pressure-power',type=float,default=2)
    parser.add_argument('--save-deliveries',action='store_true',
                        help='Save chronological scored deliveries for ranking uncertainty audits.')
    parser.add_argument('--rankings-dir',type=type(OUTPUT),default=OUTPUT)
    args = parser.parse_args()
    args.rankings_dir.mkdir(parents=True,exist_ok=True)
    raw = pd.read_pickle(DEST/'features.pkl')
    seed = json.loads((DEST/'warm_history_seeds.json').read_text())
    for role in (('bowl',) if args.bowling_tradeoff else ('bat','bowl')):
        data = prepare_role(raw,seed,role)
        bat = role=='bat'
        if bat:
            data = data[data.wides.eq(0)].copy()
        data['countable'] = True if bat else data.wides.eq(0)&data.noballs.eq(0)
        data['y_runs'] = data.runs if bat else data.charged_runs
        data['y_wicket'] = (data.striker_out if bat else
            (data.is_wicket&data.dismissal_type.isin(WICKETS))).astype(int)
        year = data.match_date.str[:4].astype(int)
        reference = data[year.eq(2022)]
        cutoff = reference.loc[reference.countable,role+'_pressure'].quantile(.75)
        scales = reference[['y_runs','y_wicket']].std()
        evaluated = []
        for season in range(2023,2027):
            train, test = data[year.lt(season)], data[year.eq(season)].copy()
            assert not set(train.match_id)&set(test.match_id)
            if season==2026:
                replay = pd.read_csv(OUTPUT/f'{role}_2026_predictions.csv',
                                     usecols=KEYS+['expected_runs','expected_wicket'])
                test = test.merge(replay,on=KEYS,how='left',validate='one_to_one')
            else:
                model = fit_role(train,role,cutoff)
                test[['expected_runs','expected_wicket']] = predict(model,test,role)
            assert test[['expected_runs','expected_wicket']].notna().all().all()
            sign = 1 if bat else -1
            test['contribution'] = combine_contributions(
                sign*(test.y_runs-test.expected_runs),
                sign*(test.expected_wicket-test.y_wicket),
                *scales,'batting' if bat else 'bowling')
            evaluated.append(test)
            print(role,season,len(test),flush=True)
        evaluated = pd.concat(evaluated)
        if args.save_deliveries:
            evaluated.to_pickle(args.rankings_dir/f'{role}_scored_deliveries.pkl')
        if args.bowling_tradeoff:
            bowling_tradeoff(evaluated,cutoff,scales)
            continue
        board = rank(evaluated,role,cutoff,pressure_power=args.pressure_power)
        board.insert(0,'rank',range(1,len(board)+1))
        board.to_csv(args.rankings_dir/f'{role}_pressure_rankings.csv',index=False)
        print(board.head(10)[['rank','player','pressure_balls','pressure_matches','rating']].round(3).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
