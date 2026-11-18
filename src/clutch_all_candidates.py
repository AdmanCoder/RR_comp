"""Test a shrunk estimate of above-expectation performance in decisive fixtures."""
import numpy as np
import pandas as pd
from audit import OUT
from exposure_sensitivity import rank

def main():
    source = OUT/'v1_candidate_comparison'
    stakes = pd.read_csv(OUT/'combined_replay/combined_pressure_trial.csv.gz',
                         usecols=['match_id','fixture_bonus']).drop_duplicates()
    assert stakes.match_id.is_unique
    tables = []
    output = OUT/'clutch_all_candidates'
    output.mkdir(exist_ok=True)
    models = ['selected_v1', 'ebm', 'warm_cat', 'pressure_cat', 'blended', 'chronological_choice']
    for role, pid in [('bat','striker_id'), ('bowl','bowler_id')]:
        for model in models:
            base = pd.read_csv(OUT/f'v1_candidate_comparison/{role}_{model}_rankings.csv')
            clutch = pd.read_csv(source/f'{role}_{model}_appearances.csv').merge(
                stakes, on='match_id', validate='many_to_one').query('balls >= 6 and fixture_bonus >= 14').copy()
            clutch['evidence'] = np.sqrt(clutch.balls/12).clip(upper=1) * clutch.source_format.map({'IPL':1,'T20I':.5})
            clutch['credit'] = clutch.evidence * clutch.adjusted_mean
            g = clutch.groupby(pid).agg(clutch_matches=('match_id','size'),
                                        clutch_weight=('evidence','sum'), clutch_total=('credit','sum'),
                                        positive=('adjusted_mean',lambda x: (x>0).sum()))
            combined = base.rename(columns={'rank':'original_rank','positive':'overall_positive'}).join(g,on='player_id')
            columns = ['clutch_matches','clutch_weight','clutch_total','positive']
            combined[columns] = combined[columns].fillna(0)
            combined['clutch_rating'] = combined.clutch_total/(combined.clutch_weight+4)
            board = rank(combined.assign(rating=.85*combined.rating+.15*combined.clutch_rating))
            if (role, model) in {('bat','ebm'), ('bowl','warm_cat')}:
                verified = pd.read_csv(OUT/f'clutch_performance_trial/{role}_0.15.csv')
                assert board.player_id.tolist() == verified.player_id.tolist()
            tables.append(board.assign(role=role, model=model))
    full = pd.concat(tables, ignore_index=True)
    full.to_csv(output/'rankings.csv', index=False)
    top = full[full['rank'].le(10)]
    top.to_csv(output/'top_tens.csv', index=False)
    for role, group in top.groupby('role'):
        print(role)
        print(group.pivot(index='rank',columns='model',values='player')[models].to_string())

if __name__ == '__main__':
    main()
