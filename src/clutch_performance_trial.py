"""Test a shrunk estimate of above-expectation performance in decisive fixtures."""
import numpy as np
import pandas as pd
from audit import OUT
from exposure_sensitivity import rank

def main():
    source = OUT/'fixture_weight_trial'
    output = OUT/'clutch_performance_trial'
    output.mkdir(exist_ok=True)
    for role, model, pid in [('bat','ebm','striker_id'), ('bowl','warm_cat','bowler_id')]:
        base = pd.read_csv(OUT/f'v1_candidate_comparison/{role}_{model}_rankings.csv')
        clutch = pd.read_csv(source/f'{role}_appearances.csv').query('fixture_bonus >= 14').copy()
        clutch['evidence'] = np.sqrt(clutch.balls/12).clip(upper=1) * clutch.source_format.map({'IPL':1,'T20I':.5})
        clutch['credit'] = clutch.evidence * clutch.adjusted_mean
        g = clutch.groupby(pid).agg(clutch_matches=('match_id','size'),
                                    clutch_weight=('evidence','sum'), clutch_total=('credit','sum'),
                                    positive=('adjusted_mean',lambda x: (x>0).sum()))
        combined = base.rename(columns={'rank':'original_rank','positive':'overall_positive'}).join(g,on='player_id')
        columns = ['clutch_matches','clutch_weight','clutch_total','positive']
        combined[columns] = combined[columns].fillna(0)
        combined['clutch_rating'] = combined.clutch_total/(combined.clutch_weight+4)
        for share in [0,.15,.30]:
            board = rank(combined.assign(rating=(1-share)*combined.rating+share*combined.clutch_rating))
            if share == 0:
                assert board.player_id.tolist()==base.player_id.tolist()
            board.to_csv(output/f'{role}_{share:g}.csv',index=False)
            print(role,share, list(zip(board.player.head(10),board.original_rank.head(10),board.positive.head(10),board.clutch_matches.head(10))))
if __name__=='__main__':
    main()
