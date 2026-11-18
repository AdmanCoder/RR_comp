"""Isolate the ranking impact of the combined pressure trial."""
import numpy as np
import pandas as pd
from audit import OUT
from ipl_pressure import KEYS
from ipl_ratings import rank

def main():
    dest = OUT/'combined_replay'
    trial = pd.read_csv(dest/'combined_pressure_trial.csv.gz')
    features = pd.read_pickle(dest/'features.pkl')
    reference = features[features.match_date.str.startswith('2022')].merge(
        trial[KEYS+['bat_candidate','bowl_candidate']],on=KEYS,validate='one_to_one')
    important = trial.loc[trial.fixture_bonus.ge(7) & trial.match_date.ge('2023'),'match_id'].unique()
    for role,player in [('bat','striker_id'),('bowl','bowler_id')]:
        field = role+'_pressure'
        d = pd.read_pickle(dest/f'{role}_scored_deliveries.pkl').merge(
            trial[KEYS+[role+'_candidate']],on=KEYS,validate='one_to_one')
        d[field] = d[role+'_candidate']
        legal = reference.wides.eq(0) & (True if role=='bat' else reference.noballs.eq(0))
        cutoff = reference.loc[legal,role+'_candidate'].quantile(.75)
        w = (d[field]/100)**3
        total = w.groupby([d[player],d.match_id,d.innings]).transform('sum')
        d['rating_weight'] = w/np.sqrt(total.where(total.gt(0),1))
        board = rank(d,role,cutoff,pressure_power=3)
        board['rank'] = np.arange(1,len(board)+1)
        old = pd.read_csv(dest/f'{role}_appearance.csv').set_index('player_id')['rank']
        board['old_rank'] = board.player_id.map(old)
        deletions = {}
        for match in important:
            reduced = rank(d[d.match_id.ne(match)],role,cutoff,pressure_power=3)
            deletions[match] = pd.Series(np.arange(1,len(reduced)+1),index=reduced.player_id)
        sensitivity = pd.DataFrame(deletions).reindex(board.player_id)
        board['worst_rank_without_one_important_match'] = sensitivity.max(axis=1).to_numpy()
        board['deletions_losing_eligibility'] = sensitivity.isna().sum(axis=1).to_numpy()
        board['deletions_outside_top_ten'] = (~sensitivity.le(10)).sum(axis=1).to_numpy()
        board.to_csv(dest/f'{role}_pressure_trial_rankings.csv',index=False)
        print(role,round(cutoff,2),len(board),len(important))
        print(board.head(10)[['player','rank','old_rank','worst_rank_without_one_important_match',
                             'deletions_outside_top_ten','deletions_losing_eligibility']].to_string(index=False))

if __name__ == '__main__':
    main()
