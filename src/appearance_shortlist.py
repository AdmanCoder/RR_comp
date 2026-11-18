"""Compare cubed delivery weighting with diminishing weight per appearance."""
import numpy as np
import pandas as pd
from audit import OUT
from ipl_ratings import rank

def main():
    dest = OUT / 'appearance_shortlist'
    dest.mkdir(exist_ok=True)
    ref = pd.read_pickle(OUT/'model_optimization/features.pkl')
    ref = ref[ref.match_date.str.startswith('2022')]
    for role, player in [('bat','striker_id'), ('bowl','bowler_id')]:
        d = pd.read_pickle(OUT/f'methodology_audit/replay/{role}_scored_deliveries.pkl')
        pressure = role+'_pressure'
        legal = ref.wides.eq(0) & (True if role=='bat' else ref.noballs.eq(0))
        cutoff = ref.loc[legal,pressure].quantile(.75)
        old = rank(d,role,cutoff,pressure_power=3)
        weights = (d[pressure]/100)**3
        total = weights.groupby([d[player],d.match_id,d.innings]).transform('sum')
        d['rating_weight'] = weights / np.sqrt(total.where(total.gt(0),1))
        board = rank(d,role,cutoff,pressure_power=3)
        board.insert(0,'rank',np.arange(1,len(board)+1))
        board['previous_rank'] = board.player_id.map(pd.Series(np.arange(1,len(old)+1),index=old.player_id))
        board.to_csv(dest/f'{role}_rankings.csv',index=False)
        print(board[(board['rank']<=10)|board.player.str.contains('Cummins|Arshdeep|Prasidh')][['player','rank','previous_rank','effective_matches']].round(1).to_string(index=False))

if __name__=='__main__':
    main()
