"""Earlier-date, format-specific personal baselines for submission evidence."""
import numpy as np
import pandas as pd
from audit import OUT
from ipl_pressure import KEYS

def main():
    source = OUT/'combined_replay'
    dest = OUT/'submission'
    dest.mkdir(exist_ok=True)
    pressure = pd.read_csv(source/'combined_pressure_trial.csv.gz')
    ref = pd.read_pickle(source/'features.pkl')
    ref = ref[ref.match_date.str.startswith('2022')].merge(
        pressure[KEYS+['bat_candidate','bowl_candidate']],on=KEYS,validate='one_to_one')
    for role, player in [('bat','striker_id'),('bowl','bowler_id')]:
        d = pd.read_pickle(source/f'{role}_scored_deliveries.pkl').merge(
            pressure[KEYS+[role+'_candidate']],on=KEYS,validate='one_to_one')
        legal = ref.wides.eq(0)&(True if role=='bat' else ref.noballs.eq(0))
        cutoff = ref.loc[legal,role+'_candidate'].quantile(.75)
        d['ordinary_sum'] = d.contribution.where(d[role+'_candidate'].lt(cutoff),0)
        d['ordinary_count'] = d[role+'_candidate'].lt(cutoff).astype(int)
        groups = [player,'source_format','match_date']
        daily = d.groupby(groups)[['ordinary_sum','ordinary_count']].sum().sort_index()
        prior = daily.groupby(level=[0,1]).cumsum()-daily
        values = prior.reindex(pd.MultiIndex.from_frame(d[groups]))
        # 120 delivery prior toward zero contextual contribution.
        d['usual_level'] = values.ordinary_sum.to_numpy()/(values.ordinary_count.to_numpy()+120)
        d['prior_ordinary_balls'] = values.ordinary_count.to_numpy()
        high = d[d[role+'_candidate'].ge(cutoff)].copy()
        high['response'] = high.contribution-high.usual_level
        rows=[]
        for pid,g in high.groupby(player):
            w=(g[role+'_candidate']/100)**3
            rows.append(dict(player_id=pid,high_pressure_mean=np.average(g.contribution,weights=w),
                earlier_usual_level=np.average(g.usual_level,weights=w),
                pressure_response=np.average(g.response,weights=w),
                baseline_coverage=np.average(g.prior_ordinary_balls.ge(60),weights=w)))
        board=pd.read_csv(source/f'{role}_pressure_trial_rankings.csv')
        board.merge(pd.DataFrame(rows),on='player_id',validate='one_to_one').to_csv(
            dest/f'{role}_selection_evidence.csv',index=False)
        print(role,'evidence saved',len(board))

if __name__=='__main__':
    main()
