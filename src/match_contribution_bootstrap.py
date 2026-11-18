import sys
sys.path.insert(0,'src')
import numpy as np,pandas as pd
from audit import OUT
from submission_selection import score
rng=np.random.default_rng(17)
for role,model,pid in [('bat','ebm','striker_id'),('bowl','warm_cat','bowler_id')]:
 apps=pd.read_csv(OUT/f'v1_candidate_comparison/{role}_{model}_appearances.csv')
 base=pd.read_csv(OUT/f'v1_candidate_comparison/{role}_{model}_rankings.csv')
 events=pd.read_pickle(OUT/f'combined_replay/{role}_scored_deliveries.pkl')
 # Match totals need event counts in the exact qualifying passage, including illegal bowling events.
 from ipl_pressure import KEYS
 pressure=pd.read_csv(OUT/'combined_replay/combined_pressure_trial.csv.gz',usecols=KEYS+[role+'_candidate'])
 ref=pd.read_pickle(OUT/'combined_replay/features.pkl')
 ref=ref[ref.match_date.str.startswith('2022')].merge(pressure,on=KEYS)
 cut=ref.loc[ref.wides.eq(0)&((role=='bat')|ref.noballs.eq(0)),role+'_candidate'].quantile(.75)
 events=events.merge(pressure,on=KEYS)
 counts=events[events[role+'_candidate'].ge(cut)].groupby([pid,'match_id']).size().rename('events')
 apps=apps.join(counts,on=[pid,'match_id'])
 apps=apps[apps.balls.ge(6)&apps[pid].isin(base.player_id)].copy()
 apps['weight']/=np.sqrt(apps.balls)
 apps['adjusted_mean']*=apps.events/12
 rows=[]
 for player,g in apps.groupby(pid):
  g=g.reset_index(drop=True)
  draws=[score(g.iloc[rng.integers(0,len(g),len(g))],ipl_multiplier=2)[2] for _ in range(300)]
  rows.append(dict(player_id=player,rating=score(g,ipl_multiplier=2)[2],conservative_score=np.quantile(draws,.2),matches=len(g),positive=int(g.adjusted_mean.gt(0).sum())))
 board=pd.DataFrame(rows).merge(base[['player_id','player']],on='player_id').sort_values(['conservative_score','player_id'],ascending=[False,True])
 board.insert(0,'rank',range(1,len(board)+1))
 board.to_csv(OUT/f'match_contribution_trial/{role}_bootstrap.csv',index=False)
 print(role,board.head(10)[['rank','player','matches','positive']].to_string(index=False),flush=True)
