"""Compare substantial match contributions with the existing rate ranking."""
import numpy as np
import pandas as pd
from audit import OUT
from submission_selection import score

def main():
    source = OUT/'v1_candidate_comparison'
    dest = OUT/'match_contribution_trial'
    dest.mkdir(exist_ok=True)
    outputs = []
    for role,model,pid in [('bat','ebm','striker_id'),('bowl','warm_cat','bowler_id')]:
        apps = pd.read_csv(source/f'{role}_{model}_appearances.csv')
        base = pd.read_csv(source/f'{role}_{model}_rankings.csv')
        apps = apps[apps.balls.ge(6)&apps[pid].isin(base.player_id)].copy()
        # Recover the existing recency factor, giving each match one observation.
        apps['weight'] /= np.sqrt(apps.balls)
        for power in [.5,1.]:
            d=apps.copy()
            d['adjusted_mean'] *= (d.balls/12)**power
            rows=[]
            for player,g in d.groupby(pid):
                mean,floor,rating=score(g,ipl_multiplier=2)
                without=min(score(g.drop(i),ipl_multiplier=2)[2] for i in g.index)
                rows.append(dict(player_id=player,matches=len(g),positive=int(g.adjusted_mean.gt(0).sum()),
                                 mean=mean,floor=floor,rating=rating,without_best=without))
            board=pd.DataFrame(rows).merge(base[['player_id','player','rank']].rename(columns={'rank':'old_rank'}),on='player_id')
            # Consistent contribution is the full score; repeat frequency is already represented by the distribution.
            board=board.sort_values(['rating','player_id'],ascending=[False,True])
            board.insert(0,'rank',range(1,len(board)+1))
            board['worst_removal_rank']=[1+int(((board.rating>v)&board.player_id.ne(pid)).sum()) for pid,v in zip(board.player_id,board.without_best)]
            board['eligible_after_removal']=board.matches.gt(10)
            board.to_csv(dest/f'{role}_power{power:g}.csv',index=False)
            outputs.append(board.assign(role=role,power=power))
    pd.concat(outputs).to_csv(dest/'all_rankings.csv',index=False)

if __name__=='__main__':main()
