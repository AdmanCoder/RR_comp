"""Compare bowling ratings against next-season pressure performance."""
import numpy as np
import pandas as pd
from audit import OUT
from exposure_sensitivity import rank
from submission_selection import score

def main():
    directory = OUT / 'bowling_wicket_context'
    apps = pd.read_csv(directory/'forward_appearances.csv').merge(
        pd.read_csv(OUT/'v1_candidate_comparison/bowl_warm_cat_appearances.csv')[
            ['bowler_id','match_id','adjusted_mean']], on=['bowler_id','match_id'], validate='one_to_one')
    # Use all eligible-by-ball appearances, including players without ten career matches.
    results = []
    for year in [2024, 2025, 2026]:
        dates = pd.to_datetime(apps.match_date)
        prior = apps[dates.dt.year.lt(year)].copy()
        future = apps[dates.dt.year.eq(year)]
        prior['weight'] = np.sqrt(prior.balls)*.7**((prior.match_date.pipe(pd.to_datetime).max()
                                                     -pd.to_datetime(prior.match_date)).dt.days/(2*365.25))
        for target in ['adjusted_mean','saved','wicket_runs']:
            actual = future.groupby('bowler_id').filter(lambda g: len(g)>=3).groupby('bowler_id')[target].mean()
            for name in ['current','runs_equivalent']:
                rows=[]
                for pid,g in prior.groupby('bowler_id'):
                    if len(g)<5: continue
                    if name=='runs_equivalent':
                        g=g.assign(adjusted_mean=(g.saved+g.wicket_runs)/g.events)
                    rows.append(dict(player_id=pid,rating=score(g,ipl_multiplier=2)[2],
                                     repeat_credit=g.adjusted_mean.gt(0).sum()**2/len(g)))
                board=rank(pd.DataFrame(rows)).set_index('player_id')
                paired=board.join(actual.rename('actual'),how='inner')
                results.append(dict(year=year,model=name,target=target,players=len(paired),
                                    spearman=paired.score.corr(paired.actual,method='spearman')))
    output=pd.DataFrame(results)
    output.to_csv(directory/'forward_validation.csv',index=False)
    print(output.round(3).to_string(index=False))

if __name__=='__main__':main()
