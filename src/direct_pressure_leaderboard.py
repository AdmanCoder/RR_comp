"""Direct IPL performance on deliveries above fixed historical pressure cutoffs."""
import numpy as np
import pandas as pd

from audit import OUT
from ipl_pressure import run
from ipl_ratings import WICKETS


def summarize(data, role, cutoff):
    batting = role == 'bat'
    player = 'striker_id' if batting else 'bowler_id'
    name = 'striker_name' if batting else 'bowler_name'
    high = data[data[role+'_pressure'].ge(cutoff)]
    matches = high.groupby([player,'match_id']).agg(
        runs=('credited_runs','sum'), balls=('countable','sum'), wickets=('credited_wicket','sum'))
    matches['run_rate'] = matches.runs / matches.balls.replace(0,np.nan)
    matches['wicket_rate'] = matches.wickets / matches.balls.replace(0,np.nan)
    board = matches.groupby(level=0).agg(
        runs=('runs','sum'), balls=('balls','sum'), wickets=('wickets','sum'),
        matches=('balls',lambda b: b.gt(0).sum()), match_run_rate_sd=('run_rate','std'),
        match_wicket_rate_sd=('wicket_rate','std'))
    board = board.join(data.groupby(player)[name].first().rename('player'))
    board['eligible'] = board.balls.ge(60) & board.matches.ge(8)
    board['runs_per_100_balls' if batting else 'economy'] = (100 if batting else 6)*board.runs/board.balls.replace(0,np.nan)
    board['dismissals_per_100' if batting else 'wickets_per_100'] = 100*board.wickets/board.balls.replace(0,np.nan)
    board['match_run_rate_sd'] *= 100 if batting else 6
    board['match_wicket_rate_sd'] *= 100
    return board.reset_index()


def main():
    frame = run(report=False)
    years = pd.to_datetime(frame.match_date).dt.year
    dest = OUT/'direct_pressure'
    dest.mkdir(exist_ok=True)
    info = []
    for role in ['bat','bowl']:
        data = frame.copy()
        batting = role=='bat'
        data['countable'] = data.wides.eq(0) & (batting | data.noballs.eq(0))
        data['credited_runs'] = data.runs if batting else data.runs+data.wides+data.noballs
        data['credited_wicket'] = data.is_wicket & (data.dismissed_player_id.eq(data.striker_id)
            if batting else data.dismissal_type.isin(WICKETS))
        reference = data.loc[years.eq(2022)&data.countable,role+'_pressure']
        sample = data.loc[years.gt(2022)]
        primary = 'runs_per_100_balls' if batting else 'economy'
        for quantile in [.70,.75,.80]:
            cutoff = reference.quantile(quantile)
            board = summarize(sample,role,cutoff).sort_values(
                ['eligible',primary,'balls'],ascending=[False,not batting,False])
            board.to_csv(dest/f'{role}_p{int(100*quantile)}.csv',index=False)
            high = sample.countable & sample[role+'_pressure'].ge(cutoff)
            info.append(dict(role=role,quantile=quantile,cutoff=cutoff,
                high_balls=high.sum(),share=high.sum()/sample.countable.sum(),
                eligible=board.eligible.sum()))
            if quantile==.75:
                print(role,board.loc[board.eligible].head(5).round(2).to_string(index=False))
    print(pd.DataFrame(info).to_string(index=False))
    pd.DataFrame(info).to_csv(dest/'thresholds.csv',index=False)

if __name__=='__main__':
    main()
