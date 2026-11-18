"""Generate the submission rankings from saved pressure appearance evidence."""
import numpy as np
import pandas as pd

from audit import OUT
from ipl_pressure import KEYS
from rating import recency_weight
from submission_selection import score


def main():
    dest = OUT / 'submission'
    replay = OUT / 'combined_replay'
    reference = pd.read_pickle(replay / 'features.pkl')
    reference = reference[reference.match_date.str.startswith('2022')]
    trial = pd.read_csv(replay / 'combined_pressure_trial.csv.gz', usecols=KEYS+['bowl_candidate'])
    reference = reference.merge(trial, on=KEYS, validate='one_to_one')
    legal = reference.wides.eq(0) & reference.noballs.eq(0)
    cutoff = reference.loc[legal, 'bowl_candidate'].quantile(.75)
    runs_scale = (reference.runs + reference.wides + reference.noballs).std()
    from final_ipl_model import WICKETS
    wicket_scale = (reference.is_wicket & reference.dismissal_type.isin(WICKETS)).astype(int).std()
    deliveries = pd.read_pickle(replay / 'bowl_scored_deliveries.pkl').merge(
        trial, on=KEYS, validate='one_to_one')
    events = deliveries[deliveries.bowl_candidate.ge(cutoff)].groupby(
        ['bowler_id', 'match_id']).size().rename('events')

    results = []
    for role, player in [('bat','striker'), ('bowl','bowler')]:
        appearances = pd.read_csv(dest / f'{role}_pressure_appearances.csv')
        if role == 'bowl':
            appearances = appearances.join(events, on=['bowler_id', 'match_id'])
            assert appearances.events.gt(0).all()
            appearances['adjusted_mean'] = (
                .8 * appearances.runs_above_expectation/runs_scale
                + .2 * appearances.wickets_above_expectation/wicket_scale
            ) / appearances.events
        dates = pd.to_datetime(appearances.match_date)
        appearances['weight'] = np.sqrt(appearances.balls)*recency_weight((dates.max()-dates).dt.days)
        rows = []
        for pid, g in appearances[appearances.balls.ge(6)].groupby(f'{player}_id'):
            if len(g) < 10:
                continue
            positive = int(g.adjusted_mean.gt(0).sum())
            mean, floor, rating = score(g, ipl_multiplier=2)
            rows.append(dict(player_id=pid, player=g[f'{player}_name'].iloc[0],
                matches=len(g), positive=positive, pressure_balls=int(g.balls.sum()),
                pressure_runs=g.runs.sum(), pressure_wickets=g.wickets.sum(),
                mean=mean, floor=floor, rating=rating, repeat_credit=positive**2/len(g)))
        board = pd.DataFrame(rows)
        board['score'] = .8*board.rating.rank(pct=True)+.2*board.repeat_credit.rank(pct=True)
        board = board.sort_values(['score','rating','player_id'], ascending=[False,False,True])
        board.insert(0, 'rank', range(1,len(board)+1))
        assert len(board) >= 10 and board.player_id.is_unique
        board.to_csv(dest / f'final_{role}_rankings.csv', index=False)
        results.append(board.head(10).assign(role=role))
    final = pd.concat(results)
    final.to_csv(dest / 'final_top_tens.csv', index=False)
    print(final[['role','rank','player','positive','matches']].to_string(index=False))


if __name__ == '__main__':
    main()
