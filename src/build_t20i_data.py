"""Build verified regulation T20I deliveries and pre-delivery scoreboard states."""
import json
import re

import pandas as pd

from audit import DATA, OUT, ROOT
from ipl_pressure import KEYS, prior_sum


def main():
    dest = OUT / 'verified_t20i'
    dest.mkdir(exist_ok=True)
    competitions = pd.read_csv(DATA / 'competitions.csv')
    matches = pd.read_csv(DATA / 'matches.csv').merge(competitions, on='comp_id')
    matches = matches[matches.match_format.eq('T20I')].copy()
    balls = pd.read_csv(DATA / 'ball_by_ball.csv')
    balls = balls[balls.match_id.isin(matches.match_id) & balls.innings.isin([1, 2])]
    extras = pd.read_csv(OUT / 't20i_trial_verified_extras.csv', dtype={'json_match_id': str})
    balls = balls.merge(extras, on=KEYS, how='left', validate='one_to_one')
    grouped = dict(list(balls.groupby('match_id')))
    reasons, totals = {}, {}
    for match in matches.itertuples():
        g = grouped.get(match.match_id)
        if g is None:
            reason = 'no_supplied_deliveries'
        elif g.json_match_id.isna().any():
            reason = 'incomplete_linkage' if g.json_match_id.notna().any() else 'no_verified_linkage'
        elif any(pd.notna(value) for value in (match.first_inn_revised_overs,
                 match.second_inn_revised_overs, match.revised_target)) or re.search(
                     r'DLS|D/L|Duckworth|rain|abandon|no result', str(match.result), re.I):
            reason = 'revised_or_interrupted'
        elif g.json_match_id.nunique() != 1:
            reason = 'ambiguous_source'
        else:
            source = ROOT / 't20s_json (1)' / (g.json_match_id.iloc[0] + '.json')
            data = json.loads(source.read_text())
            innings = data['innings'][:2]
            reason = 'verified'
            if (len(innings) != 2 or set(g.innings) != {1, 2}
                    or data['info'].get('balls_per_over') != 6
                    or data['info'].get('overs') != 20
                    or data['info']['outcome'].get('method')
                    or any(i.get('target', {}).get('overs', 20) != 20 for i in innings)):
                reason = 'nonstandard_source'
            else:
                scores = []
                for number, inning in enumerate(innings, 1):
                    deliveries = [ball for over in inning['overs'] for ball in over['deliveries']]
                    actual = sum(ball['runs']['total'] for ball in deliveries)
                    local = g[g.innings.eq(number)]
                    if (inning.get('penalty_runs') or len(deliveries) != len(local)
                            or actual != local.total_runs.sum()):
                        reason = 'innings_reconciliation_failed'
                    scores.append(actual)
                if innings[1].get('target', {}).get('runs', scores[0] + 1) != scores[0] + 1:
                    reason = 'target_reconciliation_failed'
                margin = re.search(r'won by (\d+) runs?', str(match.result), re.I)
                if margin and scores[0] - scores[1] != int(margin[1]):
                    reason = 'reported_margin_conflict'
                if reason == 'verified':
                    totals[match.match_id] = scores
        reasons[match.match_id] = reason

    b = balls[balls.match_id.isin(totals)].sort_values(KEYS).reset_index(drop=True)
    legal = (b.wides.eq(0) & b.noballs.eq(0)).astype(int)
    groups = [b.match_id, b.innings]
    b['score'] = prior_sum(b.total_runs, groups)
    b['balls_left'] = 120 - prior_sum(legal, groups)
    # Retired hurt is a departure rather than a wicket lost.
    wicket = b.is_wicket.astype(bool) & ~b.dismissal_type.fillna('').str.contains(
        r'retir.*(?:hurt|not out)', case=False, regex=True)
    b['wickets'] = prior_sum(wicket.astype(int), groups)
    b['target'] = b.match_id.map({key: value[0] + 1 for key, value in totals.items()}).where(b.innings.eq(2))
    b['needed'] = b.target - b.score
    b['charged_runs'] = b.runs + b.wides + b.noballs
    assert b.balls_left.between(1, 120).all() and b.wickets.between(0, 9).all()
    assert b.loc[b.innings.eq(2), 'needed'].gt(0).all()
    assert b.total_runs.eq(b.runs + b.extras_runs).all()
    b = b.merge(matches[['match_id','comp_name','match_date','stage','knock_out',
                          'venue_id','home_team','away_team']], on='match_id', validate='many_to_one')
    b.to_pickle(dest / 'deliveries.pkl')
    b.to_csv(dest / 'deliveries.csv.gz', index=False)
    matches['status'] = matches.match_id.map(reasons)
    matches['first_innings_runs'] = matches.match_id.map({k:v[0] for k,v in totals.items()})
    matches['second_innings_runs'] = matches.match_id.map({k:v[1] for k,v in totals.items()})
    matches.to_csv(dest / 'fixture_audit.csv', index=False)
    counts = matches.groupby(['comp_name','stage','knock_out','status'], dropna=False).size()
    counts.rename('matches').reset_index().to_csv(dest / 'coverage.csv', index=False)
    by_team = pd.concat([matches.assign(team=matches[side]) for side in ['home_team', 'away_team']])
    by_team.groupby(['team', 'status'], dropna=False).size().rename('matches').reset_index().to_csv(
        dest / 'team_coverage.csv', index=False)
    print(matches.status.value_counts().to_string())
    print('Verified deliveries:', len(b))
    print(b.groupby([b.match_date.str[:4], 'knock_out']).match_id.nunique().to_string())


if __name__ == '__main__':
    main()
