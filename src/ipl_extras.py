"""Verify IPL delivery alignment and export exact extras for verified overs."""
from collections import Counter, defaultdict
import json
import re

import pandas as pd

from audit import ROOT, DATA, OUT


def team(name):
    name = re.sub(r'[^a-z]', '', name.lower())
    return {
        'royalchallengersbangalore': 'royalchallengersbengaluru',
        'delhidaredevils': 'delhicapitals',
        'kingsxipunjab': 'punjabkings',
    }.get(name, name)


def event(delivery):
    extras = delivery.get('extras', {})
    return (delivery['runs']['batter'], bool(delivery.get('wickets')),
            bool(extras.get('wides')), bool(extras.get('noballs')))


def run():
    files = defaultdict(list)
    for path in sorted((ROOT / 'ipl_json (1)').glob('*.json')):
        data = json.loads(path.read_text())
        info = data['info']
        if info['dates'][0] >= '2022':
            key = (info['dates'][0], frozenset(map(team, info['teams'])))
            files[key].append((path.stem, data))

    matches = pd.read_csv(DATA / 'matches.csv')
    competitions = pd.read_csv(DATA / 'competitions.csv')
    matches = matches.merge(competitions, on='comp_id')
    matches = matches[matches.comp_name.str.fullmatch(r'IPL \d{4}')]
    balls = pd.read_csv(DATA / 'ball_by_ball.csv', keep_default_na=False)
    balls = balls[balls.match_id.isin(matches.match_id) & balls.innings.isin([1, 2])]
    balls = balls.sort_values(['match_id', 'innings', 'overid', 'over_ball_num'])
    by_match = dict(tuple(balls.groupby('match_id')))
    aligned = []
    votes = defaultdict(Counter)
    counts = Counter()
    for match in matches.itertuples():
        if match.match_id not in by_match:
            continue
        counts['royals_matches_with_deliveries'] += 1
        key = (match.match_date, frozenset(map(team, [match.home_team, match.away_team])))
        candidates = files[key]
        if len(candidates) != 1:
            continue
        counts['unique_fixture_matches'] += 1
        source_id, source = candidates[0]
        registry = source['info']['registry']['people']
        for innings_number, innings in enumerate(source['innings'][:2], 1):
            local = by_match[match.match_id]
            local = local[local.innings.eq(innings_number)]
            batting_team = match.home_team if local.empty or local.batting_team_id.iloc[0] == match.home_team_id else match.away_team
            if local.empty or team(batting_team) != team(innings['team']):
                continue
            external_overs = {o['over'] + 1: o['deliveries'] for o in innings['overs']}
            for over_number, over in local.groupby('overid'):
                deliveries = external_overs.get(over_number, [])
                if len(over) != len(deliveries):
                    continue
                pairs = list(zip(over.itertuples(), deliveries))
                if any((r.runs, r.is_wicket, r.extras_type == 'wide',
                        r.extras_type in ('Overstep', 'other')) != event(d)
                       for r, d in pairs):
                    continue
                block = []
                for row, delivery in pairs:
                    identities = [(registry[delivery[role]], getattr(row, local_role + '_id'),
                                   delivery[role], getattr(row, local_role + '_name'))
                                  for role, local_role in [('batter', 'striker'),
                                                          ('non_striker', 'non_striker'),
                                                          ('bowler', 'bowler')]]
                    for public_id, royals_id, _, _ in identities:
                        votes[public_id][royals_id] += 1
                    block.append((row, delivery, identities))
                aligned.append((source_id, block))

    resolved = {}
    for public_id, options in votes.items():
        top = options.most_common(2)
        royals_id, count = top[0]
        if count >= 6 and count / sum(options.values()) >= .98:
            resolved[public_id] = royals_id
    reverse = Counter(resolved.values())
    resolved = {key: value for key, value in resolved.items() if reverse[value] == 1}
    records, aliases = [], set()
    for source_id, block in aligned:
        if any(resolved.get(pub) != local for _, _, ids in block for pub, local, _, _ in ids):
            continue
        for row, delivery, ids in block:
            extras = delivery.get('extras', {})
            records.append(dict(match_id=row.match_id, innings=row.innings, overid=row.overid,
                                over_ball_num=row.over_ball_num, json_match_id=source_id,
                                total_runs=delivery['runs']['total'], extras_runs=delivery['runs']['extras'],
                                **{kind: extras.get(kind, 0) for kind in
                                   ('wides', 'noballs', 'byes', 'legbyes', 'penalty')}))
            aliases.update(ids)
    result = pd.DataFrame(records)
    keys = ['match_id', 'innings', 'overid', 'over_ball_num']
    joined = balls.merge(result, on=keys, how='left', validate='one_to_one')
    known = joined.total_runs.notna()
    missing = joined.extras_type.isin(['wide', 'byes', 'legbyes'])
    complete = joined.assign(linked=known).groupby('match_id').linked.all()
    result.to_csv(OUT / 'ipl_verified_extras.csv', index=False)
    pd.DataFrame(sorted(aliases), columns=['public_player_id', 'royals_player_id',
                                           'public_name', 'royals_name']).to_csv(
        OUT / 'ipl_verified_name_aliases.csv', index=False)
    counts.update(royals_deliveries=len(balls), verified_deliveries=len(result),
                  fully_linked_matches=int(complete.sum()), unknown_amount_events=int(missing.sum()),
                  linked_unknown_amount_events=int((known & missing).sum()),
                  verified_player_ids=len({alias[1] for alias in aliases}))
    (OUT / 'ipl_linkage_summary.json').write_text(json.dumps(counts, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    run()
