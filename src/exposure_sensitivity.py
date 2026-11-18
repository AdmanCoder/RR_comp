"""Test whether pressure exposure changes the reliability of the shortlist."""
import numpy as np
import pandas as pd

from audit import OUT
from submission_selection import score


def player_summary(matches, repeat, duration, margin):
    matches = matches.copy()
    if duration:
        matches['weight'] *= np.sqrt(matches.balls.clip(upper=24) / 12)
    rating = score(matches, ipl_multiplier=2)[2]
    if repeat:
        exposure = np.minimum(matches.balls / 12, 1)
        format_weight = np.where(matches.source_format.eq('IPL'), 1., .5)
        opportunity = exposure * format_weight
        success = opportunity * (matches.adjusted_mean / margin).clip(0, 1)
        repeat_credit = success.sum() ** 2 / opportunity.sum()
    else:
        repeat_credit = matches.adjusted_mean.gt(0).sum() ** 2 / len(matches)
    return rating, repeat_credit


def rank(board):
    board = board.copy()
    board['score'] = .8 * board.rating.rank(pct=True) + .2 * board.repeat_credit.rank(pct=True)
    board = board.sort_values(['score', 'rating', 'player_id'], ascending=[False, False, True])
    return board.assign(rank=np.arange(1, len(board) + 1))


def main():
    source = OUT / 'v1_candidate_comparison'
    destination = OUT / 'exposure_sensitivity'
    destination.mkdir(exist_ok=True)
    results, stability = [], []
    for role, model, player in [('bat', 'ebm', 'striker_id'), ('bowl', 'warm_cat', 'bowler_id')]:
        base = pd.read_csv(source / f'{role}_{model}_rankings.csv')
        apps = pd.read_csv(source / f'{role}_{model}_appearances.csv')
        qualified = apps[apps.balls.ge(6)]
        margin = qualified.loc[qualified.match_date.str.startswith('2023') &
                               qualified.adjusted_mean.gt(0), 'adjusted_mean'].median()
        groups = dict(tuple(qualified[qualified[player].isin(base.player_id)].groupby(player)))
        for label, repeat, duration in [('current', False, False), ('partial_repeat', True, False),
                                        ('longer_passages', False, True), ('both', True, True)]:
            board = base[['player_id', 'player', 'matches']].rename(columns={'matches': 'qualifying_matches'})
            board[['rating', 'repeat_credit']] = [
                player_summary(groups[pid], repeat, duration, margin) for pid in board.player_id]
            board = rank(board)
            if label == 'current':
                assert board.player_id.tolist() == base.player_id.tolist()
                np.testing.assert_allclose(board.score, base.score)
            results.append(board.assign(role=role, variant=label, repeat_margin=margin))
            for entry in board.itertuples():
                matches = groups[entry.player_id]
                if entry.rank > 10 and entry.player_id not in set(base.head(10).player_id):
                    continue
                ranks = []
                for idx in matches.index:
                    updated = board.copy()
                    updated.loc[updated.player_id.eq(entry.player_id), ['rating', 'repeat_credit']] = (
                        player_summary(matches.drop(idx), repeat, duration, margin))
                    ranks.append(int(rank(updated).loc[updated.player_id.eq(entry.player_id), 'rank'].iloc[0]))
                stability.append(dict(role=role, variant=label, player=entry.player, rank=entry.rank,
                                      worst_rank=max(ranks), top_ten_removals=sum(r <= 10 for r in ranks),
                                      matches=len(ranks), eligible_after_removal=len(ranks)>10))
    pd.concat(results).to_csv(destination / 'rankings.csv', index=False)
    pd.DataFrame(stability).to_csv(destination / 'match_removal.csv', index=False)


if __name__ == '__main__':
    main()
