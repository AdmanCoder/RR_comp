"""Test additional match-level credit for fixtures with pre-match stakes."""
import pandas as pd
from audit import OUT
from exposure_sensitivity import rank
from submission_selection import score


def main():
    output = OUT/'fixture_weight_trial'
    output.mkdir(exist_ok=True)
    stakes = pd.read_csv(OUT/'combined_replay/combined_pressure_trial.csv.gz',
                         usecols=['match_id','fixture_bonus']).drop_duplicates()
    assert stakes.match_id.is_unique
    boards = []
    for role, model, pid in [('bat','ebm','striker_id'),('bowl','warm_cat','bowler_id')]:
        source = OUT/'v1_candidate_comparison'
        base = pd.read_csv(source/f'{role}_{model}_rankings.csv')
        apps = pd.read_csv(source/f'{role}_{model}_appearances.csv').merge(
            stakes, on='match_id', validate='many_to_one')
        apps = apps[apps.balls.ge(6)&apps[pid].isin(base.player_id)]
        apps.assign(role=role).to_csv(output/f'{role}_appearances.csv',index=False)
        for extra in [0., .5, 1.]:
            rows = []
            for player, group in apps.groupby(pid):
                group = group.assign(weight=group.weight*(1+extra*group.fixture_bonus/20))
                positive = group.adjusted_mean.gt(0)
                clutch = group.fixture_bonus.ge(14)
                rows.append(dict(player_id=player, rating=score(group,ipl_multiplier=2)[2],
                                 repeat_credit=positive.sum()**2/len(group),
                                 clutch_matches=int(clutch.sum()),clutch_positive=int((clutch&positive).sum())))
            board = rank(pd.DataFrame(rows).merge(base[['player_id','player','matches']],on='player_id'))
            if extra == 0:
                assert board.player_id.tolist()==base.player_id.tolist()
            boards.append(board.assign(role=role,final_multiplier=1+extra))
    results = pd.concat(boards)
    results.to_csv(output/'rankings.csv',index=False)
    for key, group in results.groupby(['role','final_multiplier']):
        print(key, list(zip(group.player.head(10),group.clutch_positive.head(10),group.clutch_matches.head(10))))
    for role, group in results.groupby('role'):
        print(role, 'players with elimination/final evidence',sum(group[group.final_multiplier.eq(1)].clutch_matches.gt(0)))


if __name__ == '__main__':
    main()
