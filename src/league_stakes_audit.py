"""Conservative IPL qualification stakes from the pre-match points table."""
import pandas as pd
from audit import DATA, OUT

def main():
    matches = pd.read_csv(DATA/'matches.csv').merge(pd.read_csv(DATA/'competitions.csv'), on='comp_id')
    league = matches[matches.comp_name.str.fullmatch(r'IPL \d{4}') & matches.stage.eq('League')]
    rows = []
    for _, season in league.groupby('comp_id'):
        teams = set(season.home_team_id) | set(season.away_team_id)
        for match in season.itertuples():
            prior = season[season.match_date.lt(match.match_date)]
            points = dict.fromkeys(teams, 0)
            played = dict.fromkeys(teams, 0)
            for game in prior.itertuples():
                for team in [game.home_team_id, game.away_team_id]:
                    played[team] += 1
                    points[team] += 1 if pd.isna(game.win_team_id) else 2 * int(team == game.win_team_id)
            bonuses = []
            for team, name in [(match.home_team_id, match.home_team), (match.away_team_id, match.away_team)]:
                remaining = max(0, 14-played[team])
                rivals = [r for r in teams if r != team]
                maximum = points[team]+2*remaining
                floor = sorted((points[r] for r in rivals),reverse=True)[3]
                possible_above = sum(points[r]+2*max(0,14-played[r]) >= points[team] for r in rivals)
                alive = maximum >= floor and possible_above >= 4
                bonus = (7 if alive and maximum-2 < floor else
                         4 if alive and remaining <= 3 and abs(points[team]-floor)<=2 else 0)
                bonuses.append(bonus)
                rows.append(dict(match_id=match.match_id,date=match.match_date,season=match.comp_name,
                    team=name,points=points[team],remaining=remaining,rival_cutoff=floor,
                    win_out_ceiling=maximum,loss_ceiling=maximum-2,team_bonus=bonus))
    result = pd.DataFrame(rows)
    result['fixture_bonus'] = result.groupby('match_id').team_bonus.transform('max')
    verified = pd.read_pickle(OUT/'combined_replay/features.pkl')[['match_id']].drop_duplicates()
    result['rated'] = result.match_id.isin(verified.match_id) & ~result.date.str.startswith('2022')
    result.to_csv(OUT/'combined_replay/league_stakes_audit.csv',index=False)
    print(result[result.rated].drop_duplicates('match_id').fixture_bonus.value_counts().sort_index())
    print(result[result.rated & result.team_bonus.eq(7)].tail(8).to_string(index=False))

if __name__ == '__main__':
    main()
