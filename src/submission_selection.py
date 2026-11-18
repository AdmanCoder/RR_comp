"""Rank sustained quality in substantial high-pressure appearances."""
import numpy as np
import pandas as pd
from audit import DATA, OUT
from rating import recency_weight
from repeatable_pressure_rank import quantile


def score(matches, ipl_multiplier=1.):
    weights = matches.weight.to_numpy()
    format_factor = np.where(matches.source_format.eq('IPL'), 1., 1./ipl_multiplier)
    weights = weights * format_factor
    rates = matches.adjusted_mean.to_numpy()
    mean = np.average(rates, weights=weights)
    floor = quantile(rates, weights, .40)
    effective = weights.sum()**2 / np.square(weights).sum()
    effective *= np.average(format_factor, weights=weights)
    return mean, floor, (.70*mean + .30*floor)*effective/(effective+8)


def main(scope=None, minimum_matches=10, ipl_multiplier=1.):
    dest = OUT/'submission'
    fixtures = pd.read_csv(DATA/'matches.csv', usecols=['match_id','comp_id'])
    if scope:
        teams = {'South Africa','England','New Zealand','Australia','Pakistan','India'}
        metadata = pd.read_csv(DATA/'matches.csv',usecols=['match_id','home_team','away_team'])
        home, away = metadata.home_team.isin(teams), metadata.away_team.isin(teams)
        allowed = set(metadata.loc[(home & away) if scope=='both' else (home | away),'match_id'])
        verified = pd.read_csv(OUT/'verified_t20i/fixture_audit.csv')
        rated = verified.status.eq('verified') & verified.match_date.ge('2023')
        print(scope, 'retained rated T20I matches', verified.loc[rated,'match_id'].isin(allowed).sum(), 'of', rated.sum())
    suffix = (f'_{scope}' if scope else '') + f'_min{minimum_matches}'
    if ipl_multiplier != 1.:
        suffix += f"_ipl{ipl_multiplier:g}"
    lists = []
    for role, pid, name in [('bat','striker_id','striker_name'),('bowl','bowler_id','bowler_name')]:
        d = pd.read_csv(dest/f'{role}_pressure_appearances.csv').merge(fixtures,on='match_id',validate='many_to_one')
        if scope:
            d = d[d.source_format.eq('IPL') | d.match_id.isin(allowed)].copy()
        date = pd.to_datetime(d.match_date)
        d['weight'] = np.sqrt(d.balls)*recency_weight((date.max()-date).dt.days)
        rows = []
        for player, g in d[d.balls.ge(6)].groupby(pid):
            if len(g)<minimum_matches or g.balls.sum()<60:
                continue
            mean, floor, rating = score(g, ipl_multiplier)
            without_match = min(score(g.drop(index), ipl_multiplier)[2] for index in g.index)
            series_scores = [score(g[g.comp_id.ne(key)], ipl_multiplier)[2]
                             for key in g.loc[g.source_format.eq('T20I'),'comp_id'].unique()
                             if g.comp_id.ne(key).any()]
            rows.append(dict(player_id=player,player=g[name].iloc[0],matches=len(g),
                positive=int(g.adjusted_mean.gt(0).sum()),runs=g.runs.sum(),balls=g.balls.sum(),
                wickets=g.wickets.sum(),mean=mean,floor=floor,rating=rating,
                worst_without_match=without_match,
                worst_without_t20i_series=min(series_scores,default=np.nan),
                ipl_weight=g.loc[g.source_format.eq('IPL'),'weight'].sum() / (g.loc[g.source_format.eq('IPL'),'weight'].sum() + g.loc[g.source_format.eq('T20I'),'weight'].sum()/ipl_multiplier)))
        board = pd.DataFrame(rows)
        baseline = pd.read_csv(dest/f'{role}_selection_evidence.csv')
        board = board.merge(baseline[['player_id','pressure_response','baseline_coverage']],
                            on='player_id',how='left',validate='one_to_one')
        board = board.sort_values(['rating','pressure_response','player_id'],ascending=[False,False,True])
        board.insert(0,'rank',range(1,len(board)+1))
        for label in ['match','t20i_series']:
            board[f'worst_rank_without_{label}'] = [
                1+int(((board.rating>value)&board.player_id.ne(pid)).sum()) if pd.notna(value) else np.nan
                for pid,value in zip(board.player_id,board[f'worst_without_{label}'])]
        board.to_csv(dest/f'{role}_recommended_rankings{suffix}.csv',index=False)
        assert len(board)>=10 and board.player_id.is_unique
        lists.append(board.head(10).assign(role=role))
        print(role,len(board))
        print(board.head(10)[['rank','player','runs','balls','wickets','positive','matches','worst_rank_without_match','worst_rank_without_t20i_series']].to_string(index=False))
    pd.concat(lists).to_csv(dest/f'recommended_top_tens{suffix}.csv',index=False)


if __name__=='__main__':
    main()
