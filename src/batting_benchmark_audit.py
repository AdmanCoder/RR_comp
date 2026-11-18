"""Cross-check the frozen batting benchmark and trace influential innings."""
import numpy as np
import pandas as pd
from audit import OUT
from ipl_ratings import rank
from rating import recency_weight

def main():
    dest = OUT / 'batting_benchmark_audit'
    dest.mkdir(exist_ok=True)
    replay = OUT / 'methodology_audit/replay'
    balls = pd.read_pickle(replay / 'bat_scored_deliveries.pkl')
    raw = pd.read_pickle(OUT / 'model_optimization/features.pkl')
    raw = raw.loc[raw.wides.eq(0)].copy()
    reference = raw[raw.match_date.str.startswith('2022')]
    cutoff = reference.bat_pressure.quantile(.75)
    scale = .7 / reference.runs.std()
    for data in [raw, balls]:
        data['year'] = data.match_date.str[:4].astype(int)
        data['wicket_band'] = data.wickets // 3
        data['demand_band'] = pd.cut(data.needed / data.balls_left,
            [-np.inf, 1, 1.5, 2, np.inf], labels=False).fillna(-1)
    groups = ['innings', 'phase', 'wicket_band', 'demand_band']
    balls['pooled_runs'] = np.nan
    for year, test in balls.groupby('year'):
        train = raw[raw.year.lt(year)]
        assert train.match_date.max() < test.match_date.min()
        phase = train.groupby(groups[:2]).runs.mean()
        stats = train.groupby(groups).runs.agg(['sum','size'])
        stats['anchor'] = phase.reindex(stats.index.droplevel(groups[2:])).to_numpy()
        baseline = (stats['sum'] + 150 * stats.anchor) / (stats['size'] + 150)
        prediction = baseline.reindex(pd.MultiIndex.from_frame(test[groups])).to_numpy()
        fallback = phase.reindex(pd.MultiIndex.from_frame(test[groups[:2]])).to_numpy()
        balls.loc[balls.year.eq(year), 'pooled_runs'] = np.where(np.isnan(prediction), fallback, prediction)
    assert balls.pooled_runs.notna().all()
    boards = []
    for label, share in [('ml', 0.), ('pooled_25pct', .25), ('pooled_50pct', .50), ('pooled_runs', 1.)]:
        data = balls.assign(contribution=balls.contribution + share * scale * (balls.expected_runs-balls.pooled_runs))
        board = rank(data,'bat',cutoff,pressure_power=3)
        board['rank'] = np.arange(1,len(board)+1)
        boards.append(board.assign(benchmark=label))
    boards = pd.concat(boards)
    boards.to_csv(dest/'rank_comparison.csv',index=False)
    balls['weight'] = (balls.bat_pressure/100)**3 * recency_weight(
        (pd.to_datetime(balls.match_date.max())-pd.to_datetime(balls.match_date)).dt.days)
    balls['weighted_surplus'] = balls.weight*(balls.y_runs-balls.expected_runs)
    rows = []
    for (player, match, innings), g in balls.groupby(['striker_name','match_id','innings']):
        g = g.sort_values(['overid','over_ball_num'])
        first = g.iloc[0]
        rows.append(dict(player=player,match_id=match,date=first.match_date,innings=innings,
            first_faced_score=first.score,wickets_down=first.wickets,needed=first.needed,
            balls_left=first.balls_left,runs=g.y_runs.sum(),balls=len(g),dismissals=g.y_wicket.sum(),
            ml_runs=g.expected_runs.sum(),pooled_runs=g.pooled_runs.sum(),max_pressure=g.bat_pressure.max(),
            weight=g.weight.sum(),weighted_run_surplus=g.weighted_surplus.sum()))
    appearances = pd.DataFrame(rows)
    appearances['weight_share'] = appearances.weight / appearances.groupby('player').weight.transform('sum')
    appearances.to_csv(dest/'innings.csv',index=False)
    print(appearances[appearances.player.eq('Patrick Cummins')].nlargest(5,'weight').round(2).to_string(index=False))
    print(boards[boards.player.str.contains('Cummins|Klaasen|Parag')][['player','rank','mean','benchmark']].to_string(index=False))
    metrics = balls.assign(
        ml_squared_error=(balls.runs-balls.expected_runs)**2,
        pooled_squared_error=(balls.runs-balls.pooled_runs)**2)
    metrics = metrics.groupby('year')[['ml_squared_error','pooled_squared_error']].mean()
    metrics.to_csv(dest/'run_prediction_metrics.csv')
    print(metrics.to_string())

if __name__ == '__main__':
    main()
