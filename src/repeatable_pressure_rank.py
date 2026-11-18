"""Rank the contribution level players repeatedly reach under pressure."""
import numpy as np
import pandas as pd
from audit import OUT
from rating import recency_weight


def quantile(rates, weights, q):
    order = np.argsort(rates)
    rates, weights = rates[order], weights[order]
    return np.interp(q, (weights.cumsum() - weights / 2) / weights.sum(), rates)


def main(dest=None, feature_path=None, replay=None):
    dest = dest or OUT / 'repeatable_pressure'
    feature_path = feature_path or OUT / 'model_optimization/features.pkl'
    replay = replay or OUT / 'methodology_audit/replay'
    dest.mkdir(exist_ok=True)
    ref = pd.read_pickle(feature_path)
    ref = ref[ref.match_date.str.startswith('2022')]
    for role, player in [('bat', 'striker_id'), ('bowl', 'bowler_id')]:
        d = pd.read_pickle(replay / f'{role}_scored_deliveries.pkl')
        name = 'striker_name' if role == 'bat' else 'bowler_name'
        pressure = role + '_pressure'
        legal = ref.wides.eq(0) & (True if role == 'bat' else ref.noballs.eq(0))
        cutoff = ref.loc[legal, pressure].quantile(.75)
        end = pd.to_datetime(d.match_date.max())
        d = d[d[pressure].ge(cutoff)].copy()
        d['w'] = (d[pressure] / 100)**3
        d['t'] = d.w * d.contribution
        m = d.groupby([player, 'match_id', 'match_date']).agg(w=('w','sum'),
            total=('t','sum'), balls=('countable','sum'), name=(name,'first')).reset_index()
        # Require six countable pressure deliveries in a match for a repeatability observation.
        m = m[m.balls.ge(6)].copy()
        m['rate'] = m.total / m.w
        m['weight'] = np.sqrt(m.w) * recency_weight((end-pd.to_datetime(m.match_date)).dt.days)
        players = [g for _, g in m.groupby(player) if len(g)>=8 and g.balls.sum()>=60]
        for q in [.30, .35, .40, .45, .50]:
            anchor = quantile(m.rate.to_numpy(), m.weight.to_numpy(), q)
            def score(g):
                w = g.weight.to_numpy()
                eff = w.sum()**2 / (w**2).sum()
                value = quantile(g.rate.to_numpy(), w, q)
                return anchor+(value-anchor)*eff/(eff+8)
            board = pd.DataFrame([dict(player=g.name.iloc[0], pressure_matches=len(g),
                pressure_balls=g.balls.sum(), mean=np.average(g.rate,weights=g.weight),
                rating=score(g)) for g in players])
            if q == .40:
                scores = board.rating.to_numpy()
                worst = np.argsort(np.argsort(-scores))+1
                for match in m.match_id.unique():
                    changed = np.array([score(g[g.match_id.ne(match)]) if match in g.match_id.values
                                        else scores[i] for i,g in enumerate(players)])
                    worst = np.maximum(worst,np.argsort(np.argsort(-changed))+1)
                board['worst_deletion_rank'] = worst
            board = board.sort_values('rating',ascending=False)
            board.insert(0,'rank',range(1,len(board)+1))
            board.to_csv(dest / f'{role}_{int(q*100)}.csv',index=False)
            print(role,q,len(board),', '.join(board.head(10).player))
            if q == .40:
                print(board.head(10).round(3).to_string(index=False))

if __name__ == '__main__':
    main()
