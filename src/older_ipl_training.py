"""Train on historical IPL JSON using features available in both sources."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from audit import OUT
from optimize_expectations import metrics, WICKETS, DEST


def load_older(start_date="2017-01-01"):
    rows = []
    for path in sorted((OUT.parent / 'ipl_json (1)').glob('*.json')):
        game = json.loads(path.read_text())
        info = game['info']
        date = info['dates'][0]
        regular = [i for i in game['innings'] if not i.get('super_over')]
        if not (start_date <= date < '2022-01-01' and info['overs'] == 20
                and len(regular) == 2 and info['outcome'].get('winner')
                and not info['outcome'].get('method')
                and all('penalty_runs' not in i for i in regular)
                and regular[1].get('target', {}).get('overs', 20) == 20):
            continue
        first_total = 0
        for number, inning in enumerate(regular, 1):
            score = legal = wickets = 0
            faced, workload = {}, {}
            for over in inning['overs']:
                for delivery in over['deliveries']:
                    extras = delivery.get('extras', {})
                    wide, no_ball = extras.get('wides', 0), extras.get('noballs', 0)
                    batter, bowler = delivery['batter'], delivery['bowler']
                    runs = delivery['runs']['batter']
                    dismissals = delivery.get('wickets', [])
                    if legal < 120:
                        rows.append(dict(match_id='json_' + path.stem, match_date=date,
                            innings=number, score=score, balls_left=120-legal, wickets=wickets,
                            needed=first_total+1-score if number == 2 else np.nan,
                            batter_faced=faced.get(batter, 0), bowler_balls=workload.get(bowler, 0),
                            runs=runs, wides=wide, noballs=no_ball,
                            bat_wicket=int(any(w['player_out']==batter for w in dismissals)),
                            bowl_wicket=int(any(w['kind'] in {'caught','caught and bowled',
                                'bowled','lbw','stumped','hit wicket'} for w in dismissals))))
                    score += delivery['runs']['total']
                    wickets += bool(dismissals)
                    legal_ball = not (wide or no_ball)
                    legal += legal_ball
                    faced[batter] = faced.get(batter, 0) + int(not wide)
                    workload[bowler] = workload.get(bowler, 0) + legal_ball
            if number == 1:
                first_total = score
    return pd.DataFrame(rows)


def main():
    recent = pd.read_pickle(DEST / 'features.pkl')
    recent['bat_wicket'] = recent.is_wicket & recent.dismissed_player_id.eq(recent.striker_id)
    recent['bowl_wicket'] = recent.is_wicket & recent.dismissal_type.isin(WICKETS)
    older = load_older()
    print(len(older), older.match_id.nunique(), flush=True)
    columns = ['innings','score','balls_left','wickets','needed','batter_faced',
               'bowler_balls','required_rpo','current_rpo']
    fields = list(dict.fromkeys(columns[:-2]+['match_date','match_id','runs','wides',
                                             'noballs','bat_wicket','bowl_wicket']))
    data = pd.concat([older[fields], recent[fields]], ignore_index=True)
    data['required_rpo'] = 6*data.needed/data.balls_left
    data['current_rpo'] = 6*data.score/(121-data.balls_left)
    rows = []
    for role in ['bat','bowl']:
        sample = data[data.wides.eq(0)].copy() if role == 'bat' else data.copy()
        year = pd.to_datetime(sample.match_date).dt.year
        sample['y_runs'] = sample.runs if role == 'bat' else sample.runs+sample.wides+sample.noballs
        sample['y_wicket'] = sample[role+'_wicket'].astype(int)
        for evaluation in [2024,2025,2026]:
            test = sample[year.eq(evaluation)]
            modern = sample[year.between(2022,evaluation-1)]
            rows.append(dict(role=role,year=evaluation,model='constant',**metrics(test,
                  np.full(len(test),modern.y_runs.mean()),np.full(len(test),modern.y_wicket.mean()))))
            for model, old_weight in [('recent_only',0),('older_quarter',.25),('older_full',1)]:
                train = modern if old_weight == 0 else sample[year.lt(evaluation)]
                assert train.match_date.max() < test.match_date.min()
                assert not set(train.match_id)&set(test.match_id)
                weights = np.where(train.match_date.lt('2022-01-01'),old_weight,1.)
                params = dict(max_iter=190,max_leaf_nodes=14,l2_regularization=10,
                              min_samples_leaf=125,learning_rate=.05,early_stopping=False)
                with threadpool_limits(limits=2):
                    r = HistGradientBoostingRegressor(**params).fit(train[columns],train.y_runs,
                        sample_weight=weights).predict(test[columns])
                    w = HistGradientBoostingClassifier(**params).fit(train[columns],train.y_wicket,
                        sample_weight=weights).predict_proba(test[columns])[:,1]
                rows.append(dict(role=role,year=evaluation,model=model,**metrics(test,r,w)))
            print(role,evaluation,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(DEST/'older_ipl_training_metrics.csv',index=False)

if __name__ == '__main__':
    main()
