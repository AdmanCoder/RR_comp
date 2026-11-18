"""Provisional IPL ratings using expanding-year contextual benchmarks."""
import json
import hashlib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, brier_score_loss
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

from audit import OUT
from ipl_pressure import run
from rating import combine_contributions, recency_weight

FEATURES = ['innings', 'balls_left', 'wickets', 'score', 'needed', 'knock_out']
WICKETS = ['Caught', 'Bowled', 'LBW', 'Stump', 'Hit Wicket']


def contributions(balls, role):
    batting = role == 'bat'
    if batting:
        balls = balls[balls.wides.eq(0)].copy()
        runs = balls.runs
        wickets = balls.is_wicket & balls.dismissed_player_id.eq(balls.striker_id)
    else:
        runs = balls.runs + balls.wides + balls.noballs
        wickets = balls.is_wicket & balls.dismissal_type.isin(WICKETS)
    years = pd.to_datetime(balls.match_date).dt.year
    scored = []
    params = dict(interactions=2, outer_bags=1, validation_size=0,
                  max_rounds=300, max_bins=32, min_samples_leaf=80, n_jobs=2,
                  random_state=17)
    for year in sorted(years.unique())[1:]:
        train, test = years.lt(year), years.eq(year)
        x_train, x_test = balls.loc[train, FEATURES], balls.loc[test, FEATURES]
        scoring = ExplainableBoostingRegressor(**params).fit(x_train, runs[train])
        dismissal = ExplainableBoostingClassifier(**params).fit(x_train, wickets[train])
        assert pd.to_datetime(balls.loc[train, 'match_date']).max() < pd.to_datetime(balls.loc[test, 'match_date']).min()
        assert not set(balls.loc[train, 'match_id']) & set(balls.loc[test, 'match_id'])
        direction = 1 if batting else -1
        result = balls.loc[test].copy()
        result['expected_runs'] = scoring.predict(x_test)
        result['expected_wicket'] = dismissal.predict_proba(x_test)[:, 1]
        result['actual_runs'] = runs[test]
        result['actual_wicket'] = wickets[test].astype(int)
        result['evaluation_year'] = year
        result['train_end'] = balls.loc[train, 'match_date'].max()
        result['baseline_runs'] = runs[train].mean()
        result['baseline_wicket'] = wickets[train].mean()
        result['contribution'] = combine_contributions(
            direction * (runs[test] - scoring.predict(x_test)),
            direction * (dismissal.predict_proba(x_test)[:, 1] - wickets[test]),
            runs[train].std(), wickets[train].std(), 'batting' if batting else 'bowling')
        scored.append(result)
    return pd.concat(scored)


def rank(scored, role, cutoff, penalty=.15, pressure_power=2, evidence_adjusted=False):
    player = 'striker_id' if role == 'bat' else 'bowler_id'
    name = 'striker_name' if role == 'bat' else 'bowler_name'
    d = scored.copy()
    d['weight'] = d.get('rating_weight', (d[role+'_pressure']/100)**pressure_power)
    d['weighted'] = d.weight * d.contribution
    d['high'] = d[role+'_pressure'].ge(cutoff)
    d['ordinary'] = d.contribution.where(~d.high)
    if 'countable' in d:
        d['high'] &= d.countable
    baseline = d.groupby(player).ordinary.mean()
    match = d.groupby([player, 'match_id', 'match_date']).agg(
        total=('weighted', 'sum'), weight=('weight', 'sum'), high=('high', 'sum')).reset_index()
    match = match[match.weight.gt(0)].copy()
    match['rate'] = match.total/match.weight
    dates = pd.to_datetime(match.match_date)
    match['weight'] *= recency_weight((dates.max()-dates).dt.days)
    names = d.groupby(player)[name].first()
    rows = []
    for pid, g in match.groupby(player):
        exposure, games = int(g.high.sum()), int(g.high.gt(0).sum())
        if exposure < 60 or games < 8:
            continue
        mean = np.average(g.rate, weights=g.weight)
        sd = np.sqrt(np.average((g.rate-mean)**2, weights=g.weight))
        effective_matches = g.weight.sum()**2 / (g.weight**2).sum()
        evidence = min(games, effective_matches) if evidence_adjusted else games
        moderated = mean * evidence/(evidence+8)
        rows.append(dict(player_id=pid, player=names[pid], pressure_balls=exposure,
                         pressure_matches=games, mean=mean, ordinary_mean=baseline[pid],
                         match_sd=sd, effective_matches=effective_matches,
                         shrinkage_evidence=evidence, rating=moderated-penalty*sd))
    return pd.DataFrame(rows).sort_values('rating', ascending=False).reset_index(drop=True)


def main():
    from ipl_pressure import score_states
    base = run(report=False)
    dest = OUT / 'player_evaluation_v1'
    dest.mkdir(exist_ok=True)
    keys = ['match_id', 'innings', 'overid', 'over_ball_num']
    settings = [('default', base)]
    for label, retention, personal in [('personal_05', .35, .05),
                                       ('personal_15', .35, .15),
                                       ('retention_20', .20, .10),
                                       ('retention_50', .50, .10)]:
        settings.append((label, score_states(base.copy(), base.bat_personal,
                         base.bowl_personal, retention, personal)))
    metrics, stability = [], []
    for role in ['bat', 'bowl']:
        scores = contributions(base, role)
        pressure_col = role+'_pressure'
        scores.to_csv(dest/f'{role}_delivery_evaluation.csv.gz', index=False)
        for year, g in scores.groupby('evaluation_year'):
            metrics.append(dict(role=role, year=int(year), deliveries=len(g),
                runs_mae=mean_absolute_error(g.actual_runs,g.expected_runs),
                constant_runs_mae=mean_absolute_error(g.actual_runs,g.baseline_runs),
                wicket_brier=brier_score_loss(g.actual_wicket,g.expected_wicket),
                constant_wicket_brier=brier_score_loss(g.actual_wicket,g.baseline_wicket)))
        lists = {}
        for label, states in settings:
            # Calibrate each formula's exposure threshold on 2022 only.
            cutoff = states.loc[pd.to_datetime(states.match_date).dt.year.eq(2022), pressure_col].quantile(.75)
            sample = scores.drop(columns=pressure_col).merge(
                states[keys+[pressure_col]], on=keys, validate='one_to_one')
            lists[label] = rank(sample, role, cutoff)
            if label == 'default':
                for penalty in [0., .30]:
                    lists[f'penalty_{penalty}'] = rank(sample, role, cutoff, penalty)
        result = lists['default']
        # Raw high-pressure outcomes are companions, not replacements for context adjustment.
        cutoff = base.loc[pd.to_datetime(base.match_date).dt.year.eq(2022), pressure_col].quantile(.75)
        high = scores[scores[pressure_col].ge(cutoff)]
        player = 'striker_id' if role == 'bat' else 'bowler_id'
        raw = high.groupby(player).agg(raw_pressure_runs=('actual_runs','sum'),
                raw_pressure_wickets=('actual_wicket','sum'),
                raw_pressure_events=('actual_runs','size'))
        result = result.join(raw,on='player_id')
        result.to_csv(dest/f'{role}_provisional_ratings.csv', index=False)
        print(role, '2022 threshold', round(cutoff, 2), 'eligible', len(result), flush=True)
        print(result.head(10).round(3).to_string(index=False), flush=True)
        top = set(result.head(10).player_id)
        for label, alternative in lists.items():
            overlap = len(top & set(alternative.head(10).player_id))
            joined = result.set_index('player_id').rating.rank().to_frame('a').join(
                alternative.set_index('player_id').rating.rank().rename('b')).dropna()
            stability.append(dict(role=role,variant=label,top10_overlap=overlap,
                                  rank_correlation=joined.a.corr(joined.b,method='spearman')))
            print(label, overlap, '/ 10', flush=True)
    pd.DataFrame(metrics).to_csv(dest/'chronological_metrics.csv',index=False)
    pd.DataFrame(stability).to_csv(dest/'ranking_sensitivity.csv',index=False)
    manifest = dict(pressure_version='v1_personal10_smooth_extreme',
        pressure_sha256=hashlib.sha256(Path(__file__).with_name('ipl_pressure.py').read_bytes()).hexdigest(),
        training='Each evaluation year uses only earlier years; 2022 initializes training and thresholds.',
        status='Development evaluation, not a fresh holdout or final competition ranking.',
        scope='Complete unrevised matched IPL only; T20I not included.',
        features=FEATURES, personal_weight=.10, reliability_penalty=.15)
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
