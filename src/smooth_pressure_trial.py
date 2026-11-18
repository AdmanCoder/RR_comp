"""Compare smooth delivery weights within the frozen V1 eligible appearances."""
import numpy as np
import pandas as pd

from audit import OUT
from ipl_pressure import KEYS
from rating import recency_weight
from submission_selection import score


def main():
    source = OUT / 'combined_replay'
    comparison = OUT / 'v1_candidate_comparison'
    destination = OUT / 'smooth_pressure_trial'
    destination.mkdir(exist_ok=True)
    pressure = pd.read_csv(source / 'combined_pressure_trial.csv.gz',
                           usecols=KEYS + ['bat_candidate', 'bowl_candidate'])
    reference = pd.read_pickle(source / 'features.pkl')
    reference = reference[reference.match_date.str.startswith('2022')].merge(
        pressure, on=KEYS, validate='one_to_one')
    top = []
    for role, model, player in [('bat', 'ebm', 'striker_id'),
                                ('bowl', 'warm_cat', 'bowler_id')]:
        batting = role == 'bat'
        column = role + '_candidate'
        countable = reference.wides.eq(0) & (batting | reference.noballs.eq(0))
        cutoff = reference.loc[countable, column].quantile(.75)
        data = pd.read_pickle(source / f'{role}_scored_deliveries.pkl').merge(
            pressure[KEYS + [column]], on=KEYS, validate='one_to_one')
        predictions = pd.concat(pd.read_pickle(comparison / f'{role}_{year}_predictions.pkl')
                                for year in range(2023, 2027))
        runs, wicket = model + '_runs', model + '_wicket'
        data = data.merge(predictions[KEYS + [runs, wicket]], on=KEYS, validate='one_to_one')
        assert len(data) == len(predictions)
        dataset_2022 = reference[reference.wides.eq(0)] if batting else reference
        from final_ipl_model import WICKETS
        run_scale = (dataset_2022.runs if batting else
                     dataset_2022.runs + dataset_2022.wides + dataset_2022.noballs).std()
        wicket_scale = ((dataset_2022.is_wicket & dataset_2022.dismissed_player_id.eq(dataset_2022.striker_id)) if batting else
                        dataset_2022.is_wicket & dataset_2022.dismissal_type.isin(WICKETS)).astype(int).std()
        run_share, sign = (.7, 1) if batting else (.8, -1)
        contribution = sign * (run_share * (data.y_runs - data[runs]) / run_scale
                               + (1-run_share) * (data[wicket] - data.y_wicket) / wicket_scale)
        appearances = pd.read_csv(comparison / f'{role}_{model}_appearances.csv')
        dates = pd.to_datetime(appearances.match_date)
        appearances['weight'] = np.sqrt(appearances.balls) * recency_weight((dates.max()-dates).dt.days)
        baseline = pd.read_csv(comparison / f'{role}_{model}_rankings.csv')
        appearances = appearances[appearances.balls.ge(6) & appearances[player].isin(baseline.player_id)]
        match_keys = [player, 'match_id']
        for width in (3, 7):
            # Weight is 0.5 at the old cutoff and approaches 1 at high pressure.
            weights = 1 / (1 + np.exp(-(data[column] - cutoff) / width))
            weighted = data.assign(numerator=weights*contribution, denominator=weights)
            sums = weighted.groupby(match_keys)[['numerator', 'denominator']].sum()
            match_scores = appearances.join(sums, on=match_keys)
            match_scores['cutoff_score'] = match_scores.adjusted_mean
            match_scores['adjusted_mean'] = match_scores.numerator / match_scores.denominator
            match_scores['score_change'] = match_scores.adjusted_mean - match_scores.cutoff_score
            match_scores.to_csv(destination / f'{role}_width{width}_matches.csv', index=False)
            rows = []
            for pid, matches in match_scores.groupby(player):
                mean, floor, rating = score(matches, ipl_multiplier=2)
                positive = matches.adjusted_mean.gt(0).sum()
                rows.append(dict(player_id=pid, mean=mean, floor=floor, rating=rating,
                                 positive=positive, repeat_credit=positive**2/len(matches)))
            board = pd.DataFrame(rows).merge(baseline[['player_id','player','rank','matches']]
                .rename(columns={'rank':'cutoff_rank'}), on='player_id', validate='one_to_one')
            assert len(board) == len(baseline)
            board['score'] = .8*board.rating.rank(pct=True) + .2*board.repeat_credit.rank(pct=True)
            board = board.sort_values(['score','rating','player_id'], ascending=[False,False,True])
            board.insert(0, 'rank', range(1,len(board)+1))
            board.to_csv(destination / f'{role}_width{width}_rankings.csv', index=False)
            top.append(board.head(10).assign(role=role, width=width))
    pd.concat(top).to_csv(destination / 'top_tens.csv', index=False)


if __name__ == '__main__':
    main()
