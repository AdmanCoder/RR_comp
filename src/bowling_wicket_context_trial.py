"""Estimate the future-runs value of a wicket from earlier-season match states."""
import numpy as np
import pandas as pd
from audit import OUT
from exposure_sensitivity import rank
from final_ipl_model import WICKETS
from ipl_pressure import KEYS
from rating import recency_weight
from submission_selection import score


def main():
    root = OUT/'combined_replay'
    output = OUT/'bowling_wicket_context'
    output.mkdir(exist_ok=True)
    raw = pd.read_pickle(root/'features.pkl')
    # Exclude the current ball's runs to avoid double-counting them.
    final = (raw.score + raw.total_runs).groupby([raw.match_id, raw.innings]).transform('max')
    raw['future'] = final - raw.score - raw.total_runs
    raw['credited'] = raw.is_wicket & raw.dismissal_type.isin(WICKETS)
    raw['left_group'] = (raw.balls_left-1)//12
    raw['wicket_group'] = raw.wickets//2
    raw['year'] = raw.match_date.str[:4]
    raw['pace_group'] = np.where(raw.innings.eq(1), raw.current_rpo, raw.required_rpo)
    raw['pace_group'] = pd.cut(raw.pace_group, [-np.inf, 7, 9, 11, np.inf], labels=False)
    raw['set_batter'] = raw.batter_faced.ge(12)
    pressure = pd.read_csv(root/'combined_pressure_trial.csv.gz',usecols=KEYS+['bowl_candidate'])
    raw = raw.merge(pressure,on=KEYS,validate='one_to_one')
    cutoff = raw.loc[raw.year.eq('2022') & raw.wides.eq(0) & raw.noballs.eq(0),'bowl_candidate'].quantile(.75)
    # Chases end at the target, so use first-innings scoring to estimate the
    # production lost to a wicket over the remaining scheduled balls.
    parts = []
    for year in sorted(raw.year.unique())[1:]:
        train = raw[raw.year.lt(year) & raw.innings.eq(1) & raw.wides.eq(0) & raw.noballs.eq(0)]
        cells = train.groupby(['left_group','wicket_group','credited']).future.agg(['mean','size']).unstack()
        gap = cells['mean'][False]-cells['mean'][True]
        support = cells['size'][True].fillna(0)
        values = (gap.fillna(0).clip(0,20)*support/(support+50)).rename('wicket_value')
        test = raw[raw.year.eq(year)&raw.bowl_candidate.ge(cutoff)].join(values,on=['left_group','wicket_group'])
        keys = ['left_group','wicket_group','pace_group','set_batter']
        detailed = train.groupby(keys+['credited']).future.agg(['mean','size']).unstack()
        difference = detailed['mean'][False]-detailed['mean'][True]
        weight = (detailed['size'].min(axis=1).fillna(0)/50).clip(upper=1)
        local = pd.DataFrame({'gap':difference.clip(0,20), 'weight':weight})
        test = test.join(local,on=keys)
        prior = test.wicket_value.fillna(0)
        fraction = test.weight.where(test.gap.notna(),0).fillna(0)
        test['wicket_value'] = (prior+fraction*(test.gap.fillna(prior)-prior)).where(test.balls_left.gt(1),0)
        pred = pd.read_pickle(OUT/f'v1_candidate_comparison/bowl_{year}_predictions.pkl')
        test = test.merge(pred[KEYS+['warm_cat_runs','warm_cat_wicket']],on=KEYS,validate='one_to_one')
        parts.append(test)
    d = pd.concat(parts)
    d['legal'] = d.wides.eq(0)&d.noballs.eq(0)
    d['saved'] = d.warm_cat_runs-d.runs-d.wides-d.noballs
    d['extra_wickets'] = d.credited.astype(int)-d.warm_cat_wicket
    d['wicket_runs'] = d.wicket_value*d.extra_wickets
    fields = ['bowler_id','match_id','match_date','source_format']
    apps = d.groupby(fields).agg(balls=('legal','sum'),events=('legal','size'),saved=('saved','sum'),
                                 extra_wickets=('extra_wickets','sum'),wicket_runs=('wicket_runs','sum')).reset_index()
    dates = pd.to_datetime(apps.match_date)
    apps['weight'] = np.sqrt(apps.balls)*recency_weight((dates.max()-dates).dt.days)
    base = pd.read_csv(OUT/'v1_candidate_comparison/bowl_warm_cat_rankings.csv')
    apps[apps.balls.ge(6)].to_csv(output/'forward_appearances.csv', index=False)
    apps = apps[apps.balls.ge(6)&apps.bowler_id.isin(base.player_id)]
    d.merge(apps[fields],on=fields,validate='many_to_one').groupby(
        fields+['phase']).agg(balls=('legal','sum'),runs_saved=('saved','sum'),
                              wickets_above=('extra_wickets','sum'),wicket_runs=('wicket_runs','sum'),
                              mean_wicket_value=('wicket_value','mean')).reset_index().to_csv(output/'spells_by_phase.csv',index=False)
    apps.to_csv(output/'appearances.csv',index=False)
    for multiplier in [0.5,0.75,1.,1.25,1.5]:
        groups = {}
        rows = []
        for pid,group in apps.groupby('bowler_id'):
            group = group.assign(adjusted_mean=(group.saved+multiplier*group.wicket_runs)/group.events)
            groups[pid] = group
            positive = int(group.adjusted_mean.gt(0).sum())
            rows.append(dict(player_id=pid,rating=score(group,ipl_multiplier=2)[2],
                             repeat_credit=positive**2/len(group),positive=positive,matches=len(group),
                             saved=group.saved.sum(),wickets_above=group.extra_wickets.sum()))
        board = rank(pd.DataFrame(rows).merge(base[['player_id','player','rank']].rename(columns={'rank':'old_rank'}),on='player_id'))
        for entry in board.head(10).itertuples():
            ranks = []
            for idx in groups[entry.player_id].index:
                remaining = groups[entry.player_id].drop(idx)
                revised = board.copy()
                mask = revised.player_id.eq(entry.player_id)
                revised.loc[mask,'rating'] = score(remaining,ipl_multiplier=2)[2]
                revised.loc[mask,'repeat_credit'] = remaining.adjusted_mean.gt(0).sum()**2/len(remaining)
                ranks.append(rank(revised).loc[mask,'rank'].iloc[0])
            board.loc[board.player_id.eq(entry.player_id),'worst_rank'] = max(ranks)
        board.to_csv(output/f'multiplier_{multiplier:g}.csv',index=False)
        print(multiplier,board.head(10)[['rank','player','old_rank','worst_rank','saved','wickets_above']].round(1).to_string(index=False))
    print(d.wicket_value.describe())

if __name__ == '__main__':
    main()
