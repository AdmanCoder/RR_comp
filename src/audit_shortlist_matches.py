"""Inspect match context and rank sensitivity for the selected candidate pair."""
import pandas as pd
from audit import DATA, OUT
from ipl_pressure import KEYS
from submission_selection import score

PLAYERS = {
    'bat': ('ebm', 'striker', 'Naman Dhir|Abdul Samad|Shreyas Iyer|Suryakumar Yadav'),
    'bowl': ('warm_cat', 'bowler', 'Kartik Tyagi|Maheesh Theekshana|Nathan Ellis|Prince Yadav'),
}

def main():
    source = OUT / 'v1_candidate_comparison'
    dest = OUT / 'shortlist_match_audit'
    dest.mkdir(exist_ok=True)
    fixtures = pd.read_csv(DATA/'matches.csv', usecols=['match_id','home_team','away_team'])
    pressure = pd.read_csv(OUT/'combined_replay/combined_pressure_trial.csv.gz',
                           usecols=KEYS+['bat_candidate','bowl_candidate'])
    reference = pd.read_pickle(OUT/'combined_replay/features.pkl')
    reference = reference[reference.match_date.str.startswith('2022')].merge(pressure,on=KEYS)
    for role, (model, position, names) in PLAYERS.items():
        pid = position+'_id'
        board = pd.read_csv(source/f'{role}_{model}_rankings.csv')
        targets = board[board.player.str.fullmatch(names)]
        assert len(targets)==4
        apps = pd.read_csv(source/f'{role}_{model}_appearances.csv')
        apps = apps[apps.balls.ge(6) & apps[pid].isin(targets.player_id)].copy()
        apps['impact'] = apps.adjusted_mean * apps.weight * apps.source_format.map({'IPL':1.,'T20I':.5})
        apps = apps.merge(fixtures,on='match_id',validate='many_to_one')
        data = pd.read_pickle(OUT/f'combined_replay/{role}_scored_deliveries.pkl')
        data = data.merge(apps[[pid,'match_id']],on=[pid,'match_id'],validate='many_to_one')
        predictions = pd.concat(pd.read_pickle(source/f'{role}_{year}_predictions.pkl')
                                for year in range(2023,2027))
        data = data.drop(columns=['expected_runs','expected_wicket']).merge(
            predictions[KEYS+[model+'_runs',model+'_wicket']],on=KEYS,validate='one_to_one')
        column = role+'_candidate'
        cutoff = reference.loc[reference.wides.eq(0)&((role=='bat')|reference.noballs.eq(0)),column].quantile(.75)
        data = data.merge(pressure[KEYS+[column]],on=KEYS,validate='one_to_one').sort_values(
            ['match_id','innings','overid','innings_ball_num'],kind='stable')
        data['high'] = data[column].ge(cutoff)
        full = data.groupby([pid,'match_id']).agg(total_balls=('countable','sum'),total_runs=('y_runs','sum'),
                                                   total_wickets=('y_wicket','sum'))
        high = data[data.high]
        context = high.groupby([pid,'match_id']).agg(
            innings=('innings','first'),start_over=('overid','first'),end_over=('overid','last'),
            start_score=('score','first'),start_wickets=('wickets','first'),needed=('needed','first'),
            balls_left=('balls_left','first'))
        apps = apps.join(full,on=[pid,'match_id']).join(context,on=[pid,'match_id'])
        phases = high.groupby([pid,'match_id','phase']).agg(
            balls=('countable','sum'),runs=('y_runs','sum'),expected_runs=(model+'_runs','sum'),
            wickets=('y_wicket','sum'),expected_wickets=(model+'_wicket','sum')).reset_index()
        phases.to_csv(dest/f'{role}_phases.csv',index=False)
        apps.sort_values([pid,'impact'],ascending=[True,False]).to_csv(dest/f'{role}_matches.csv',index=False)
        rows = []
        for target in targets.itertuples():
            group = apps[apps[pid].eq(target.player_id)]
            strongest = group.loc[group.impact.idxmax()]
            for label, removed in [('strongest',strongest.match_id),('worst_rank',None)]:
                trials = []
                for match_id in ([removed] if removed is not None else group.match_id):
                    remainder = group[group.match_id.ne(match_id)]
                    _,_,rating = score(remainder,ipl_multiplier=2)
                    changed = board.copy()
                    match = changed.player_id.eq(target.player_id)
                    changed.loc[match,'rating'] = rating
                    changed.loc[match,'repeat_credit'] = remainder.adjusted_mean.gt(0).sum()**2/len(remainder)
                    changed['score'] = .8*changed.rating.rank(pct=True)+.2*changed.repeat_credit.rank(pct=True)
                    ranked = changed.sort_values(['score','rating','player_id'],ascending=[False,False,True])
                    trials.append((ranked.player_id.tolist().index(target.player_id)+1, match_id))
                rank, match_id = max(trials)
                rows.append(dict(player=target.player,rank_before=target.rank,matches=len(group),
                                 positive=int(group.adjusted_mean.gt(0).sum()),test=label,
                                 removed_match=match_id,rank_after_fixed_cohort=rank,
                                 still_eligible=len(group)>10))
        pd.DataFrame(rows).to_csv(dest/f'{role}_rank_sensitivity.csv',index=False)

if __name__ == '__main__':
    main()
