"""IPL pressure pilot using complete, unrevised matched fixtures."""
import numpy as np
import pandas as pd
from audit import DATA, OUT
from rating import pressure

KEYS = ['match_id', 'innings', 'overid', 'over_ball_num']

def prior_sum(values, groups):
    return values.groupby(groups).cumsum() - values

def historical_par(b, venue_weight=None):
    """Estimate first-innings par from earlier dates at the same ball count."""
    checkpoints = b.loc[b.innings.eq(1)].drop_duplicates(['match_id', 'balls_left'])

    def previous_totals(groups):
        daily = checkpoints.groupby(groups + ['match_date']).score.agg(['sum', 'count'])
        daily = daily.sort_index()
        prior = daily.groupby(level=groups).cumsum() - daily
        keys = pd.MultiIndex.from_frame(b[groups + ['match_date']])
        return prior.reindex(keys).fillna(0)

    league = previous_totals(['balls_left'])
    anchor = 1.5 * (120 - b.balls_left.to_numpy())
    par = (league['sum'].to_numpy() + 5 * anchor) / (league['count'].to_numpy() + 5)
    if venue_weight is None:
        return par

    venue = previous_totals(['balls_left', 'venue_id'])
    return (venue['sum'].to_numpy() + venue_weight * par) / (
        venue['count'].to_numpy() + venue_weight
    )


def score_states(b, bat_personal, bowl_personal, retention=0.35, personal_weight=0.10):
    left = b.balls_left
    chase = b.innings.eq(2)
    late = np.clip((60-left)/60, 0, 1)
    played = 120-left
    fixed_par = 1.5 * played
    par = b.get('first_innings_par', fixed_par)
    deficit = (par-b.score)/(played+12)
    bat_comp = np.clip(deficit/0.6, 0, 1)
    bowl_comp = np.clip(-deficit/0.6, 0, 1)
    required = b.needed/left
    bat_comp = np.where(chase, np.clip((required-0.65)/1.45, 0, 1), bat_comp)
    bowl_comp = np.where(chase, np.clip((2.5-required)/1.6, 0, 1), bowl_comp)
    # Preserve the earlier formula for comparison.
    old_feasibility = np.where(chase, np.clip((required-0.25)/0.65, 0, 1)
                               * np.clip((4-required)/1.5, 0, 1), 1)
    old_bowl_comp = bowl_comp.copy()
    old_bat_comp = bat_comp.copy()
    fixed_deficit = (fixed_par-b.score)/(played+12)
    old_bat_comp = np.where(chase, old_bat_comp, np.clip(fixed_deficit/0.6, 0, 1))
    old_bowl_comp = np.where(chase, old_bowl_comp, np.clip(-fixed_deficit/0.6, 0, 1))
    # Use the remaining-run cutoffs in the last two overs and required
    # strike rate before then. Only the competitive chase term is zeroed.
    # Keep demand increasing through four runs per ball. Beyond that,
    # taper the competitive term and phase urgency toward the agreed limits.
    # For one ball, a six is a normal scoring outcome and seven needs extras.
    limit = np.where(left <= 6, 30, np.where(left <= 12, 55, 25 * left / 6))
    # Across the last two overs, ease into the final-over demand limit.
    # Six and seven balls share the same taper. The peak then rises to 36 at 12 balls.
    taper_start = np.where(left.between(7, 12), 24 + 2.4 * (left - 7), 4 * left)
    # Join the penultimate-over curve back to the general curve over one over.
    taper_start = np.where(left.between(13, 18),
                           36 + 6 * (left - 12), taper_start)
    taper_start = np.where(left == 1, 6, taper_start)
    taper_end = np.maximum(taper_start + 1, np.minimum(limit, 6 * left + 1))
    taper_end = np.where(left.between(7, 12), np.minimum(taper_end, 30 + 4.8 * (left - 7)), taper_end)
    taper_end = np.where(left == 1, 8, taper_end)
    # Scale difficulty to the start of the taper rather than saturating
    # at 2.1 runs per ball. This preserves ordering throughout live chases.
    bat_comp = np.where(
        chase, np.clip((required - 0.65) / (taper_start / left - 0.65), 0, 1),
        bat_comp,
    )
    # Broaden the extreme-demand transition before the last two overs.
    # Keep the demand scale above unchanged: ordinary situations retain their
    # scores. Start fading at 18 rpo rather than squeezing the entire fade
    # between 24 and 25 rpo. Final-two-over and one-ball rules stay unchanged.
    fade_start = np.where(left > 12, 3 * left, taper_start)
    fade_progress = np.clip((taper_end - b.needed) / (taper_end - fade_start), 0, 1)
    # Smoothstep joins both ends with zero slope, avoiding linear corners.
    extreme = fade_progress ** 2 * (3 - 2 * fade_progress)
    b['chase_extreme_factor'] = np.where(chase, extreme, 1)
    beyond_reach = np.where(left <= 12, b.needed > limit, 6 * b.needed > 25 * left)
    low_rate = np.clip((required - 0.25) / 0.65, 0, 1)
    feasibility = np.where(chase, low_rate * extreme * (~beyond_reach), 1)
    # A live ball always carries basic execution pressure. Fade that
    # baseline with extreme demand, independently of the easy-chase discount.
    live_weight = np.where(chase, extreme * (~beyond_reach), 1)
    baseline = 8 * live_weight
    # In the last over, the bowler has fewer chances to recover from a
    # poor ball. The feasibility multiplier still dampens extreme chases.
    final_ball_stakes = np.clip((7 - left) / 6, 0, 1)
    # Defending a single requires a dot or wicket. Defending six leaves
    # more room: ordinary singles and boundaries cannot tie the match.
    # Scale final-over urgency by this margin as well as by time remaining.
    defensive_margin = 1 - 0.35 * np.clip((required - 1) / 5, 0, 1)
    bowl_comp = np.where(
        chase, np.maximum(bowl_comp, final_ball_stakes * defensive_margin),
        bowl_comp,
    )
    bat_resource = 1-(1-late)*(1-np.clip(b.wickets/8, 0, 1))
    bowl_resource = late*(0.55+0.45*np.clip(1-b.wickets/10, 0, 1))
    for role, resource, competitive, personal in [
        ('bat', bat_resource, bat_comp, bat_personal),
        ('bowl', bowl_resource, bowl_comp, bowl_personal),
    ]:
        b[role+'_original'] = pressure(resource, old_bat_comp if role == 'bat' else old_bowl_comp, personal, old_feasibility, b.knock_out)
        old_comp = old_bat_comp if role == 'bat' else old_bowl_comp
        old_resource = resource * np.where(chase, retention + (1-retention) * old_feasibility, 1)
        b[role+'_previous'] = pressure(old_resource, old_comp, personal, old_feasibility, b.knock_out)
        # Chase feasibility reduces urgency from the phase as well as the score.
        # Personal deficit still describes the player's own innings.
        resource = resource * np.where(chase, retention + (1-retention) * feasibility, 1)
        # A final wicket can end an easy chase before the runs are scored.
        if role == 'bat':
            last_wicket = chase & b.wickets.eq(9) & ~beyond_reach
            resource = np.where(last_wicket, np.maximum(resource, 0.65 * extreme), resource)
        b[role+'_resources'] = resource
        b[role+'_competition'] = competitive*feasibility
        b[role+'_personal'] = personal
        situational = pressure(
            resource, competitive, personal * personal_weight / 0.20,
            feasibility, b.knock_out,
        )
        b[role+'_baseline'] = baseline
        b[role+'_pressure'] = baseline + (1 - baseline / 100) * situational
    return b

def run(retention=0.35, personal_weight=0.10, report=True):
    outcomes = pd.read_csv(OUT / 'ipl_verified_extras.csv')
    balls = pd.read_csv(DATA / 'ball_by_ball.csv')
    meta = pd.read_csv(DATA / 'matches.csv')
    meta = meta[meta[['first_inn_revised_overs', 'second_inn_revised_overs',
                      'revised_target']].isna().all(axis=1)]
    meta = meta[~meta.result.str.contains(r'DLS|D/L|Duckworth|rain|abandon|no result',
                                         case=False, na=False)]
    balls = balls[balls.match_id.isin(outcomes.match_id) & balls.innings.isin([1, 2])]
    b = balls.merge(outcomes, on=KEYS, how='left', validate='one_to_one')
    complete = b.total_runs.notna().groupby(b.match_id).all()
    b = b[b.match_id.isin(complete[complete].index)].merge(
        meta[['match_id', 'match_date', 'knock_out', 'venue_id']], on='match_id', validate='many_to_one')
    b = b.sort_values(KEYS).reset_index(drop=True)
    legal = (b.wides.eq(0) & b.noballs.eq(0)).astype(int)
    groups = [b.match_id, b.innings]
    b['score'] = prior_sum(b.total_runs, groups)
    b['balls_left'] = 120 - prior_sum(legal, groups)
    b['wickets'] = prior_sum(b.is_wicket.astype(int), groups)
    target = b[b.innings.eq(1)].groupby('match_id').total_runs.sum() + 1
    b = b[b.balls_left.gt(0)].copy()
    b['needed'] = (b.match_id.map(target) - b.score).where(b.innings.eq(2))
    # Personal deficit uses prior events in the current innings and fixed anchors.
    bat_groups = [b.match_id, b.innings, b.striker_id]
    bowl_groups = [b.match_id, b.innings, b.bowler_id]
    faced = prior_sum(b.wides.eq(0).astype(int), bat_groups)
    bowled = prior_sum((b.wides.eq(0)&b.noballs.eq(0)).astype(int), bowl_groups)
    bat_runs = prior_sum(b.runs, bat_groups)
    charged = prior_sum(b.runs+b.wides+b.noballs, bowl_groups)
    bat_personal = np.clip((1.30*faced-bat_runs)/(faced+12), 0, 1)
    bowl_personal = np.clip((charged-(11/6)*bowled)/(bowled+12), 0, 1)
    b['first_innings_par'] = historical_par(b, venue_weight=20)
    b = score_states(b, bat_personal, bowl_personal, retention, personal_weight)
    left = b.balls_left
    chase = b.innings.eq(2)
    required = b.needed / left
    if not report:
        return b
    assert b[['bat_pressure','bowl_pressure']].apply(lambda s:s.between(0,100).all()).all()
    cases = {
        'Early collapse': b.innings.eq(1)&b.wickets.ge(4)&left.ge(84),
        'Tight final over': chase&left.between(4,6)&b.needed.between(7,10),
        'Comfortable final over': chase&left.between(4,6)&b.needed.le(2)&b.wickets.le(5),
    }
    cols = KEYS+['score','wickets','balls_left','needed','knock_out',
                 'bat_pressure','bowl_pressure']
    examples = pd.concat([b.loc[mask,cols].head(1).assign(scenario=label)
                          for label, mask in cases.items()])
    example_components = ['bat_resources', 'bat_competition', 'bat_personal',
                          'bowl_resources', 'bowl_competition', 'bowl_personal']
    examples = examples.join(b[example_components])
    assert len(examples) == len(cases)
    tight = examples.loc[examples.scenario.eq('Tight final over')].iloc[0]
    comfortable = examples.loc[examples.scenario.eq('Comfortable final over')].iloc[0]
    assert tight.bat_pressure > comfortable.bat_pressure
    assert tight.bowl_pressure > comfortable.bowl_pressure
    final = b[chase & left.eq(1) & b.needed.between(1, 9)]
    # A boundary finish should score above the previous formula on each
    # of these deliveries, regardless of the player's personal component.
    boundary_finish = final[final.needed.eq(4)]
    assert not boundary_finish.empty
    for role in ['bat', 'bowl']:
        assert boundary_finish[role+'_pressure'].gt(boundary_finish[role+'_previous']).all()
    final.groupby('needed').agg(
        deliveries=('match_id', 'size'),
        **{col: (col, 'mean') for col in ['bat_previous', 'bowl_previous',
                                          'bat_pressure', 'bowl_pressure']}
    ).to_csv(OUT/'ipl_final_ball_pressure.csv')
    trace_match = tight.match_id
    trace = b[b.match_id.eq(trace_match) & chase & left.le(12)]
    trace[cols + ['total_runs', 'is_wicket'] + example_components].to_csv(
        OUT/'ipl_pressure_ball_trace.csv', index=False)
    examples.to_csv(OUT/'ipl_pressure_examples.csv', index=False)
    # Compare close and comfortable finishes at identical balls/wickets remaining.
    close = chase & left.between(4, 12) & required.between(1.1, 1.7)
    easy = chase & left.between(4, 12) & required.le(.4)
    first = ~chase & left.between(84, 102)
    rows = []
    for role in ['bat', 'bowl']:
        for comparison, high, low, keys in [
            ('tight_vs_comfortable', close, easy, ['balls_left', 'wickets']),
            ('collapse_vs_healthy', first & b.wickets.ge(4),
             first & b.wickets.le(1), ['balls_left']),
        ]:
            for version in ['previous', 'pressure']:
                col = role + '_' + version
                groups = [b.loc[mask].groupby(keys)[col].agg(['mean','size'])
                          for mask in [high, low]]
                pairs = groups[0].join(groups[1], lsuffix='_high', rsuffix='_low',
                                       how='inner')
                pairs = pairs[(pairs.size_high >= 3) & (pairs.size_low >= 3)]
                gap = pairs.mean_high - pairs.mean_low
                rows.append(dict(role=role, comparison=comparison, version=version,
                                 strata=len(gap), mean_gap=gap.mean(),
                                 positive_strata=int(gap.gt(0).sum()),
                                 high_deliveries=int(pairs.size_high.sum()),
                                 low_deliveries=int(pairs.size_low.sum())))
    checks = pd.DataFrame(rows)
    checks.to_csv(OUT/'ipl_pressure_validation.csv', index=False)
    print(checks.to_string(index=False))
    print(f'{len(b):,} deliveries across {b.match_id.nunique()} matches')
    print(examples.to_string(index=False))
    print(b[['bat_pressure','bowl_pressure']].quantile([.5,.75,.9,.99]).to_string())

if __name__ == '__main__':
    run()
