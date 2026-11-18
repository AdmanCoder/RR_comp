"""Inspect the selected fixture and scoreboard candidates together."""
import numpy as np
import pandas as pd
from audit import DATA, OUT
from ipl_pressure import KEYS, score_states

def main():
    dest = OUT / 'combined_replay'
    data = pd.read_pickle(dest/'features.pkl')
    fixtures = pd.read_csv(DATA/'matches.csv')
    data = data.merge(fixtures[['match_id','stage','comp_id','match_number','home_team','away_team']],
                      on='match_id', validate='many_to_one')
    league = pd.read_csv(dest/'league_stakes_audit.csv').drop_duplicates('match_id')
    league_bonus = data.match_id.map(league.set_index('match_id').fixture_bonus).fillna(0.)
    eliminators = fixtures[fixtures.stage.eq('Eliminator')].groupby('comp_id').match_number.min()
    knockout = data.knock_out.eq(1)
    qualifier_one = (knockout & data.source_format.eq('IPL') & data.stage.eq('Qualifier')
                     & data.match_number.lt(data.comp_id.map(eliminators)))
    bonus = np.select([knockout & data.stage.eq('Final'), qualifier_one, knockout],
                      [20., 7., 14.], default=league_bonus)
    old = data[['bat_pressure','bowl_pressure']].copy()
    d = score_states(data.assign(knock_out=False),data.bat_personal,data.bowl_personal)
    d['knock_out'] = data.knock_out
    first = d.innings.eq(1)
    left = d.balls_left
    late = ((60-left)/60).clip(0,1)
    par_gap = (d.score-d.first_innings_par)/(.6*(132-left))
    live = d.bat_baseline/8
    low = ((d.needed/left-.25)/.65).clip(0,1)
    fade = live.gt(0)&left.between(3,18)&~first
    for role in ['bat','bowl']:
        bat = role=='bat'
        col = role+'_pressure'
        resource = (1-(1-late)*(1-(d.wickets/8).clip(upper=1)) if bat
                    else late*(.55+.45*(1-d.wickets/10)))
        cushion = (par_gap if bat else -par_gap).clip(0,1)
        middle = ((left-30)/30).clip(0,1)*((120-left)/36).clip(0,1)
        change = (32.2*((1-(1-late)*(1-d.wickets/9))*(1-.25*cushion)-resource) if bat
                  else -32.2*resource*.25*cushion+3*middle*(1-cushion))
        d.loc[first,col] += change[first]
        # Preserve residual execution pressure and personal pressure through the fade.
        residual = 100*(.35*.35*resource + .1*d[role+'_personal'])
        d.loc[fade,col] = ((d[col]+residual)/2 +
            (d[col]-residual)/(2*np.sqrt(live.where(live.gt(0)))))[fade]
        if not bat:
            d.loc[~first,col] += (8*(1-low)*live*d.wickets.eq(9))[~first]
        d[role+'_candidate'] = (d[col]+bonus*live).clip(0,100)
        d[role+'_original_pressure'] = old[col]
        assert d[role+'_candidate'].between(0,100).all()
    d['fixture_bonus'] = bonus
    fields = KEYS+['source_format','match_date','home_team','away_team','stage','fixture_bonus',
                  'score','wickets','balls_left','needed','bat_original_pressure','bat_candidate',
                  'bowl_original_pressure','bowl_candidate']
    d[fields].to_csv(dest/'combined_pressure_trial.csv.gz',index=False)
    cases = d[d.match_date.isin(['2024-05-18','2024-05-26','2025-09-28']) &
              ((d.innings.eq(1)&d.balls_left.isin([120,60,12])) |
               (d.innings.eq(2)&d.balls_left.isin([12,6,1])))]
    cases = cases.drop_duplicates(['match_id','innings','balls_left'])
    cases[fields].to_csv(dest/'combined_pressure_examples.csv',index=False)
    print(cases[['match_date','innings','score','wickets','balls_left','needed','fixture_bonus',
                 'bat_original_pressure','bat_candidate','bowl_original_pressure','bowl_candidate']].round(1).to_string(index=False))

if __name__ == '__main__':
    main()
