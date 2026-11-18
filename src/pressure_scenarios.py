"""Controlled examples for reviewing the IPL formula."""
import pandas as pd
from audit import OUT
from ipl_pressure import score_states


def main():
    # Values are pre-ball states. Hold personal deficits at zero except where tested.
    cases = [
        ('First innings healthy start', 1, 36, 54, 1, None, 0, 0, 0),
        ('First innings slow start', 1, 36, 30, 1, None, 0, 0, 0),
        ('First innings collapse', 1, 36, 30, 4, None, 0, 0, 0),
        ('First innings fast start', 1, 36, 78, 1, None, 0, 0, 0),
        ('First innings death', 1, 114, 171, 1, None, 0, 0, 0),
        ('Chase 9 per over early', 2, 24, 36, 2, 144, 0, 0, 0),
        ('Chase 9 per over late', 2, 114, 171, 2, 9, 0, 0, 0),
        ('Same chase in knockout', 2, 114, 171, 2, 9, 1, 0, 0),
        ('Same chase eight down', 2, 114, 171, 8, 9, 0, 0, 0),
        ('Same chase personal deficit', 2, 114, 171, 2, 9, 0, .5, .5),
    ]
    for needed in [1, 4, 6, 12, 24, 36]:
        cases.append((f'{needed} needed from six', 2, 114, 180-needed, 2, needed, 0, 0, 0))
    for needed in [1, 2, 4, 6, 7]:
        cases.append((f'{needed} needed from one', 2, 119, 180-needed, 2, needed, 0, 0, 0))
    b = pd.DataFrame(cases, columns=['scenario', 'innings', 'balls_used', 'score',
                                    'wickets', 'needed', 'knock_out', 'bat_deficit', 'bowl_deficit'])
    b['balls_left'] = 120-b.balls_used
    scored = score_states(b, b.bat_deficit, b.bowl_deficit)
    columns = ['scenario', 'innings', 'balls_left', 'score', 'wickets', 'needed', 'knock_out']
    for role in ['bat', 'bowl']:
        columns += [role+'_'+field for field in ['resources', 'competition', 'personal', 'pressure']]
    scored[columns].round(3).to_csv(OUT/'ipl_pressure_scenarios.csv', index=False)
    print(scored[['scenario', 'bat_pressure', 'bowl_pressure']].round(1).to_string(index=False))


if __name__ == '__main__':
    main()
