"""Audit extreme-demand transitions and last-wicket urgency."""
import numpy as np
import pandas as pd
from audit import OUT
from ipl_pressure import score_states

def main():
    states = pd.DataFrame([
        dict(innings=2, balls_left=b, wickets=w, score=150, needed=r, knock_out=False)
        for b in range(1,19) for w in [5,9] for r in range(1,78)
    ])
    zero = pd.Series(0.,index=states.index)
    old = score_states(states.copy(),zero,zero)
    # Soften the existing fade without moving the point where chase stakes vanish.
    # Apply only to the chase-dependent terms, retaining residual execution pressure.
    factor = old.bat_baseline/8
    softened = factor.pow(np.where(old.balls_left.le(2), 1., .5))
    ratio = softened / factor.replace(0,np.nan)
    for role in ['bat','bowl']:
        # Reconstruct the residual from the underlying phase term.
        late = (60-old.balls_left)/60
        resource = (1-(1-late)*(1-old.wickets.div(8).clip(upper=1)) if role=='bat'
                    else late*(.55+.45*(1-old.wickets/10)))
        residual = 100*.35*.35*resource
        candidate = residual + (old[role+'_pressure']-residual)*ratio
        old[role+'_softer_fade'] = candidate.where(factor.gt(0),old[role+'_pressure']).clip(0,100)
        # Mix the original and softened curves to spread the decline.
        old[role+'_blended_fade'] = (old[role+'_pressure'] + old[role+'_softer_fade']) / 2
    # A wicket can end a chase immediately when nine are down.
    low_gate = ((old.needed/old.balls_left-.25)/.65).clip(0,1)
    old['bowl_last_wicket_trial'] = (old.bowl_pressure +
        8*(1-low_gate)*factor*old.wickets.eq(9)).clip(0,100)
    assert old.filter(regex='softer_fade|last_wicket_trial').apply(lambda s:s.between(0,100).all()).all()
    old.to_csv(OUT/'combined_replay/chase_pressure_trial.csv',index=False)
    print(old[old.wickets.eq(5)&old.balls_left.eq(6)&old.needed.between(24,31)][
        ['needed','bat_pressure','bat_softer_fade','bat_blended_fade']].round(1).to_string(index=False))
    print(old[old.balls_left.eq(6)&old.needed.le(2)][
        ['needed','wickets','bat_pressure','bowl_pressure','bowl_last_wicket_trial']].round(1).to_string(index=False))
    for col in ['bat_pressure','bat_softer_fade','bat_blended_fade']:
        steepest = old.groupby(['balls_left','wickets'])[col].diff()
        row = old.loc[steepest.idxmin()]
        print(col,round(steepest.min(),1),int(row.balls_left),int(row.needed))
        print('Worst decline by balls left:', steepest.groupby(old.balls_left).min().round(1).to_dict())

if __name__ == '__main__':
    main()
