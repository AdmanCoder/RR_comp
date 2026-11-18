"""Small, explicit pressure candidate; original formula remains reproducible."""
import numpy as np
from ipl_pressure import score_states
from rating import pressure


def improved_states(data, powerplay_points=6.):
    if not 0 <= powerplay_points <= 10:
        raise ValueError('Powerplay premium must be between zero and ten points')
    b=score_states(data.copy(),data.bat_personal,data.bowl_personal)
    left=b.balls_left
    chase=b.innings.eq(2)
    live=b.bowl_baseline/8
    low=np.clip((b.needed/left-.25)/.65,0,1)
    # Execution urgency rises over the last two overs; a last wicket also
    # offers a direct route to ending the innings even in a low-rate chase.
    urgency=np.clip((12-left)/11,0,1)
    opportunity=np.maximum(urgency,(b.wickets.eq(9)).astype(float))
    old_gate=np.where(chase,low*live,1.)
    new_gate=np.where(chase,(low+(1-low)*opportunity)*live,1.)
    resource=np.clip((60-left)/60,0,1)*(.55+.45*np.clip(1-b.wickets/10,0,1))
    resource*=np.where(chase,.35+.65*new_gate,1.)
    competitive=np.clip((2.5-b.needed/left)/1.6,0,1)
    final=np.clip((7-left)/6,0,1)*(1-.35*np.clip((b.needed/left-1)/5,0,1))
    competitive=np.where(chase,np.maximum(competitive,final),0.)
    # Change only chase states whose old shared gate suppressed execution pressure.
    changed=chase & (new_gate>old_gate)
    situational=pressure(resource,competitive,b.bowl_personal*.5,new_gate,b.knock_out)
    b.loc[changed,'bowl_pressure']=(b.bowl_baseline+(1-b.bowl_baseline/100)*situational)[changed]
    b.loc[changed,'bowl_resources']=resource[changed]
    b.loc[changed,'bowl_competition']=(competitive*new_gate)[changed]
    # Smoothly remove the premium over overs five and six. No abrupt over-seven drop.
    x=np.clip((left-84)/12,0,1)
    premium=powerplay_points*x*x*(3-2*x)*live*np.where(b.knock_out,1.15,1.)
    for role in ['bat','bowl']:
        b[role+'_powerplay_premium']=premium
        b[role+'_pressure']=np.minimum(100,b[role+'_pressure']+premium)
        b.loc[chase & b.needed.le(0),role+'_pressure']=0.
    return b
