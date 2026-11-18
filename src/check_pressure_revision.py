"""Compare a frozen prior formula with the revised formula, without fitting outcomes."""
import importlib.util
import numpy as np
import pandas as pd
from audit import OUT
from ipl_pressure import run, score_states
p=OUT/'pressure_revision'
spec=importlib.util.spec_from_file_location('prior',p/'formula_before.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
b=run(report=False)
prior=old.score_states(b.copy(),b.bat_personal,b.bowl_personal,personal_weight=.20)
rows=[]
for weight in [.05,.10,.15,.20]:
    scored=score_states(b.copy(),b.bat_personal,b.bowl_personal,personal_weight=weight)
    zero=score_states(b.copy(),b.bat_personal*0,b.bowl_personal*0,personal_weight=0)
    for role in ['bat','bowl']:
        gap=scored[role+'_pressure']-zero[role+'_pressure']
        rows.append(dict(weight=weight,role=role,mean_addition=gap.mean(),p95_addition=gap.quantile(.95),max_addition=gap.max()))
pd.DataFrame(rows).to_csv(p/'weight_sensitivity.csv',index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
# Dense controlled grid checks bounds, explicit cutoffs and ordinary demand order.
states=pd.DataFrame([(left,n,w) for left in range(1,121) for n in range(1,int(6*left+10)) for w in [0,5,9]],columns=['balls_left','needed','wickets'])
states['innings']=2;states['score']=100;states['knock_out']=0
z=pd.Series(0.,index=states.index)
g=score_states(states,z,z)
assert g[['bat_pressure','bowl_pressure']].apply(lambda x:x.between(0,100).all()).all()
cut=np.where(g.balls_left.le(6),30,np.where(g.balls_left.le(12),55,25*g.balls_left/6))
assert g.loc[g.needed>cut,['bat_competition','bowl_competition']].eq(0).all().all()
# Before the declared extreme zone, harder chases cannot lower batting pressure.
ordinary=g[g.needed.le(np.where(g.balls_left.gt(12),3*g.balls_left,np.where(g.balls_left.eq(1),6,np.where(g.balls_left.between(7,12),24+2.4*(g.balls_left-7),4*g.balls_left))))]
assert ordinary.groupby(['balls_left','wickets']).bat_pressure.diff().dropna().ge(-1e-9).all()
# Same-ball result fields must not influence computed pressure.
mut=b.copy();mut['runs']=999;mut['total_runs']=999
changed=score_states(mut,b.bat_personal,b.bowl_personal)
assert all(np.allclose(changed[r+'_pressure'],b[r+'_pressure']) for r in ['bat','bowl'])
# Specific previously troublesome real transitions, preserving actual personal deficit.
transitions=[]
for mid in [20247,31248]:
    g=b[b.match_id.eq(mid)&b.innings.eq(2)]
    for idx in g.index:
        if idx-1 not in g.index:continue
        delta=prior.loc[idx,'bat_pressure']-prior.loc[idx-1,'bat_pressure']
        if abs(delta)>40:
            transitions.append(dict(match_id=mid,from_needed=b.loc[idx-1,'needed'],from_balls=b.loc[idx-1,'balls_left'],to_needed=b.loc[idx,'needed'],to_balls=b.loc[idx,'balls_left'],old_from=prior.loc[idx-1,'bat_pressure'],old_to=prior.loc[idx,'bat_pressure'],new_from=b.loc[idx-1,'bat_pressure'],new_to=b.loc[idx,'bat_pressure']))
pd.DataFrame(transitions).to_csv(p/'boundary_transitions.csv',index=False)
print(pd.DataFrame(transitions).round(2).to_string(index=False))
# Preserve fresh complete traces for the same five review examples.
ids=[31250,31275,20247,50360,31248]
cols=['match_id','innings','balls_left','needed','score','wickets','bat_pressure','bowl_pressure','bat_personal','bowl_personal','chase_extreme_factor']
b[b.match_id.isin(ids)][cols].to_csv(p/'five_match_updated_traces.csv',index=False)
print('PASS: dense-grid bounds, strict cutoffs, ordinary-range batting ordering, current-outcome invariance;',len(states),'states')
