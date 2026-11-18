"""Descriptive pressure ablation; does not fit parameters or change production scores."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit import DATA, OUT
from ipl_pressure import run, score_states, historical_par

DEST = OUT / 'pressure_review'
DEST.mkdir(exist_ok=True)
b = run(report=False)
meta = pd.read_csv(DATA / 'matches.csv').set_index('match_id')
team_names = {}
for prefix in ['home', 'away']:
    team_names.update(dict(zip(meta[prefix+'_team_id'], meta[prefix+'_team'])))
zero = pd.Series(0., index=b.index)
simple = score_states(b.copy(), zero, zero, personal_weight=0)
for role in ['bat', 'bowl']:
    b[role+'_shared'] = simple[role+'_pressure']
    base = b[role+'_baseline']
    # Optional normalization of the remaining .35+.45=.80 component budget.
    b[role+'_rescaled'] = base + (100-base) * np.minimum(
        1, (simple[role+'_pressure']-base)/(100-base)/.8)

# Select contrasting examples by observed score/result only, never by pressure.
# Outcomes select examples retrospectively; they are not pressure inputs.
rows = []
for mid, g in b.groupby('match_id'):
    first = g[g.innings.eq(1)]
    chase = g[g.innings.eq(2)]
    if chase.empty: continue
    last = chase.iloc[-1]
    rows.append(dict(match_id=mid, total=first.total_runs.sum(),
                     won=last.score+last.total_runs >= first.total_runs.sum()+1,
                     remaining=last.balls_left-int(last.wides==0 and last.noballs==0),
                     shortfall=first.total_runs.sum()+1-last.score-last.total_runs,
                     early_wickets=first.loc[first.balls_left.ge(84),'wickets'].max()))
s = pd.DataFrame(rows).set_index('match_id')
chosen = []
def pick(label, candidates, sort, ascending):
    candidates = candidates.drop(index=[m for _,m in chosen], errors='ignore')
    mid = int(candidates.sort_values(sort, ascending=ascending, kind='stable').index[0])
    chosen.append((label, mid))
pick('Tight successful chase', s[s.won & s.remaining.le(1)], ['remaining','total'], [True,False])
pick('Comfortable chase', s[s.won & s.total.ge(140)], ['remaining','total'], [False,False])
pick('Failed chase', s[~s.won & s.remaining.eq(0)], ['shortfall'], [False])
pick('First-innings collapse', s, ['early_wickets','total'], [False,True])
pick('High-scoring match', s, ['total'], [False])

summary=[]
fig, axs=plt.subplots(5,2,figsize=(17,19), sharey=True)
for row,(label,mid) in enumerate(chosen):
    match=b[b.match_id.eq(mid)].copy()
    m=meta.loc[mid]
    for inn in [1,2]:
        g=match[match.innings.eq(inn)].copy()
        x=(120-g.balls_left)/6
        batting=team_names.get(g.batting_team_id.iloc[0],str(g.batting_team_id.iloc[0]))
        bowling=team_names.get(g.bowling_team_id.iloc[0],str(g.bowling_team_id.iloc[0]))
        ax=axs[row,inn-1]
        for role,color in [('bat','#1674b8'),('bowl','#d45a27')]:
            ax.plot(x,g[role+'_pressure'],color=color,lw=1,alpha=.5,label=f'{role}: current')
            ax.plot(x,g[role+'_shared'],color=color,lw=1.8,ls='--',label=f'{role}: situation only')
            summary.append(dict(category=label,match_id=mid,date=m.match_date,innings=inn,
                batting=batting,bowling=bowling,role=role,deliveries=len(g),
                current_median=g[role+'_pressure'].median(),shared_median=g[role+'_shared'].median(),
                current_max=g[role+'_pressure'].max(),shared_max=g[role+'_shared'].max(),
                mean_personal_addition=(g[role+'_pressure']-g[role+'_shared']).mean()))
        wicket=g[g.is_wicket]
        ax.scatter((120-wicket.balls_left)/6,np.full(len(wicket),3),marker='x',color='#333333',s=20)
        endscore=int(g.total_runs.sum()); endwk=int(g.is_wicket.sum())
        ax.set_title(f'{label} | {str(m.match_date)[:10]} | innings {inn}\n{batting} batting vs {bowling} bowling | {endscore}/{endwk}',fontsize=10)
        ax.set(xlim=(0,20),ylim=(0,100),xlabel='Overs completed before delivery',ylabel='Pressure index')
        ax.grid(alpha=.15)
        if row==0 and inn==1:ax.legend(fontsize=8,loc='upper left',ncol=2)
    match.to_csv(DEST/f'match_{mid}_deliveries.csv',index=False)
fig.suptitle('Current pressure versus situation-only pressure\nSame scale and chase rules; crosses show wicket deliveries. Scores use pre-delivery information.',fontsize=15)
fig.tight_layout(rect=[0,0,1,.96])
fig.savefig(DEST/'five_match_comparison.png',dpi=150)
plt.close(fig)
pd.DataFrame(summary).to_csv(DEST/'match_summary.csv',index=False)

stats=[]
for role in ['bat','bowl']:
    current=b[role+'_pressure']; shared=b[role+'_shared']; scaled=b[role+'_rescaled']
    gap=current-shared
    top=current.ge(current.quantile(.75))
    for variant,values in [('current',current),('situation_only',shared),('situation_rescaled',scaled)]:
        selected=values.ge(values.quantile(.75))
        stats.append(dict(role=role,variant=variant,median=values.median(),p75=values.quantile(.75),
            p90=values.quantile(.9),spearman_to_current=current.corr(values,method='spearman'),
            top_quartile_jaccard=(top&selected).sum()/(top|selected).sum(),
            mean_personal_addition=gap.mean(),p95_personal_addition=gap.quantile(.95),
            max_personal_addition=gap.max(),pct_addition_over_5=100*gap.gt(5).mean()))
pd.DataFrame(stats).to_csv(DEST/'cohort_comparison.csv',index=False)
# Distribution across all sampled deliveries, not player-balanced or a calibrated probability.
fig,axs=plt.subplots(1,2,figsize=(12,4))
for role,ax in zip(['bat','bowl'],axs):
    for variant,title in [('pressure','Current'),('shared','Situation only')]:
        ax.hist(b[role+'_'+variant],bins=np.arange(0,105,5),weights=np.full(len(b),100/len(b)),histtype='step',lw=2,label=title)
    ax.set(title=role,xlabel='Pressure index',ylabel='% of deliveries'); ax.legend(); ax.grid(alpha=.15)
fig.tight_layout(); fig.savefig(DEST/'cohort_distribution.png',dpi=150);plt.close(fig)

# Invariance checks: a delivery's outcome cannot change its pre-delivery score.
checks={}
mut=b.copy(); mut['runs']=99;mut['total_runs']=99;mut['is_wicket']=~mut.is_wicket
scored=score_states(mut,b.bat_personal,b.bowl_personal)
checks['score_function_ignores_current_outcome']=all(np.allclose(scored[r+'_pressure'],b[r+'_pressure']) for r in ['bat','bowl'])
cut=sorted(b.match_date.unique())[len(b.match_date.unique())//2]
mut=b.copy();mut.loc[mut.match_date.ge(cut),'score']+=1000
par=historical_par(mut,venue_weight=20)
checks['par_ignores_same_date_and_future_scores']=bool(np.allclose(par[b.match_date.le(cut)],b.loc[b.match_date.le(cut),'first_innings_par']))
checks['bounded_scores']=all(b[r+'_'+v].between(0,100).all() for r in ['bat','bowl'] for v in ['pressure','shared','rescaled'])
checks['removing_personal_never_increases_pressure']=all((b[r+'_shared']<=b[r+'_pressure']+1e-10).all() for r in ['bat','bowl'])
checks={k:bool(v) for k,v in checks.items()}
assert all(checks.values()),checks
(DEST/'checks.json').write_text(json.dumps(checks,indent=2))
print(pd.DataFrame(stats).round(3).to_string(index=False))
print(pd.DataFrame(summary).round(2).to_string(index=False))
print('Selected matches:', chosen)
print('Checks:',checks)
# A compact single-match view, including the requested pressure distribution.
mid=chosen[0][1]
g=b[b.match_id.eq(mid) & b.innings.eq(2)]
fig,axs=plt.subplots(1,2,figsize=(13,4.5))
for role,color,label in [('bat','#1674b8','RR batting'),('bowl','#d45a27','KKR bowling')]:
    axs[0].plot((120-g.balls_left)/6,g[role+'_pressure'],color=color,alpha=.45,lw=1)
    axs[0].plot((120-g.balls_left)/6,g[role+'_shared'],color=color,lw=2,label=label)
    axs[1].hist(g[role+'_shared'],bins=np.arange(0,105,10),weights=np.full(len(g),100/len(g)),histtype='step',lw=2,color=color,label=label)
axs[0].set(xlabel='Overs completed before delivery',ylabel='Pressure index',ylim=(0,100),title='Pressure through the chase')
axs[1].set(xlabel='Situation-only pressure index',ylabel='% of deliveries',title='Distribution within this chase')
for ax in axs:ax.legend();ax.grid(alpha=.15)
fig.suptitle('Rajasthan chase 224 vs Kolkata | 16 April 2024\nSolid lines: situation only; faint lines: current formula with personal deficit',fontsize=12)
fig.tight_layout();fig.savefig(DEST/'single_match_trace_distribution.png',dpi=150);plt.close(fig)
