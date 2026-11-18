"""Source-internal extras reconstruction: evidence, ambiguity, no result backfilling.

Strike transitions constrain completed outcomes only. Their availability is the
NEXT delivery; never use them in the current delivery's pre-ball features.
Candidate upper endpoints assume ordinary scoring (no exceptional overthrows).
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from audit import DATA,OUT

GROUPS=['wide_even','wide_odd','wide_unknown','bye_even','bye_odd','bye_unknown']
LOWER=np.array([1,2,1,2,1,1.])
UPPER=np.array([5,4,5,4,3,4.])
POINT=np.array([1,2,1,4,1,1.])

def preball_states(raw):
    """Operational sensitivity states, NOT exact scores or statistical intervals.

    Fixed rule; no fitted weights or match results. A previous event's parity may
    use the current striker, but no event at/after the current ball enters its
    pre-ball state. First-innings totals are available only in the second innings.
    Targets here assume an unrevised match; callers must exclude revised matches.
    """
    b=infer_events(raw)
    b['point_total']=b.known_total+b.group.map(dict(zip(GROUPS,POINT))).fillna(0)
    keys=['match_id','innings']
    out=b[['match_id','innings','overid','over_ball_num']].copy()
    for label,col in [('estimate','point_total'),('low','ordinary_lower'),('high','ordinary_upper')]:
        out['score_'+label]=b.groupby(keys)[col].cumsum()-b[col]
        first=b[b.innings.eq(1)].groupby('match_id')[col].sum()+1
        out['unrevised_target_'+label]=b.match_id.map(first).where(b.innings.eq(2))
    out['legal_balls_before']=b.groupby(keys).legal_delivery.cumsum()-b.legal_delivery.astype(int)
    out['wickets_before']=b.groupby(keys).is_wicket.cumsum()-b.is_wicket.astype(int)
    out['prior_ambiguous_events']=b.group.ne('').astype(int).groupby([b.match_id,b.innings]).cumsum()-b.group.ne('').astype(int)
    out['score_is_estimated']=out.prior_ambiguous_events.gt(0)
    return out

def infer_events(b):
    b=b.sort_values(['match_id','innings','overid','over_ball_num']).reset_index(drop=True).copy()
    g=b.groupby(['match_id','innings'],sort=False)
    ns=g.striker_id.shift(-1);nn=g.non_striker_id.shift(-1);no=g.overid.shift(-1)
    consecutive=g.innings_ball_num.shift(-1).eq(b.innings_ball_num+1)
    samepair=((ns.eq(b.striker_id)&nn.eq(b.non_striker_id))|(ns.eq(b.non_striker_id)&nn.eq(b.striker_id)))
    valid=(~b.is_wicket)&samepair&consecutive&(no.eq(b.overid)|no.eq(b.overid+1))
    b['running_parity']=np.where(valid,(ns.ne(b.striker_id)^no.ne(b.overid)).astype(int),-1)
    b['no_ball']=b.extras_type.isin(['Overstep','other'])
    b['legal_delivery']=~(b.no_ball|b.extras_type.eq('wide'))
    b['known_total']=b.runs+b.extras_2_runs+b.no_ball.astype(int)+5*b.extras_type.eq('Penalty').astype(int)
    b['group']=''
    for prefix,types in [('wide',['wide']),('bye',['byes','legbyes'])]:
        for parity,suffix in [(0,'even'),(1,'odd'),(-1,'unknown')]:
            b.loc[b.extras_type.isin(types)&b.running_parity.eq(parity),'group']=prefix+'_'+suffix
    b['ordinary_lower']=b.known_total+b.group.map(dict(zip(GROUPS,LOWER))).fillna(0)
    b['ordinary_upper']=b.known_total+b.group.map(dict(zip(GROUPS,UPPER))).fillna(0)
    return b

def design(b,m):
    amounts=b.groupby(['match_id','innings']).known_total.sum().unstack()
    counts=pd.crosstab([b.match_id,b.innings],b.group).reindex(columns=GROUPS,fill_value=0)
    delta=counts.xs(1,level=1).subtract(counts.xs(2,level=1),fill_value=0)
    d=m.merge(amounts.rename(columns={1:'known_1',2:'known_2'}),on='match_id').merge(delta,on='match_id')
    d['margin']=pd.to_numeric(d.result.str.extract(r'(?i)won by (\d+) runs?')[0],errors='coerce')
    revised=d.result.str.contains(r'DLS|D/L|Duckworth|rain',case=False,regex=True)
    d=d[d.margin.notna() & ~revised & d.revised_target.eq('') & d.first_inn_revised_overs.eq('') & d.second_inn_revised_overs.eq('') & d.known_2.notna()].copy()
    d['residual_margin']=d.margin-(d.known_1-d.known_2)
    d['year']=pd.to_datetime(d.match_date).dt.year
    return d

def run():
    b=pd.read_csv(DATA/'ball_by_ball.csv',keep_default_na=False)
    m=pd.read_csv(DATA/'matches.csv',keep_default_na=False)
    c=pd.read_csv(DATA/'competitions.csv',keep_default_na=False)
    m=m.merge(c,on='comp_id',validate='many_to_one')
    m=m[m.match_format.eq('T20I')|m.comp_name.str.match(r'^IPL \d{4}$')]
    b=infer_events(b[b.match_id.isin(m.match_id)&b.innings.isin([1,2])])
    # Validate no-ball interpretation on complete overs, independent of result.
    ov=b.groupby(['match_id','innings','overid']).agg(events=('runs','size'),legal=('legal_delivery','sum'),other=('extras_type',lambda x:x.eq('other').sum())).reset_index()
    ov=ov.merge(b.groupby(['match_id','innings']).overid.max().rename('last_over'),on=['match_id','innings'])
    complete=ov[ov.overid<ov.last_over]
    complete[complete.legal.ne(6)].to_csv(OUT/'extras_over_anomalies.csv',index=False)
    # Fit only aggregate missing-extra expectations on old training matches.
    # Results never select per-ball amounts or backfill a match's score.
    d=design(b,m)
    tr=d.year.le(2023)
    fit=lsq_linear(d.loc[tr,GROUPS].to_numpy(float),d.loc[tr,'residual_margin'].to_numpy(float),bounds=(LOWER,UPPER))
    weights=dict(zip(GROUPS,fit.x))
    rows=[]
    assumptions={'one_per_event':np.ones(6),'parity_minimum':LOWER,'parity_boundary':np.array([1,2,1,4,1,1]),'training_fitted_expectation':fit.x}
    for period,mask in [('train_2022_23',tr),('validation_2024_25',d.year.isin([2024,2025])),('previously_consumed_2026',d.year.eq(2026))]:
        for label,w in assumptions.items():
            error=d.loc[mask,GROUPS].to_numpy(float)@w-d.loc[mask,'residual_margin'].to_numpy(float)
            rows.append({'period':period,'method':label,'matches':len(error),'mae':float(np.mean(abs(error))), 'rmse':float(np.sqrt(np.mean(error**2))),'exact':int(np.isclose(error,0).sum()),'within_2_runs':int((abs(error)<=2).sum())})
    pd.DataFrame(rows).to_csv(OUT/'extras_hypothesis_metrics.csv',index=False)
    # Interval feasibility is a diagnostic only: generous ordinary-scoring domains
    # cannot certify actual amounts and must not be tightened using match outcomes.
    lo=b.groupby(['match_id','innings']).ordinary_lower.sum().unstack()
    hi=b.groupby(['match_id','innings']).ordinary_upper.sum().unstack()
    d=d.set_index('match_id')
    d['margin_lower']=lo[1]-hi[2];d['margin_upper']=hi[1]-lo[2]
    d['ordinary_domain_feasible']=d.margin.between(d.margin_lower,d.margin_upper)
    d[['result','margin','margin_lower','margin_upper','ordinary_domain_feasible']].to_csv(OUT/'extras_interval_reconciliation.csv')
    sparse=b[b.extras_type.ne('None')|b.extras_2_runs.ne(0)].copy()
    sparse['expected_total_train_2023']=sparse.known_total+sparse.group.map(weights).fillna(0)
    sparse[['match_id','innings','overid','over_ball_num','innings_ball_num','extras_type','extras1_runs','extras_2_runs','runs','running_parity','no_ball','legal_delivery','group','ordinary_lower','ordinary_upper','expected_total_train_2023']].to_csv(OUT/'extras_event_audit.csv',index=False)
    valid=b.running_parity.ge(0)
    sec=b.extras_2_runs.ne(0)&valid
    summary={'scope':'IPL/T20I regular innings','complete_overs':len(complete),'complete_overs_six_legal':int(complete.legal.eq(6).sum()),
      'other_events':int(b.extras_type.eq('other').sum()),
      'nonzero_secondary_with_parity':int(sec.sum()),'secondary_parity_agreement':int(((b.runs+b.extras_2_runs).mod(2).eq(b.running_parity)&sec).sum()),
      'missing_amount_events':int(b.group.ne('').sum()),'missing_amount_with_parity':int((b.group.ne('')&valid).sum()),
      'training_matches_for_expectations':int(tr.sum()),'fitted_expected_missing_runs':weights,
      'ordinary_margin_domain_feasible':int(d.ordinary_domain_feasible.sum()),'margin_checks':len(d),
      'exact_reconstruction':False,
      'leakage_policy':'Match results only diagnose hypotheses or fit aggregate expectations on earlier training matches. No individual match is forced to its final result. Strike-transition constraints become available at the next delivery, not before the current ball.',
      'limitations':['Parity does not distinguish 2 versus 4 byes or 1 versus 5 wides.', 'Typical-range bounds are not absolute legal maxima: exceptional overthrows can exceed them.', 'Missing extra magnitudes cannot be recovered uniquely from these columns.', 'Secondary run magnitude is supported; bye versus leg-bye attribution on no-balls is not identified.']}
    (OUT/'extras_findings.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':run()
