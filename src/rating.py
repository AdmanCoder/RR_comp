"""Transparent rating primitives. Not a substitute for verified input states."""
import numpy as np
import pandas as pd

def pressure(resources, competitive, personal, feasibility=1., knockout=False):
    parts=[np.asarray(x,dtype=float) for x in [resources,competitive,personal,feasibility]]
    if any(np.any((x<0)|(x>1)|~np.isfinite(x)) for x in parts):
        raise ValueError('Pressure components must be finite and between 0 and 1')
    r,c,u,g=parts
    multiplier=np.where(knockout,1.15,1.)
    return 100*np.minimum(1.,multiplier*(.35*r+.45*g*c+.20*u))

def recency_weight(age_days):
    age=np.asarray(age_days,dtype=float)
    if np.any(age<0):raise ValueError('Evaluation cutoff precedes observation')
    return .7**(age/(2*365.25))

def combine_contributions(scoring, wicket, scoring_scale, wicket_scale, role):
    if role not in ['batting','bowling']:raise ValueError('Unknown role')
    if min(scoring_scale,wicket_scale)<=0:raise ValueError('Training scales must be positive')
    a=.7 if role=='batting' else .6
    return a*np.asarray(scoring)/scoring_scale+(1-a)*np.asarray(wicket)/wicket_scale

def innings_rates(deliveries):
    """Input requires attributed player_id, signed contribution and verified pressure."""
    d=deliveries.copy()
    d['weight']=(d.pressure/100)**2
    d['weighted_contribution']=d.weight*d.contribution
    d['squared_weight']=d.weight**2
    out=d.groupby(['player_id','match_id','innings'],as_index=False).agg(
        weight=('weight','sum'),weighted_contribution=('weighted_contribution','sum'),
        squared_weight=('squared_weight','sum'),deliveries=('weight','size'))
    out=out[out.weight>0].copy()
    out['rate']=out.weighted_contribution/out.weight
    out['effective_deliveries']=out.weight**2/out.squared_weight
    return out

def assert_ready(audit):
    if not audit.get('score_reconstruction_verified',False):
        raise ValueError('Final score-dependent rankings blocked: score reconstruction is unverified')
