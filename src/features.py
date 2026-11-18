"""Explicit allowlist of available-before-delivery features."""
import numpy as np
import pandas as pd

NUMERIC=['innings','overid','over_ball_num','prior_wickets','prior_recorded_bat_runs','prior_events','knock_out','year']
CATEGORICAL=['competition_type','venue_country','batting_type','bowling_type']

def prepare_features(deliveries):
    b=deliveries.sort_values(['match_date','match_id','innings','overid','over_ball_num']).reset_index(drop=True).copy()
    g=b.groupby(['match_id','innings'],sort=False)
    b['prior_wickets']=g.is_wicket.cumsum()-b.is_wicket.astype(int)
    b['prior_recorded_bat_runs']=g.runs.cumsum()-b.runs
    b['prior_events']=g.cumcount()
    b['competition_type']=np.where(b.match_format.eq('T20I'),'T20I','IPL')
    b['year']=pd.to_datetime(b.match_date).dt.year
    X=b[NUMERIC+CATEGORICAL].copy()
    X[NUMERIC]=X[NUMERIC].apply(pd.to_numeric,errors='coerce')
    X[CATEGORICAL]=X[CATEGORICAL].fillna('unknown').astype(str)
    return b,X

def assert_split(b,train,validation):
    if not train.any() or not validation.any():raise ValueError('Empty chronological partition')
    assert not set(b.loc[train,'match_id']) & set(b.loc[validation,'match_id'])
    assert b.loc[train,'match_date'].max() < b.loc[validation,'match_date'].min()
