"""Two simple consistency candidates; frozen ML and pressure, development only.
Noise subtraction assumes independent delivery residuals within an appearance.
It is an approximation, not an estimate of psychological or causal consistency.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from rating import recency_weight
from ipl_ratings import rank
from consistency_validation import future_profiles

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/consistency_alternatives'


def variation_components(rates,weights,mean_variances):
    rates=np.asarray(rates,dtype=float)
    a=np.asarray(weights,dtype=float);a=a/a.sum()
    mean=np.sum(a*rates)
    observed=np.sum(a*(rates-mean)**2)
    noise=np.sum(a*(1-a)*np.asarray(mean_variances,dtype=float))
    denom=1-np.sum(a*a)
    return dict(observed_variance=observed,estimated_noise=noise,
        noise_adjusted_sd=np.sqrt(max(0.,observed-noise)/denom) if denom>1e-12 else np.nan,
        downside=np.sqrt(np.sum(a*np.minimum(rates,0)**2)))


def profiles(data,role,cutoff):
    pid='striker_id' if role=='bat' else 'bowler_id'
    d=data.copy()
    d['w']=(d[role+'_pressure']/100)**3
    d['w2']=d.w**2
    d['wx']=d.w*d.contribution
    d['wx2']=d.w*d.contribution**2
    m=d.groupby([pid,'match_id','match_date']).agg(w=('w','sum'),w2=('w2','sum'),wx=('wx','sum'),wx2=('wx2','sum')).reset_index()
    m=m[m.w>0].copy()
    m['rate']=m.wx/m.w
    m['effective_balls']=m.w**2/m.w2
    denom=m.w-m.w2/m.w
    pooled=float(d.contribution.var())
    m['within_variance']=np.where(denom>1e-12,np.maximum(0,m.wx2-m.wx*m.wx/m.w)/np.maximum(denom,1e-12),pooled)
    m['mean_variance']=m.within_variance/m.effective_balls
    dates=pd.to_datetime(m.match_date)
    m['match_weight']=m.w*recency_weight((dates.max()-dates).dt.days)
    rows=[]
    for player,g in m.groupby(pid):
        parts=variation_components(g.rate,g.match_weight,g.mean_variance)
        parts.update(player_id=player,median_effective_balls=g.effective_balls.median(),
                     noise_clipped=parts['estimated_noise']>=parts['observed_variance'])
        rows.append(parts)
    b=rank(data,role,cutoff,pressure_power=3)
    b=b.merge(pd.DataFrame(rows),on='player_id',validate='one_to_one')
    np.testing.assert_allclose(b.match_sd**2,b.observed_variance,atol=1e-10)
    b['shrunk_mean']=b['mean']*b.pressure_matches/(b.pressure_matches+8)
    b['current']=b.rating
    b['downside_score']=b.shrunk_mean-.15*b.downside
    b['noise_adjusted_score']=b.shrunk_mean-.15*b.noise_adjusted_sd
    # Regularize noisy variance estimates, using the existing eight-match prior
    # strength rather than searching for a new coefficient. Fit on past data only.
    untruncated=(b.observed_variance-b.estimated_noise)/(1-1/b.effective_matches)
    pooled=max(0.,np.average(untruncated,weights=b.effective_matches-1))
    strength=b.effective_matches/(b.effective_matches+8)
    b['pooled_variance']=pooled
    b['pooled_noise_sd']=np.sqrt(strength*b.noise_adjusted_sd**2+(1-strength)*pooled)
    b['pooled_noise_score']=b.shrunk_mean-.15*b.pooled_noise_sd
    return b


def main():
    OUT.mkdir(exist_ok=True)
    raw=pd.read_pickle(ROOT/'outputs/model_optimization/features.pkl')
    records=[]; diagnostic=[]; details=[]; final=[]
    for role in ['bat','bowl']:
        d=pd.read_pickle(ROOT/f'outputs/methodology_audit/replay/{role}_scored_deliveries.pkl')
        ref=raw[raw.match_date.str.startswith('2022')]
        legal=ref.wides.eq(0) if role=='bat' else ref.wides.eq(0)&ref.noballs.eq(0)
        cutoff=ref.loc[legal,role+'_pressure'].quantile(.75)
        for year in [2024,2025,2026]:
            past=d[d.match_date.str[:4].astype(int)<year]
            future=d[d.match_date.str.startswith(str(year))]
            assert pd.to_datetime(past.match_date).max()<pd.to_datetime(future.match_date).min()
            b=profiles(past,role,cutoff)
            target=future_profiles(future,role)
            diagnostic.append(dict(role=role,year=year,eligible=len(b),noise_clipped_fraction=b.noise_clipped.mean(),
                raw_sd_vs_length=b.match_sd.corr(b.median_effective_balls,method='spearman'),
                adjusted_sd_vs_length=b.noise_adjusted_sd.corr(b.median_effective_balls,method='spearman')))
            for events,matches in [(30,3),(60,5)]:
                t=target[(target.future_events>=events)&(target.future_matches>=matches)]
                joined=b.merge(t,on='player_id',validate='one_to_one')
                for variant in ['current','downside_score','noise_adjusted_score','pooled_noise_score']:
                    selected=set(b.nlargest(10,variant).player_id)
                    observed=joined[joined.player_id.isin(selected)]
                    r=dict(role=role,year=year,min_events=events,min_matches=matches,variant=variant,players=len(joined),selected_observed=len(observed),
                        quality_correlation=joined[variant].corr(joined.future_mean,method='spearman'),
                        reliability_correlation=joined[variant].corr(joined.future_reliability,method='spearman'),
                        downside_correlation=joined[variant].corr(-joined.future_downside,method='spearman'))
                    for metric in ['future_mean','future_sd','future_downside','future_below_expected_share']:
                        r['selected_'+metric]=observed[metric].mean()
                    records.append(r)
                details.append(joined.assign(role=role,year=year,min_events=events,min_matches=matches))
        b=profiles(d,role,cutoff)
        for variant in ['current','downside_score','noise_adjusted_score','pooled_noise_score']:
            board=b.sort_values(variant,ascending=False).copy()
            board['rank']=np.arange(1,len(board)+1)
            final.append(board.assign(role=role,variant=variant))
    pd.DataFrame(records).to_csv(OUT/'future_validation.csv',index=False)
    pd.DataFrame(diagnostic).to_csv(OUT/'noise_diagnostics.csv',index=False)
    pd.concat(details).to_csv(OUT/'player_validation_details.csv',index=False)
    pd.concat(final).to_csv(OUT/'candidate_rankings.csv',index=False)
    r=pd.DataFrame(records)
    print(r[(r.year>=2025)&r.min_events.eq(30)].round(4).to_string(index=False))
    print(pd.DataFrame(diagnostic).round(4).to_string(index=False))

if __name__=='__main__':main()
