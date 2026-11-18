"""Validate quality and observed consistency separately, using frozen expectations.
All scoring choices are predeclared comparisons, not an untouched-test selection.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from ipl_ratings import rank
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/consistency_validation'


def future_profiles(d,role):
    pid='striker_id' if role=='bat' else 'bowler_id'
    d=d.copy()
    d['w']=(d[role+'_pressure']/100)**3
    d['t']=d.w*d.contribution
    d['w2']=d.w**2
    m=d.groupby([pid,'match_id']).agg(w=('w','sum'),t=('t','sum'),w2=('w2','sum'),events=('w','size')).reset_index()
    m=m[m.w>0].copy()
    m['rate']=m.t/m.w
    m['effective_balls']=m.w**2/m.w2
    rows=[]
    for player,g in m.groupby(pid):
        mean=np.average(g.rate,weights=g.w)
        sd=np.sqrt(np.average((g.rate-mean)**2,weights=g.w))
        rows.append(dict(player_id=player,future_events=int(g.events.sum()),future_matches=len(g),
            future_mean=mean,future_sd=sd,future_reliability=mean-.15*sd,
            future_below_expected_share=np.average(g.rate<0,weights=g.w),
            future_downside=np.sqrt(np.average(np.minimum(g.rate,0)**2,weights=g.w)),
            median_match_effective_balls=g.effective_balls.median()))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    raw=pd.read_pickle(ROOT/'outputs/model_optimization/features.pkl')
    summaries=[]; details=[]; diagnostics=[]; boards=[]
    for role in ['bat','bowl']:
        d=pd.read_pickle(ROOT/f'outputs/methodology_audit/replay/{role}_scored_deliveries.pkl')
        ref=raw[raw.match_date.str.startswith('2022')]
        legal=ref.wides.eq(0) if role=='bat' else ref.wides.eq(0)&ref.noballs.eq(0)
        cutoff=ref.loc[legal,role+'_pressure'].quantile(.75)
        for year in [2024,2025,2026]:
            past=d[d.match_date.str[:4].astype(int)<year]
            later=d[d.match_date.str.startswith(str(year))]
            assert pd.to_datetime(past.match_date).max()<pd.to_datetime(later.match_date).min()
            targets=future_profiles(later,role)
            base=rank(past,role,cutoff,pressure_power=3)
            joined=base.merge(targets,on='player_id',validate='one_to_one')
            for min_events,min_matches in [(30,3),(60,5)]:
                cohort=joined[(joined.future_events>=min_events)&(joined.future_matches>=min_matches)].copy()
                diagnostics.append(dict(role=role,year=year,min_events=min_events,min_matches=min_matches,players=len(cohort),
                    past_sd_vs_future_sd=cohort.match_sd.corr(cohort.future_sd,method='spearman'),
                    past_mean_vs_future_mean=cohort['mean'].corr(cohort.future_mean,method='spearman'),
                    future_sd_vs_appearance_length=cohort.future_sd.corr(cohort.median_match_effective_balls,method='spearman')))
                for evidence in [False,True]:
                    n=np.minimum(cohort.pressure_matches,cohort.effective_matches) if evidence else cohort.pressure_matches
                    for penalty in [0.,.075,.15,.30]:
                        c=cohort.copy()
                        c['candidate_score']=c['mean']*n/(n+8)-penalty*c.match_sd
                        c=c.sort_values('candidate_score',ascending=False)
                        c['candidate_rank']=np.arange(1,len(c)+1)
                        # Select from past-eligible players BEFORE checking future appearances.
                        full_n=np.minimum(base.pressure_matches,base.effective_matches) if evidence else base.pressure_matches
                        full_scores=base['mean']*full_n/(full_n+8)-penalty*base.match_sd
                        selected=set(base.loc[full_scores.nlargest(10).index,'player_id'])
                        top=c[c.player_id.isin(selected)]
                        row=dict(role=role,year=year,min_events=min_events,min_matches=min_matches,evidence_adjusted=evidence,penalty=penalty,players=len(c),selected_top10_observed=len(top),
                            quality_rank_correlation=c.candidate_score.corr(c.future_mean,method='spearman'),
                            reliability_rank_correlation=c.candidate_score.corr(c.future_reliability,method='spearman'))
                        for metric in ['future_mean','future_sd','future_reliability','future_below_expected_share','future_downside']:
                            row['top10_'+metric]=top[metric].mean() if len(top)>0 else np.nan
                        summaries.append(row)
                        details.append(c.assign(role=role,year=year,min_events=min_events,min_matches=min_matches,evidence_adjusted=evidence,penalty=penalty))
        for evidence in [False,True]:
            for penalty in [0.,.075,.15,.30]:
                b=rank(d,role,cutoff,penalty=penalty,pressure_power=3,evidence_adjusted=evidence)
                b['rank']=np.arange(1,len(b)+1)
                boards.append(b.assign(role=role,evidence_adjusted=evidence,penalty=penalty))
    pd.DataFrame(summaries).to_csv(OUT/'future_quality_and_consistency.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'consistency_repeatability.csv',index=False)
    pd.concat(details).to_csv(OUT/'player_validation_details.csv',index=False)
    pd.concat(boards).to_csv(OUT/'candidate_rankings.csv',index=False)
    s=pd.DataFrame(summaries)
    print(s[(s.year>=2025)&s.min_events.eq(30)&~s.evidence_adjusted][['role','year','penalty','players','quality_rank_correlation','reliability_rank_correlation','top10_future_mean','top10_future_sd','top10_future_below_expected_share']].round(4).to_string(index=False))
    print(pd.DataFrame(diagnostics).round(4).to_string(index=False))

if __name__=='__main__':main()
