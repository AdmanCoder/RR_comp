"""Check frozen chronological expectations by cricket context; no refitting."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/methodology_improvements'

def main():
    rows=[]
    for role in ['bat','bowl']:
        d=pd.read_pickle(ROOT/f'outputs/methodology_audit/replay/{role}_scored_deliveries.pkl')
        d['year']=d.match_date.str[:4].astype(int)
        d['phase_check']=np.select([d.balls_left>84,d.balls_left>30],['powerplay','middle'],default='death')
        d['runs_error']=d.y_runs-d.expected_runs
        d['wicket_error']=d.y_wicket-d.expected_wicket
        for keys,g in d.groupby(['year','innings','phase_check']):
            row=dict(role=role,year=keys[0],innings=keys[1],phase=keys[2],events=len(g),matches=g.match_id.nunique(),actual_runs=g.y_runs.mean(),expected_runs=g.expected_runs.mean(),runs_bias=g.runs_error.mean(),actual_wicket=g.y_wicket.mean(),expected_wicket=g.expected_wicket.mean(),wicket_bias=g.wicket_error.mean())
            # Cluster standard errors for mean residual, not iid-ball errors.
            clusters=g.groupby('match_id').agg(total=('runs_error','sum'),n=('runs_error','size'),wt=('wicket_error','sum'))
            m=len(clusters)
            for name,col,bias in [('runs','total',row['runs_bias']),('wicket','wt',row['wicket_bias'])]:
                row[name+'_cluster_se']=np.sqrt(m/(m-1)*((clusters[col]-bias*clusters.n)**2).sum()/len(g)**2) if m>1 else np.nan
            rows.append(row)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'expectation_calibration.csv',index=False)
    print(df[df.year.eq(2026)][['role','innings','phase','events','runs_bias','runs_cluster_se','wicket_bias','wicket_cluster_se']].round(4).to_string(index=False))
if __name__=='__main__':main()
