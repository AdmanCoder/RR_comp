"""Development experiment: earlier-season residual correction, runs only.
The correction is fitted exclusively from earlier out-of-season residuals.
"""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/methodology_improvements'

def main():
    rows=[]
    for role in ['bat','bowl']:
        d=pd.read_pickle(ROOT/f'outputs/methodology_audit/replay/{role}_scored_deliveries.pkl')
        d=d.reset_index(drop=True)
        d['year']=d.match_date.str[:4].astype(int)
        d['phase_calibration']=np.select([d.balls_left>84,d.balls_left>30],['powerplay','middle'],default='death')
        d['residual']=d.y_runs-d.expected_runs
        d['calibrated_runs']=d.expected_runs
        d['calibration_train_end']=''
        for year in [2024,2025,2026]:
            train=d[d.year<year]
            test=d[d.year==year]
            assert pd.to_datetime(train.match_date).max()<pd.to_datetime(test.match_date).min()
            groups=train.groupby(['innings','phase_calibration']).residual.agg(['sum','count'])
            correction=groups['sum']/(groups['count']+200)
            keys=pd.MultiIndex.from_frame(test[['innings','phase_calibration']])
            values=correction.reindex(keys).fillna(0).to_numpy()
            d.loc[test.index,'calibrated_runs']=np.maximum(0,test.expected_runs+values)
            d.loc[test.index,'calibration_train_end']=train.match_date.max()
            for phase in ['all','powerplay','middle','death']:
                mask=d.year.eq(year)&(True if phase=='all' else d.phase_calibration.eq(phase))
                g=d[mask]
                before=((g.y_runs-g.expected_runs)**2).mean()
                after=((g.y_runs-g.calibrated_runs)**2).mean()
                rows.append(dict(role=role,year=year,phase=phase,events=len(g),before_mse=before,after_mse=after,relative_improvement_pct=100*(before-after)/before))
        d[['match_id','innings','overid','over_ball_num','expected_runs','calibrated_runs','calibration_train_end']].to_csv(OUT/f'{role}_historical_calibration_predictions.csv',index=False)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'historical_calibration_metrics.csv',index=False)
    print(df[df.phase.eq('all')].round(5).to_string(index=False))
if __name__=='__main__':main()
