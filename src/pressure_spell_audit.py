"""Ablate tracking summaries built strictly from previous bowling events."""
import json
import numpy as np
import pandas as pd
from optimize_expectations import DEST, WICKETS, feature_sets, metrics
from pressure_ml_matchups import extra_context, EXTRA
from pressure_ml_optimization import cat_predictions
from warm_history_experiment import apply_warm

TRACKING = ['bowl_speed','delivery_style','over_or_around','on_or_off']

def spell_features(frame):
    keys = [frame.match_id,frame.innings,frame.bowler_id]
    speed = frame.bowl_speed.where(frame.bowl_speed.gt(0))
    tracked = pd.DataFrame(dict(speed=speed, speed_known=speed.notna().astype(float),
        around=frame.over_or_around.map({'Around':1.,'Over':0.}),
        off=frame.on_or_off.map({'Off-side':1.,'On-side':0.}),
        variation=frame.delivery_style.isin(['Off Cutter','Leg Cut','Googly','Slow Ball','Slow Bouncer','Knuckle','Back of Hand']).where(frame.delivery_style.notna()).astype(float)))
    return tracked.groupby(keys,sort=False).transform(
        lambda s:s.shift().rolling(6,min_periods=1).mean()).add_prefix('prior_spell_')


def main():
    raw = pd.read_pickle(DEST/'features.pkl')
    audit = raw[TRACKING].copy()
    audit['bowl_speed'] = audit.bowl_speed.where(audit.bowl_speed.gt(0))
    audit = audit.notna().assign(year=raw.match_date.str[:4], style=raw.pace_or_spin)
    audit.groupby(['year','style']).agg(events=('bowl_speed','size'),
        **{key:(key,'mean') for key in TRACKING}).to_csv(DEST/'tracking_coverage.csv')
    spell = spell_features(raw)
    for i in [10,250,1100]:
        sample = raw.iloc[:1500].copy()
        sample.loc[i:,TRACKING] = np.nan
        pd.testing.assert_frame_equal(spell.loc[:i],spell_features(sample).loc[:i])
    raw = apply_warm(raw,json.loads((DEST/'warm_history_seeds.json').read_text())).join(spell)
    rows, calibration = [], []
    for role in ['bat','bowl']:
        data = extra_context(raw,role)
        bat = role=='bat'
        if bat:
            data = data[data.wides.eq(0)].copy()
        data['y_runs'] = data.runs if bat else data.charged_runs
        data['y_wicket'] = (data.striker_out if bat else
            (data.is_wicket&data.dismissal_type.isin(WICKETS))).astype(int)
        year = data.match_date.str[:4].astype(int)
        pressure = role+'_pressure'
        countable = data.wides.eq(0)&(bat|data.noballs.eq(0))
        cutoff = data.loc[year.eq(2022)&countable,pressure].quantile(.75)
        for season in [2024,2025,2026]:
            train,test = data[year.lt(season)],data[year.eq(season)]
            assert train.match_date.max()<test.match_date.min()
            assert not set(train.match_id)&set(test.match_id)
            base = feature_sets(role)['opponent']+EXTRA
            actual = test[['y_runs','y_wicket']].to_numpy()
            baseline = (actual-train[['y_runs','y_wicket']].mean().to_numpy())**2
            for model, columns in [('core',base),('prior_spell',base+list(spell.columns))]:
                pred = cat_predictions(train,test,columns,None)
                for group,mask in [('all',np.ones(len(test),dtype=bool)),('high',test[pressure].ge(cutoff))]:
                    result = metrics(test.loc[mask],pred[mask,0],pred[mask,1])
                    reference = baseline[mask].mean(axis=0)
                    rows.append(dict(role=role,year=season,model=model,subset=group,
                        mse_gain_pct=100*(1-result['mse']/reference[0]),
                        brier_gain_pct=100*(1-result['brier']/reference[1]),**result))
                for group,mask in [('death',test.balls_left.le(30)),('new_batter',test.batter_faced.lt(5)),
                                   ('free_hit',test.free_hit.eq(1))]:
                    outcomes = actual[mask,1]
                    values = pred[mask,1]
                    calibration.append(dict(role=role,year=season,model=model,subset=group,
                        events=len(outcomes),wickets=int(outcomes.sum()),observed=outcomes.mean(),
                        predicted=values.mean(),bias_pp=100*(values-outcomes).mean()))
            print(role,season,flush=True)
            pd.DataFrame(rows).to_csv(DEST/'prior_spell_metrics.csv',index=False)
            pd.DataFrame(calibration).to_csv(DEST/'wicket_situation_calibration.csv',index=False)

if __name__=='__main__':
    main()
