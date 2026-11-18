"""Audit free hits and replay chronological wicket calibration."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from optimize_expectations import DEST, WICKETS
from ipl_pressure import KEYS


def main():
    data = pd.read_pickle(DEST/'features.pkl').sort_values(KEYS)
    # Recover eligibility independently from the last non-wide event.
    grouped = data.groupby(['match_id','innings'],sort=False)
    previous = grouped.noballs.transform(
        lambda s: s.where(data.loc[s.index,'wides'].eq(0)).ffill().shift())
    np.testing.assert_array_equal(previous.fillna(0).gt(0),data.free_hit.astype(bool))
    data['year'] = data.match_date.str[:4].astype(int)
    data['credited'] = data.is_wicket & data.dismissal_type.isin(WICKETS)
    free = data[data.free_hit.eq(1)]
    assert not free.credited.any()
    free.groupby(['year','dismissal_type'],dropna=False).size().rename('events').to_csv(
        DEST/'free_hit_label_audit.csv')
    rows = []
    for role in ['bat','bowl']:
        bat = role == 'bat'
        sample = (data[data.wides.eq(0)] if bat else data).copy()
        sample['actual'] = (sample.is_wicket & sample.dismissed_player_id.eq(sample.striker_id)
                            if bat else sample.credited).astype(int)
        predictions = pd.read_csv(DEST.parent/'player_evaluation_v1'/f'{role}_delivery_evaluation.csv.gz',
                                  usecols=KEYS+['expected_wicket','train_end'])
        sample = sample.merge(predictions,on=KEYS,validate='one_to_one')
        assert (sample.train_end<sample.match_date).all()
        sample['group'] = np.select([sample.free_hit.eq(1),sample.balls_left.le(30),
                                    sample.batter_faced.lt(5)],['free_hit','death','new_batter'],
                                    default='other')
        sample['residual'] = sample.actual-sample.expected_wicket
        for year in [2024,2025,2026]:
            train = sample[sample.year.lt(year)]
            test = sample[sample.year.eq(year)]
            assert not set(train.match_id)&set(test.match_id)
            prior = train.groupby('group').residual.agg(['sum','size'])
            offset = prior['sum']/(prior['size']+500)
            unchanged = test.expected_wicket
            rules = unchanged.where(~test.free_hit.eq(1),0) if not bat else unchanged
            corrected = (rules+test.group.map(offset).fillna(0)).clip(1e-6,1-1e-6)
            if not bat:
                corrected = corrected.where(~test.free_hit.eq(1),0)
            for model, prediction in [('ebm',unchanged),('free_hit_rule',rules),('prior_calibrated',corrected)]:
                for group in ['all','free_hit','death','new_batter']:
                    mask = test.group.eq(group) if group!='all' else np.ones(len(test),dtype=bool)
                    y,p = test.actual[mask], prediction[mask]
                    rows.append(dict(role=role,year=year,model=model,group=group,events=len(y),
                        wickets=y.sum(),observed=y.mean(),predicted=p.mean(),
                        brier=brier_score_loss(y,p),logloss=log_loss(y,p,labels=[0,1])))
    pd.DataFrame(rows).to_csv(DEST/'wicket_correction_replay.csv',index=False)

if __name__=='__main__':
    main()
