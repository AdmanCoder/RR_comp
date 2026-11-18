"""Test pretraining on the archive and continued fitting on modern IPL seasons."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from older_ipl_training import load_older
from optimize_expectations import DEST


def main():
    archive = load_older('2008-01-01')
    recent = pd.read_pickle(DEST/'features.pkl')
    assert archive.match_date.max() < recent.match_date.min()
    print(len(archive),archive.match_id.nunique(),flush=True)
    fields = ['innings','score','balls_left','wickets','needed','batter_faced','bowler_balls',
              'required_rpo','current_rpo']
    archive['required_rpo'] = 6*archive.needed/archive.balls_left
    archive['current_rpo'] = 6*archive.score/(121-archive.balls_left)
    rows = []
    params = dict(depth=5,learning_rate=.035,l2_leaf_reg=20,verbose=False,
                  thread_count=2,random_seed=17,allow_writing_files=False)
    for role in ['bat','bowl']:
        bat = role=='bat'
        historical = archive[archive.wides.eq(0)] if bat else archive
        data = recent[recent.wides.eq(0)] if bat else recent
        def runs(frame):
            return frame.runs if bat else frame.runs+frame.wides+frame.noballs
        pretrained = CatBoostRegressor(iterations=200,**params).fit(
            historical[fields],runs(historical))
        years = pd.to_datetime(data.match_date).dt.year
        for year in [2024,2025,2026]:
            train,test = data[years.lt(year)],data[years.eq(year)]
            assert train.match_date.max()<test.match_date.min()
            assert not set(train.match_id)&set(test.match_id)
            actual = runs(test)
            baseline = np.mean((actual-runs(train).mean())**2)
            for name, budget, start in [('modern',350,None),('archive_pretrained',150,pretrained)]:
                model = CatBoostRegressor(iterations=budget,**params).fit(
                    train[fields],runs(train),init_model=start)
                pred = np.maximum(0,model.predict(test[fields]))
                mse = np.mean((actual-pred)**2)
                rows.append(dict(role=role,year=year,model=name,mse=mse,
                                 constant_mse=baseline,mse_gain_pct=100*(1-mse/baseline)))
            pd.DataFrame(rows).to_csv(DEST/'archive_pretraining.csv',index=False)
            print(rows[-1],flush=True)


if __name__=='__main__':
    main()
