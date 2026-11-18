"""Audit wicket calibration and dismissals outside selected pressure passages."""
import pandas as pd
from audit import OUT
from ipl_pressure import KEYS

def main():
    destination=OUT/'wicket_boundary_audit'
    destination.mkdir(exist_ok=True)
    trial=pd.read_csv(OUT/'combined_replay/combined_pressure_trial.csv.gz',usecols=KEYS+['bat_candidate','bowl_candidate'])
    ref=pd.read_pickle(OUT/'combined_replay/features.pkl')
    ref=ref[ref.match_date.str.startswith('2022')].merge(trial,on=KEYS,validate='one_to_one')
    for role,model,player in [('bat','ebm','striker_id'),('bowl','warm_cat','bowler_id')]:
        data=pd.read_pickle(OUT/f'combined_replay/{role}_scored_deliveries.pkl').merge(trial,on=KEYS,validate='one_to_one')
        predictions=pd.concat(pd.read_pickle(OUT/f'v1_candidate_comparison/{role}_{year}_predictions.pkl') for year in range(2023,2027))
        data=data.merge(predictions[KEYS+[model+'_wicket']],on=KEYS,validate='one_to_one')
        cutoff=ref.loc[ref.wides.eq(0)&((role=='bat')|ref.noballs.eq(0)),role+'_candidate'].quantile(.75)
        data['high']=data[role+'_candidate'].ge(cutoff)
        data['year']=data.match_date.str[:4]
        for fields,label in [(['high'],'overall'),(['high','phase'],'phase'),(['high','year'],'year')]:
            table=data.groupby(fields).agg(events=('y_wicket','size'),actual=('y_wicket','sum'),expected=(model+'_wicket','sum'))
            table['actual_rate']=table.actual/table.events
            table['expected_rate']=table.expected/table.events
            table.to_csv(destination/f'{role}_{label}.csv')
        if role!='bat':continue
        rows=[]
        apps=pd.read_csv(OUT/'v1_candidate_comparison/bat_ebm_appearances.csv')
        qualifying=apps[apps.balls.ge(6)][['striker_id','match_id']]
        data=data.merge(qualifying,on=['striker_id','match_id'],validate='many_to_one')
        for (pid,match),group in data.groupby(['striker_id','match_id']):
            group=group.sort_values(['innings','innings_ball_num'],kind='stable').reset_index(drop=True)
            selected=group.index[group.high]
            last=int(selected.max())
            dismissals=group.index[group.y_wicket.eq(1)&~group.high]
            after=[int(i)-last for i in dismissals if i>last]
            rows.append(dict(player_id=pid,player=group.striker_name.iloc[0],match_id=match,
                             selected_dismissals=int(group.loc[group.high,'y_wicket'].sum()),
                             outside_dismissals=int(group.loc[~group.high,'y_wicket'].sum()),
                             dismissal_within_next_6_faced=int(bool(after) and min(after)<=6)))
        detail=pd.DataFrame(rows)
        detail.to_csv(destination/'bat_boundaries.csv',index=False)
        detail.groupby(['player_id','player']).agg(matches=('match_id','size'),
             selected_dismissals=('selected_dismissals','sum'),outside_dismissals=('outside_dismissals','sum'),
             nearby_dismissals=('dismissal_within_next_6_faced','sum')).to_csv(destination/'bat_player_boundaries.csv')

if __name__=='__main__':main()
