"""Earlier supplied IPL seasons seed histories; no new prediction labels or fuzzy merges."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from optimize_expectations import DEST,OUT,feature_sets,fit_predict,metrics,WICKETS

def seed_histories():
    aliases=pd.read_csv(OUT/'ipl_verified_name_aliases.csv')[['public_player_id','royals_player_id']].drop_duplicates()
    assert aliases.public_player_id.is_unique and aliases.royals_player_id.is_unique
    identity=dict(zip(aliases.public_player_id,aliases.royals_player_id))
    sums={entity:defaultdict(lambda:[0.,0.,0.]) for entity in ['striker_id','bowler_id']}
    dates=[];events=0
    for path in sorted((OUT.parent/'ipl_json (1)').glob('*.json')):
        data=json.loads(path.read_text());info=data['info'];date=str(info['dates'][0])
        if not '2017-01-01'<=date<'2022-01-01':continue
        if info.get('overs')!=20 or info.get('outcome',{}).get('method'):continue
        registry=info['registry']['people'];dates.append(date)
        for innings in data['innings']:
            if innings.get('super_over'):continue
            for over in innings['overs']:
                for delivery in over['deliveries']:
                    events+=1;r=delivery['runs']['batter']
                    w=int(any(x['player_out']==delivery['batter'] for x in delivery.get('wickets',[])))
                    for entity,key in [('striker_id','batter'),('bowler_id','bowler')]:
                        pid=identity.get(registry[delivery[key]])
                        if pid is None:continue
                        values=sums[entity][str(pid)];values[0]+=r;values[1]+=w;values[2]+=1
    assert dates and max(dates)<'2022-01-01'
    result={'counts':sums,'matches':len(dates),'events':events,'latest_date':max(dates),'earliest_date':min(dates)}
    (DEST/'warm_history_seeds.json').write_text(json.dumps(result,indent=2))
    return result

def apply_warm(b,seed):
    b=b.copy();strength=120
    for entity in ['striker_id','bowler_id']:
        values=np.array([seed['counts'][entity].get(str(int(pid)),[0,0,0]) for pid in b[entity]])
        n=b[entity+'_past_n'].to_numpy()
        for field,index,anchor in [('runs',0,1.5),('wickets',1,.05)]:
            column=entity+'_past_'+field
            b[column]=(b[column].to_numpy()*(n+strength)+values[:,index])/(n+values[:,2]+strength)
        b[entity+'_past_n']=n+values[:,2]
    return b

def main():
    seed=seed_histories();b=apply_warm(pd.read_pickle(DEST/'features.pkl'),seed);rows=[]
    print({k:v for k,v in seed.items() if k!='counts'},flush=True)
    for role in ['bat','bowl']:
        d=b[b.wides.eq(0)].copy() if role=='bat' else b.copy()
        d['y_runs']=d.runs if role=='bat' else d.runs+d.wides+d.noballs
        d['y_wicket']=(d.is_wicket&(d.dismissed_player_id.eq(d.striker_id) if role=='bat' else d.dismissal_type.isin(WICKETS))).astype(int)
        years=pd.to_datetime(d.match_date).dt.year
        for year in [2024,2025]:
            train=d[years.lt(year)];test=d[years.eq(year)]
            r,w=fit_predict(train,test,feature_sets(role)['opponent'],'cat')
            row=dict(role=role,year=year,model='warm_opponent_cat',**metrics(test,r,w));rows.append(row);print(row,flush=True)
    pd.DataFrame(rows).to_csv(DEST/'warm_history_metrics.csv',index=False)
if __name__=='__main__':main()
