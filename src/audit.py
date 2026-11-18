"""Read sources once; emit compact schema, identity and score-reconstruction audit."""
from pathlib import Path
import json
import re
import unicodedata
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'SupeRR Selector Season 2 | Data Pack'
OUT = ROOT / 'outputs'

def normalize_name(value):
    value = unicodedata.normalize('NFKD', str(value))
    return re.sub(r'[^a-z0-9]', '', ''.join(c for c in value if not unicodedata.combining(c)).lower())

def resolve_name(name, identities, threshold=94, margin=8):
    """Fallback only. Never silently merge fuzzy player identities."""
    from rapidfuzz.fuzz import ratio
    key = normalize_name(name)
    scores = identities.assign(score=identities.player_name.map(lambda n: ratio(key, normalize_name(n))))
    scores = scores.sort_values('score', ascending=False).drop_duplicates('player_id')
    if scores.empty:
        return {'status': 'unresolved', 'query': name}
    first = scores.iloc[0]
    gap = float(first.score - scores.iloc[1].score) if len(scores) > 1 else 100.
    exact = first.score == 100 and gap > 0
    return {'query': name, 'candidate_id': int(first.player_id), 'candidate_name': first.player_name,
            'score': float(first.score), 'margin': gap,
            'status': 'exact_normalized' if exact else ('review_fuzzy' if first.score >= threshold and gap >= margin else 'unresolved')}

def run():
    OUT.mkdir(exist_ok=True)
    tables = {p.stem: pd.read_csv(p, keep_default_na=False) for p in DATA.glob('*.csv')}
    b, m, c, ids = [tables[n] for n in ['ball_by_ball','matches','competitions','competition_team_player_mapping']]
    meta = m.merge(c, on='comp_id', validate='many_to_one')
    scope = meta.match_format.eq('T20I') | meta.comp_name.str.match(r'^IPL \d{4}$')
    excluded = meta.loc[~scope, ['match_id','comp_name']]
    b = b.merge(meta.loc[scope], on='match_id', validate='many_to_one')
    b = b[b.innings.isin([1,2])].copy()
    # These are documented hypotheses, NOT verified scoring fields.
    b['assumed_extra'] = b.extras_type.map({'None':0, 'wide':1, 'Overstep':1, 'byes':1, 'legbyes':1, 'other':1, 'Penalty':5})
    b['hypothesis_runs'] = b.runs + b.assumed_extra + b.extras_2_runs
    innings = b.groupby(['match_id','innings'], as_index=False).agg(
        rows=('runs','size'), bat_runs=('runs','sum'), hypothesis_total=('hypothesis_runs','sum'),
        wide_events=('extras_type',lambda s:s.eq('wide').sum()),
        bye_events=('extras_type',lambda s:s.isin(['byes','legbyes']).sum()),
        ambiguous_events=('extras_type',lambda s:s.eq('other').sum()))
    totals=innings.pivot(index='match_id', columns='innings',values='hypothesis_total')
    comparisons=meta.loc[scope].merge(totals.rename(columns={1:'innings1_total',2:'innings2_total'}),on='match_id',how='left')
    comparisons['reported_run_margin']=pd.to_numeric(comparisons.result.str.extract(r'(?i)won by (\d+) runs?')[0],errors='coerce')
    comparisons['hypothesis_margin']=comparisons.innings1_total-comparisons.innings2_total
    margin_rows=comparisons[comparisons.reported_run_margin.notna() & comparisons.revised_target.eq('') & ~comparisons.result.str.contains(r'DLS|D/L|Duckworth|rain',case=False,regex=True) & comparisons.innings2_total.notna()].copy()
    margin_rows['matches_result']=margin_rows.reported_run_margin.eq(margin_rows.hypothesis_margin)
    margin_rows[['match_id','match_date','result','innings1_total','innings2_total','reported_run_margin','hypothesis_margin','matches_result']].to_csv(OUT/'score_reconciliation.csv',index=False)
    identities=ids[['player_id','player_name']].drop_duplicates()
    identities.to_csv(OUT/'player_identity_aliases.csv',index=False)
    all_player_ids=set(b.striker_id)|set(b.non_striker_id)|set(b.bowler_id)
    audit={
      'source_shapes': {n:list(t.shape) for n,t in tables.items()},
      'scope_matches':int(scope.sum()), 'scope_regular_deliveries':len(b),
      'excluded_competitions':excluded.comp_name.value_counts().to_dict(),
      'missing_player_ids': sorted(all_player_ids-set(identities.player_id)),
      'name_alias_ids':int((identities.groupby('player_id').size()>1).sum()),
      'duplicate_delivery_keys':int(b.duplicated(['match_id','innings','overid','over_ball_num']).sum()),
      'extras_code_crosswalk': b.groupby('extras_type').extras1_runs.unique().apply(list).to_dict(),
      'margin_hypothesis_checked':len(margin_rows),
      'margin_hypothesis_exact':int(margin_rows.matches_result.sum()),
      'score_reconstruction_verified':False,
      'blocking_findings':[
         'extras1_runs is an event-category code, not a run amount.',
         'Wide/byes/legbyes rows overwhelmingly have runs=0 and extras_2_runs=0; actual extra amounts are not explicitly available.',
         'The one-extra-per-event hypothesis is diagnostic only; result reconciliation cannot prove ball-level correctness.',
         'Exact required rates, team run rates and bowler conceded-run targets cannot yet be asserted.'],
      'name_matching_policy':'IDs first; normalized exact only if unique; fuzzy candidates require review, never automatic merge.'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,default=lambda x:int(x)))
    print(json.dumps({k:v for k,v in audit.items() if k not in ['source_shapes','extras_code_crosswalk']},indent=2))
    return tables, b

if __name__=='__main__': run()
