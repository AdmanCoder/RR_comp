"""Write transparent metrics, feature decisions and limitations for this experiment."""
import json
import pandas as pd
from optimize_expectations import DEST

def main():
    val=pd.read_csv(DEST/'validation_metrics.csv');ab=pd.read_csv(DEST/'feature_ablation.csv')
    test=pd.read_csv(DEST/'reused_2026_metrics.csv');ci=pd.read_csv(DEST/'gain_intervals.csv')
    choices=json.loads((DEST/'selected_heads.json').read_text())
    warm=pd.read_csv(DEST/'warm_history_metrics.csv')
    differences=[]
    for row in ab.itertuples():
        base=val[(val.role==row.role)&(val.year==row.year)&(val.model=='opponent_cat')].iloc[0]
        differences.append(dict(role=row.role,year=row.year,variant=row.variant,
            mse_change_pct=100*(row.mse/base.mse-1),
            brier_change_pct=100*(row.brier/base.brier-1) if pd.notna(row.brier) else float('nan')))
    delta=pd.DataFrame(differences)
    delta.to_csv(DEST/'ablation_deltas.csv',index=False)
    avg=delta.groupby(['role','variant'])[['mse_change_pct','brier_change_pct']].mean()
    report=['# Expected-performance optimization: results','',
        '**The 20% error-reduction target was not met.** This experiment does not establish that 20% is impossible; it establishes that these tested additions and model families did not achieve it. No final player rankings were regenerated.','',
        '## What was tested','',
        'Two expanding-year development folds: 2024 and 2025. EBM, CatBoost, XGBoost, equal-weight ensembles, four feature-removal experiments, and a run-distribution classifier. Both batting and bowling were evaluated. Forecasting-only own-player histories were also tested, but excluded from selection for the common player-performance benchmark. See PLAN.md for the protocol and research.','',
        'Primary selection losses: MSE for expected runs and Brier for wicket probabilities, selecting heads separately. MAE from the previous report is retained and must not be silently replaced by a more flattering measure. Relative gain = 100 × (baseline loss − model loss) / baseline loss. No 20% target is claimed from improving a small pre-existing improvement by 20%.','',
        '## Older-history experiment','',warm.round(6).to_string(index=False),'','## Selected models','',json.dumps(choices,indent=2),'',
        '## Reused 2026 evaluation','',test.round(6).to_string(index=False),'',
        '2026 was previously examined in this project. Selection in this experiment used only 2024/25, but this is not a new untouched holdout. Historical feature summaries may update from earlier matches during the evaluation year, as would be possible live; model fitting remains restricted to earlier years.','',
        '### Relative gains versus the same historical-mean baseline','',ci.round(3).to_string(index=False),'',
        'Intervals use 1,000 paired whole-match bootstrap resamples; they describe match-sampling uncertainty conditional on selected models, not model-selection uncertainty. Negative gains mean worse error.','',
        '## Feature audit','',
        '| Group | Decision / meaning |','|---|---|',
        '| Original six | Innings, balls left, wickets lost, score, runs needed, knockout. Retain for comparison. |',
        '| Explicit state | Test required/current run rate, phase, legal-ball position within over and historical-par deficit. These expose useful nonlinear structure. |',
        '| Settledness/workload | Prior batter balls faced and bowler legal balls bowled. Available before the current ball. |',
        '| Recent innings history | Runs and wickets in the previous 6/12 delivery events. Shifted to exclude current outcomes; tested via removal. |',
        '| Free hit | Derived from prior no-balls and wides; no knowledge of the upcoming ball’s extras. |',
        '| Player styles | Batting hand, bowling style, pace/spin. Pre-existing traits; tested via removal. |',
        '| Venue/opponent histories | Earlier-date shrunk batter-run and striker-dismissal rates associated with venue/opponent, plus history size. These are raw historical proxies, not fully context-adjusted ability ratings or official bowling economy/averages. |',
        '| Own-player history/current scoring rate | Forecasting diagnostic only; may change the rating to above-personal-expectation rather than absolute contextual quality. Excluded from common-benchmark model selection. |',
        '| Knockout | Still in pressure formula. Removal from the outcome predictor is tested independently; a feature can matter for perceived stakes without predicting runs/wickets. |',
        '| Do not add | Current shot, ball speed/trajectory, extras amount, runs, wicket flag, final result, unshifted career averages, retrospective batting order. These are post-delivery, outcome-derived or insufficiently verified for this pre-ball task. |','',
        '### Feature-removal results','',
        'Mean percentage loss change relative to the full opponent-context CatBoost model, over 2024/25. Positive means removal makes the model worse; negative means removal improves it. Small inconsistent changes do not establish a universal keep/drop rule.',
        '',avg.round(4).to_string(),'',
        'These ablations are candidates in model selection. A run-only distributional model is assessed only on run losses. No manual rule removed a pressure component. Exact selected feature lists follow from feature_manifest.json and the selected_heads.json variant; drop_* variants use opponent features minus the named group.','',
        '## Interpretation and next work','',
        'Expected-run improvements are modest; wicket prediction shows little exploitable gain from these features. This is evidence against promising a 20% reduction from another algorithm swap, not proof of an irreducible ceiling. Individual deliveries remain highly variable.','',
        'An additional experiment uses supplied 2017–2021 IPL JSON to seed player histories through existing verified UUID-to-player-ID links; warm_history_metrics.csv reports its development results. Only earlier-date histories are added: these older matches are not new training labels. Unknown identities are skipped, not fuzzily merged. Full older-season training remains unimplemented and would need further reconstruction checks. More history cannot be promised to deliver 20%. Richer genuinely pre-ball pitch/weather/field-setting information is another possibility if available and timestamped. Changing the target to innings totals or using post-ball tracking could produce larger percentages but would be a different task.','',
        'Before replacing ranking benchmarks, check expected-run calibration and match-level ranking stability. Wicket heads need not share the run head’s model or features. Do not promote a larger model solely because one year improves.','',
        '## Leakage and scope','',
        'All 23 tests pass, including four new feature tests for current/future outcome invariance, same-date exclusion, free-hit carry and allowlists. Training/evaluation splits are date ordered and whole-match separated. Historical priors are fixed constants; categorical mappings use training data only. Tests are evidence, not a universal proof. Only complete, unrevised, verified IPL fixtures are included; unresolved T20I extras remain outside this pilot.','',
        '## Sources','',
        '- Contextual batting/bowling research: https://www.sfu.ca/~tswartz/papers/contextual.pdf',
        '- Opta next-ball methodology: https://theanalyst.com/articles/opta-next-ball-predictor',
        '- Author-posted cricstate research (peer-review status not verified): https://dudailia.github.io/cricstate/paper/',
        '- Expected-mean versus median scoring: https://scikit-learn.org/stable/modules/model_evaluation.html']
    (DEST/'RESULTS.md').write_text('\n'.join(report)+'\n')
    print(ci.round(3).to_string(index=False));print(choices);print(avg.round(3).to_string())
if __name__=='__main__':main()
