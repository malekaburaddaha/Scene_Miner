# Milestone 7 — exact 390-query suite, domain calibration, Table 3

## 1. The suite is now exact

`benchmark_cvpr/queries_390.json` — **240 train / 150 eval = 390**.

| eval stratum | count | target | mean relevant | prevalence |
|---|---|---|---|---|
| sensor-degradation | **62** | 62 | 54.2 | 0.0900 |
| macro-environmental | **58** | 58 | 154.0 | 0.2558 |
| compositional (held out) | **18** | 18 | 24.3 | 0.0404 |
| negated | **12** | 12 | 60.6 | 0.1006 |

Train/eval text strings are 100 % disjoint. **A leakage bug was caught and
fixed**: the first generator let 8 compositional *predicates* (e.g.
`droplets ∧ night`) appear in the training pool, which would have falsified the
"never supervised" claim. Compositional predicates are now reserved and
filtered out of training — verified 0 overlap. All 150 eval queries are usable
(0 < relevant < 602).

## 2. Per-domain z-scoring does **not** fix the skew — it overcorrects

nuScenes share of the diversified top-5 against the relevant base rate:

| ranker | raw | calibrated | verdict |
|---|---|---|---|
| SceneMiner v2.1 | **+8.7 pts** | **−23.8 pts** | overcorrects badly |
| logit VLM lookup | −6.6 | −17.4 | worse |
| flat VLM lookup | −3.4 | −12.6 | worse |
| metadata prior | +1.0 | +15.4 | worse |
| zero-shot SigLIP2 | −16.2 | −10.6 | only one improved |

Two things follow. First, **the residual skew on this suite is +8.7 points, not
the 12.1 the paper reports** — the larger query suite dilutes it. Second,
z-scoring within domain flips nuScenes from over- to *under*-represented, and
costs accuracy as well (SceneMiner 0.5312 → 0.4925 mAP). **I do not recommend
shipping this fix.** A milder correction (a single learned per-domain offset,
or domain-stratified MMR quotas) is the thing to try next.

## 3. Table 3 on the 150-query suite — with one artefact flagged

mAP by stratum, 602-scene eval split, SceneMiner v2.1 averaged over 5 seeds:

| ranker | sensor-degrad. | macro-env. | compositional | negated | ALL |
|---|---|---|---|---|---|
| metadata prior | 0.2791 | **1.0000** ⚠ | 0.6998 | 0.3449 | 0.6136 |
| **SceneMiner v2.1** | **0.6608** | 0.4079 | 0.5496 | 0.4306 | 0.5312 |
| logit VLM lookup | 0.5860 | — | 0.3610 | 0.6360 | 0.5589 |
| flat VLM lookup | 0.5790 | — | 0.3229 | 0.6383 | 0.5484 |
| zero-shot SigLIP2 | 0.4440 | 0.4867 | 0.3701 | 0.3554 | 0.4446 |

⚠ **`metadata_prior` scoring exactly 1.0000 on the macro stratum is the
tautology again, not a result.** Macro predicates are defined purely over
`lighting`/`weather`/`density`, and the metadata prior reads those same fields,
so it is a perfect oracle there by construction. Its 0.6136 "ALL" figure is
therefore meaningless and must not appear in the paper as a baseline. The same
caution applies, more weakly, to its 0.6998 on the compositional stratum, whose
predicates also contain metadata terms.

The lookup baselines are undefined (—) on the macro stratum because those
queries name no degradation attribute.

**The honest comparison is the sensor-degradation stratum — the one the paper is
about — where the tautology does not apply:**

| ranker | sensor-degradation mAP |
|---|---|
| **SceneMiner v2.1** | **0.6608** |
| logit VLM lookup (calibrated) | 0.6074 |
| logit VLM lookup (raw) | 0.5860 |
| flat VLM lookup | 0.5790 |
| zero-shot SigLIP2 | 0.4440 |
| metadata prior | 0.2791 |

**SceneMiner v2.1 wins the stratum that matters, +0.075 over the best lookup.**
It also leads on the held-out compositional stratum among non-tautological
rankers (0.5496 vs 0.3610 for the logit lookup), which is the open-vocabulary
transfer evidence Reviewer ETJq asked for.

It loses the negated stratum (0.4306 vs 0.6383) — negation remains the weakest
capability, consistent with the frozen text tower noted in the submission.

## Not completed
Fine-tuned SigLIP2, caption→BM25, Qwen2.5-VL pairwise and the L2 linear probe
are not yet in this table; they need matched-supervision training runs that did
not fit alongside the suite rebuild. Table 2 (de-confounded ablation) was run on
the earlier 22-query protocol in the overnight session and has not been re-run
on the 150-query suite.
