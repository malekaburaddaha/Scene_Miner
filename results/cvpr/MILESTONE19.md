# Milestone 19 — compositional synergy and single-rater sensitivity

## Verdict up front

**The compositional-synergy claim does not hold on human ground truth.** On
semantic object × adverse condition conjunctions, SceneMiner v2.4 ties
zero-shot SigLIP2 and ties a trivial late fusion of SigLIP2 with the VLM lookup.
The best ranker is a different trivial control, **z(SigLIP2) + night/rain
metadata**, which beats v2.4 on all 11 conjunctions (p = 0.0007). SceneMiner's
advantage is real but lives in **atomic degradation**, and it is robust to a
simulated second rater at the macro level (100 % of trials against SigLIP2,
92–100 % against the lookup). Per attribute it is fragile: glare and wet road
are coin flips against the lookup.

## 1. The three regimes, with the controls a reviewer would ask for

Every number is graded by humans. "Late fusion" adds z-scored rankers with no
learning at all.

| regime | SceneMiner v2.4 | SigLIP2 | logit lookup | late fusion SigLIP2+lookup | late fusion SigLIP2+metadata |
|---|---|---|---|---|---|
| atomic degradation (550 rater-labelled) | **0.7893** | 0.5688 | 0.7325 | 0.7372 | 0.6257 |
| atomic semantics (258 corpus) | 0.3371 | 0.3370 | 0.2171 (chance) | = SigLIP2 | = SigLIP2 |
| atomic semantics (clean val150) | 0.3827 | 0.3824 | 0.2200 (chance) | = SigLIP2 | = SigLIP2 |
| semantic × condition (258 corpus, 30 phrasings) | 0.2916 | 0.2927 | 0.2272 | 0.2843 | **0.3271** |

Here "v2.4" is the 5-seed mean score ranked once (0.7893). Milestone 18's
0.7879 is the mean of per-seed APs.

* **Degradation:** SceneMiner wins clearly, by +0.052 over the strongest control
  (late fusion with the lookup). This is the regime it was trained for.
* **Semantics:** the fixed-column lookup and metadata fall to chance, and every
  sensible ranker reduces to SigLIP2. SceneMiner matches it and adds nothing.
* **Conjunctions:** nothing learned beats SigLIP2 plus metadata. The brief's
  expected "+0.05 to +0.17 over every baseline" is not supported.

### Conjunction details (`results/cvpr/milestone19_compositional_synergy.json`)

Conjunctions were **generated**, not hand-picked: 4 semantic flags × 6
conditions, keeping every cell with ≥ 5 positives among the 258 human-labelled
scenes. That gave 11 conjunctions, each with one templated query
("{object} {condition}"), plus the 19 Stratum-5 hand phrasings of the same
cells. Ground truth: object from nuScenes 3D boxes; night/rain from the human
scene description; glare/wet/droplets/low contrast from the rater.

Paired permutation tests over the 11 predicates (phrasings averaged first):

| comparison | Δ mAP | p | predicates won |
|---|---|---|---|
| v2.4 − SigLIP2 | −0.0027 | 0.83 | 4/11 |
| v2.4 − logit lookup | +0.0615 | 0.098 | 8/11 |
| v2.4 − late fusion SigLIP2+lookup | +0.0063 | 0.86 | 4/11 |
| v2.4 − metadata | +0.0285 | 0.54 | 5/11 |
| **v2.4 − late fusion SigLIP2+metadata** | **−0.0355** | **0.0007** | **0/11** |
| v2.3A − SigLIP2 | +0.0020 | 0.89 | 6/11 |
| v2.3A − late fusion SigLIP2+metadata | −0.0307 | 0.017 | 1/11 |

Why: v2.4's contrastive gate treats night and rain as foils, so a query such as
"pedestrians near the road at night" gets γ ≈ 0 and abstains entirely to
SigLIP2. It is SigLIP2 on these queries by design. v2.3A's looser gate let part
of the margin through, but that was not enough to beat the metadata fusion
either. The cheapest effective system for conjunctions is to add recorded
metadata to an open-vocabulary retriever. I did not retune v2.4 against these
11 predicates; that would turn them into a validation set.

### Two parts of the brief not done as written, and why

* **val150 conjunctions with "high-confidence VLM" degradation labels.**
  Grading glare/wet/droplets/low-contrast on val150 would require using the
  VLM's own margins as ground truth. Those are the inputs of the lookup and of
  SceneMiner's margin channel, so the evaluation would be circular. Only
  human-graded conditions (night, rain) are usable on val150. There, only one
  conjunction (heavy vehicle × rain) reaches 5 positives, so val150 cannot
  support a compositional claim.
* **The 50 "unsure" scenes.** The rater set no labels on any of them, so there
  is nothing to score. Mode A (530 scenes, skipped scenes dropped) vs Mode B
  (550) is the available check of the rater's own uncertainty.

## 2. Single-rater sensitivity (`results/cvpr/milestone19_rater_sensitivity.json`)

### Subsets

| subset | v2.4 | lookup | SigLIP2 | LF SigLIP2+lookup | LF SigLIP2+meta |
|---|---|---|---|---|---|
| Mode B, 550 | 0.7893 | 0.7325 | 0.5688 | 0.7372 | 0.6257 |
| Mode A, 530 | 0.7899 | 0.7332 | 0.5713 | 0.7380 | 0.6287 |
| drop 10 % most CLIP-ambiguous | 0.7771 | 0.7076 | 0.5788 | 0.7225 | 0.6149 |
| drop 20 % most CLIP-ambiguous | 0.7638 | 0.6709 | 0.5652 | 0.7110 | 0.6105 |

"Boundary" is defined by **CLIP ViT-B/32**, a model none of the rankers uses,
so the subset is not selected by any compared system (attribute prompt vs a
neutral prompt; ambiguity = 1 − 2|p − 0.5|). The brief proposed an ensemble of
CLIP + SigLIP2 + Qwen. SigLIP2 and Qwen *are* rankers here, and selecting the
evaluation subset with them would bias it toward whichever ranker the ensemble
resembles. Removing ambiguous scenes **widens** v2.4's lead over the lookup
(+0.057 → +0.093).

### Monte Carlo second rater (500 trials per setting)

f positives and f negatives are flipped per attribute so that Cohen's κ against
the real labels hits the target, with prevalence fixed. Flips are drawn
uniformly, or weighted toward CLIP-ambiguous scenes. The table shows the share of
trials in which v2.4's **macro** mAP exceeds the comparator's:

| κ (realised) | flips | > lookup | > SigLIP2 | > LF SigLIP2+lookup | > LF SigLIP2+meta |
|---|---|---|---|---|---|
| 0.898 | uniform | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.898 | boundary | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.847 | uniform | 1.000 | 1.000 | 0.998 | 1.000 |
| 0.847 | boundary | 0.998 | 1.000 | 1.000 | 1.000 |
| 0.751 | uniform | 0.978 | 1.000 | 0.982 | 1.000 |
| 0.751 | boundary | **0.920** | 1.000 | 0.966 | 1.000 |

"Beats the lookup in 100 % of perturbations" holds at κ ≥ 0.85. At κ = 0.75 with
boundary-weighted flips it holds in 92 %. Against SigLIP2 it holds in 100 %
everywhere.

**Per attribute the picture is weaker** (share of trials won vs the lookup, at
κ = 0.85 uniform / κ = 0.75 boundary): droplets 1.00 / 0.86, **glare 0.54 /
0.31**, **wet road 0.61 / 0.49**, low contrast 1.00 / 1.00. Against late fusion
SigLIP2+lookup, glare is lost in essentially every trial (0.00–0.05). The macro
lead rests on low contrast and droplets. The paper should report the
per-attribute result, not only the macro one.

## Artefacts

| path | contents |
|---|---|
| `scripts/eval_compositional_synergy.py` | generated conjunctions, late-fusion controls, predicate-level tests |
| `scripts/eval_rater_sensitivity.py` | Mode A/B, CLIP-boundary subsets, Monte Carlo second rater |
| `results/cvpr/milestone19_compositional_synergy.json` | per-query, per-predicate, regimes table |
| `results/cvpr/milestone19_rater_sensitivity.json` | all subsets and 3000 Monte Carlo trials |
