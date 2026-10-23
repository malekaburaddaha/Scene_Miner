# Milestone 15 — Stratum 5: safety-critical semantic edge cases, 100 % human ground truth

## Verdict up front

The expected headline does **not** hold. Graded against human 3D annotation,
**SceneMiner v2.3 is worse than zero-shot SigLIP2 on semantic edge cases**
(atomic: 0.2295 vs 0.3370, p = 0.0001). The as-trained head learned only four
degradation concepts, and its routing and residual actively damage queries
about pedestrians, cones and trucks.

Two findings do hold and are worth stating:

1. **The 4-column VLM lookup is blind to semantics, exactly as predicted.** On
   all 20 atomic queries the flat lookup, logit lookup and metadata baseline
   score exactly chance (0.2171), because no column applies.
2. **A post-hoc variant, v2.3A, closes the gap to SigLIP2 and beats the lookup
   significantly.** It abstains toward SigLIP2 out of vocabulary: atomic
   +0.1106 over the logit lookup (p = 0.0001), compositional +0.0670 (p = 0.03,
   12 predicates), and within 0.009 of SigLIP2. It does not beat SigLIP2 on
   semantics, and it costs in-domain accuracy elsewhere (below).

## Ground truth (`benchmark_cvpr/human_semantic_gt.json`)

`scripts/extract_human_semantic_gt.py` reads only the official nuScenes human
tables (`sample_annotation`, `instance`, `category`, `attribute`,
`visibility`) and the human-written `scene.description`. No model output is
used. An object counts if it is inside the CAM_FRONT field of view (|azimuth| ≤
32.6° in the ego frame) and visibility > 40 %. A scene flag is set when the
condition holds in ≥ 25 % of the scene's keyframes; per-scene fractions are
stored.

| flag | definition | corpus (700) | val150 |
|---|---|---|---|
| pedestrian_close | adult/child/worker/police pedestrian ≤ 25 m | 230 | 35 |
| vru | bicycle/motorcycle **with rider** ≤ 40 m, or pedestrian ≤ 15 m | 119 | 26 |
| construction_zone | ≥ 3 cones, or cone + barrier, or worker, or construction vehicle, ≤ 40 m | 142 | 26 |
| heavy_vehicle | bus, truck, trailer, construction vehicle ≤ 25 m | 149 | 36 |
| rare_or_child | child, stroller, wheelchair, police, emergency vehicle, debris, pushable ≤ 40 m | 84 | 22 |
| night / rain | word in the human scene description | 84 / 138 | 15 / 27 |

**Deviation from the brief:** barriers alone do not mark a construction zone.
They line ordinary roads throughout nuScenes and would have labelled most scenes.

## Queries (`benchmark_cvpr/queries_semantic_edge_cases.json`)

20 atomic queries (4 phrasings × 5 flags) and 20 compositional queries over
**12 predicates**. A conjunction qualifies only with ≥ 5 positives among the 258
human-labelled eval scenes — a rule fixed from label counts before any ranker
ran — and 8 predicates carry two phrasings. Several conjunctions from the brief
were excluded for having 1–4 positives rather than being evaluated on noise:
every VRU conjunction, heavy vehicle × low contrast, and pedestrian × night ×
glare. Compositional ground truth combines nuScenes annotation (object),
the scene description (night/rain), and the human rater (glare, wet road,
droplets, low contrast).

## A contamination finding that shapes the evaluation

The 700 corpus nuScenes scenes **are the official nuScenes train split**
(422 + 278). BEVFormer, the source of the BEV embedding `e` and risk `r`, was
trained on their 3D boxes. The BLIP-2 tag head, the source of `t`, was supervised
with nuPrompt tags for all 850 nuScenes scenes, and those tags include
`object_pedestrian`, `object_bus`, `object_truck` and `scene_construction_zone`.
So on the corpus scenes, any ranker that reads `e`, `r` or `t` has seen labels
derived from the same annotations as this ground truth. Those rows are marked
**[contam.]**. The clean rankers use only frozen web-trained signals: SigLIP2,
the Qwen margins, and v2.3 in monocular/portable mode.

A second, **fully clean** evaluation uses the 150 official nuScenes **val**
scenes, which nothing in this project trained on. Only portable signals exist
there. Lens/road degradation has no human labels on val150, so only atomic
queries and the night/rain conjunctions are graded there (24 queries).

## Results — corpus, 258 eval scenes with human labels

| ranker | atomic (20) | compositional (20) | all (40) |
|---|---|---|---|
| **zero-shot SigLIP2** | **0.3370** | 0.2865 | **0.3117** |
| **v2.3A portable** (post-hoc) | 0.3318 | **0.2903** | 0.3111 |
| v2.3A monocular (post-hoc) | 0.3277 | 0.2894 | 0.3085 |
| v2.3 portable base | 0.2691 | 0.2748 | 0.2719 |
| v2.3 monocular | 0.2295 | 0.2449 | 0.2372 |
| metadata matched | 0.2171 (chance) | 0.2337 | 0.2254 |
| BLIP-2 tag lookup [contam.] | 0.2534 | 0.1896 | 0.2215 |
| logit VLM lookup | 0.2171 (chance) | 0.2201 | 0.2186 |
| v2.3 full [contam.] | 0.2067 | 0.2189 | 0.2128 |
| flat VLM lookup | 0.2171 (chance) | 0.1776 | 0.1973 |
| random | 0.2171 | 0.0552 | 0.1361 |

Paired permutation tests (atomic over 20 queries; compositional over 12
predicates, phrasings averaged first):

| comparison | atomic Δ (p) | compositional Δ (p) |
|---|---|---|
| v2.3 monocular − SigLIP2 | −0.1075 (0.0001) | −0.0359 (0.31) |
| v2.3 full [contam.] − SigLIP2 | −0.1302 (0.0001) | −0.0664 (0.09) |
| v2.3A monocular − SigLIP2 | −0.0093 (0.003) | +0.0043 (0.80) |
| **v2.3A monocular − logit lookup** | **+0.1106 (0.0001)** | **+0.0670 (0.03)** |

Atomic by category (mAP; positives in parentheses):

| category | SigLIP2 | v2.3A mono | v2.3 mono | tag lookup [contam.] | chance |
|---|---|---|---|---|---|
| pedestrian_close (77) | **0.423** | 0.414 | 0.298 | 0.342 | 0.312 |
| vru (42) | **0.240** | 0.237 | 0.180 | 0.193 | 0.179 |
| construction_zone (53) | **0.542** | 0.528 | 0.310 | 0.182 | 0.223 |
| heavy_vehicle (58) | 0.348 | 0.326 | 0.234 | **0.392** | 0.238 |
| rare_or_child (30) | 0.132 | 0.133 | 0.125 | **0.158** | 0.134 |

`rare_or_child` is at chance for every ranker: a union of children, strollers,
police and debris is not something any of these signals ranks. The contaminated
tag readout helps only on heavy vehicles. Even with memorised supervision, it
does worse than SigLIP2 on pedestrians and construction.

## Results — val150, fully clean

| ranker | atomic (20) | night/rain conjunctions (4) | all (24) |
|---|---|---|---|
| zero-shot SigLIP2 | **0.3824** | 0.3959 | 0.3847 |
| **v2.3A portable** (post-hoc) | 0.3807 | **0.4543** | **0.3930** |
| v2.3 portable | 0.3128 | 0.3981 | — |
| logit VLM lookup | 0.2200 (chance) | 0.2864 | 0.2311 |
| metadata matched | 0.2200 (chance) | 0.2380 | 0.2230 |

v2.3 portable − SigLIP2 on atomic: −0.0696 (p = 0.008). v2.3A matches SigLIP2.
Only 4 conjunctions have ≥ 3 positives on val150, so the compositional column
is indicative only.

## Why v2.3 fails here, and the post-hoc fix

The concept gate does recognise these queries as out of vocabulary (median γ =
0.09 on atomic queries). Two things still go wrong:

1. **The residual** was trained only on degradation queries and dominates
   semantic ones. "a work zone with orange cones" (γ = 0.04): base 0.566 ≈
   SigLIP2 0.569, monocular 0.417, full 0.224.
2. **Scaling the margin weight by γ is too weak.** v2.3 learned to give SigLIP2
   only ~0.13 of the routing mass, so at γ = 0.28 the margin still holds ~59 %.

**v2.3A** (post-hoc, `use_abstain`, no new hyperparameter) reads the gate
literally: the learned head only knows in-vocabulary concepts, so out of
vocabulary it abstains to the zero-shot prior, `s_A = γ·s + (1−γ)·p_siglip`.
It was designed **after** seeing these Stratum-5 results, so it was evaluated
on every benchmark rather than only this one:

| benchmark | v2.3 | v2.3A |
|---|---|---|
| human GT zero-leakage (550) | **0.7897** | 0.7613 |
| BDD100K (12) | **0.6130** | 0.6073 |
| 150-suite mAP | **0.6313** | 0.5843 |
| 150-suite top-5 skew | +17.0 | **+0.9** |
| Stratum 5 atomic (258) | 0.2295 | **0.3277** |
| Stratum 5 atomic, clean val150 | 0.3128 | **0.3807** |

It is a trade-off, not a free gain. The in-domain cost is almost entirely
low_contrast: its canonical query sits at γ = 0.409, so abstention pulls it
toward SigLIP2, which is near-random on that attribute (0.6548 → 0.5416). γ was
not re-tuned to avoid this.

## What this means for the paper

The defensible claim is narrower than the brief's: SceneMiner's learned head
adds value **within** the vocabulary it was trained on, and outside it the
right behaviour is to defer to the open-vocabulary prior. A fixed-column VLM
lookup cannot do even that, and falls to chance. "Seamlessly handles both
semantic objects and lens degradation better than SigLIP2" is not supported.

## Artefacts

| path | contents |
|---|---|
| `scripts/extract_human_semantic_gt.py` | human semantic GT from nuScenes tables |
| `scripts/build_semantic_queries.py` | the 40 Stratum-5 queries |
| `scripts/eval_semantic_edge_cases.py` | corpus + clean val150 evaluation |
| `benchmark_cvpr/human_semantic_gt.json` | 850 scenes (700 corpus + 150 val) |
| `benchmark_cvpr/queries_semantic_edge_cases.json` | 40 queries |
| `results/cvpr/milestone15_semantic_edge_cases.json` | per-query and summary results |
| `src/scene_mining/filter/sceneminer_v2_3.py` | adds `use_abstain` (off by default) |
| `results/cvpr/milestone14_v2_3.json` | now also carries v2.3A rows; v2.3 numbers unchanged |
