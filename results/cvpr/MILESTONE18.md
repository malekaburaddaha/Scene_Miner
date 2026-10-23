# Milestone 18 — SceneMiner v2.4: one model, train-derived contrastive concept gate

## Verdict up front

v2.4 replaces both v2.3 and v2.3A. It is the first model in the project to
**beat zero-shot SigLIP2 on BDD100K** (0.6387 vs 0.6218). It keeps v2.3's
in-domain accuracy (0.7879 zero-leakage vs 0.7897), with no low_contrast
collapse. It matches SigLIP2 on semantic edge cases, and it fixes the domain
skew (+0.5 pts). It pays for this on in-vocabulary *compositional* queries,
where the gate now partially abstains.

| target | v2.4 | met | (v2.3 / v2.3A) |
|---|---|---|---|
| human GT zero-leakage ~0.79 | **0.7879** | ✓ | 0.7897 / 0.7613 |
| human GT calibrated ~0.83 | 0.8165 | ✗ (−0.012) | 0.8282 / — |
| no low_contrast drop | **0.6448** | ✓ | 0.6548 / 0.5416 |
| BDD100K > 0.625 | **0.6387** | ✓ | 0.6130 / 0.6073 |
| keep in-vocab BDD win | **+0.109** over SigLIP2 | ✓ | +0.090 |
| Stratum 5 atomic ≈ SigLIP2, +0.11 over lookup | 0.3371 vs 0.3370; **+0.120** (p = 0.0001) | ✓ | 0.2295 / 0.3277 |
| val150 semantic × night/rain ~0.45, beats SigLIP2 | **0.3778** vs 0.3959 | ✗ | — / 0.4543 |
| suite skew < 5 pts | **+0.5** (CI −5.3…+6.3) | ✓ | +17.0 / +0.9 |
| suite mAP stays high | 0.5941 | partly (v2.3 0.6313) | |

## A correction to the brief, applied before building

The brief proposed foils "clear daytime driving, dry road, overcast sky, snow,
fog, dense traffic, empty road, pedestrian, cyclist, construction zone, truck".
Several of these (overcast, snow, fog, pedestrian, cyclist, construction, truck)
are exactly the BDD100K and Stratum-5 **evaluation** queries the gate failed on
or will be scored on. Using them as foils would tune the gate on the test sets,
the opposite of what this milestone is for. **The foils are therefore only
training material:** the 84 macro-environmental phrasings of the 390-query
TRAIN split and the 12-phrase generic bank fixed in Milestone 14. No evaluation
query or label was read before the gate and τ were frozen.

## The gate (`src/scene_mining/filter/sceneminer_v2_4.py`)

* **Classes.** Four concept classes, from the 156 sensor-degradation phrasings
  of the TRAIN split grouped by attribute. Seven macro foil classes, from the 84
  TRAIN macro phrasings grouped by predicate (night, day, rain, dry, dense,
  sparse, moderate). Twelve singleton OOV classes. In total 252 phrasings and 23
  classes.
* **Class score.** A soft maximum over members,
  `S_k(q) = τ · log mean_i exp(cos(q, b_i)/τ)`, on embeddings mean-centred with
  the **training bank's own mean**.
* **Specificity.** `γ(q) = Σ_{4 concepts} softmax(S(q)/τ)`, a contrastive
  posterior in which every concept must beat every foil. The margin column is
  chosen by `softmax(S_concepts/T_a)`.
* **One score.** `s = γ·s_head + (1−γ)·p_siglip`, always on, with no second
  variant.
* **τ chosen on training text only**, by leave-one-out over the bank
  (`scripts/check_v24_gate_train.py`, `results/cvpr/milestone18_gate_train_check.json`):

| τ | LOO pos p05 | LOO neg p95 | separation |
|---|---|---|---|
| 0.01 | 1.000 | 0.000 | 1.000 |
| 0.015 | 1.000 | 0.000 | 1.000 |
| **0.02** | 0.999 | 0.000 | **0.999** ← chosen |
| 0.03 | 0.983 | 0.001 | 0.982 |
| 0.05 | 0.889 | 0.010 | 0.879 |

  Rule, fixed on training data: the largest τ whose LOO separation is ≥ 0.99,
  i.e. the smoothest gate that still separates. The separation optimum, τ = 0.01,
  sits at the grid edge and is a near-hard switch.

**Pre-evaluation verification (LOO, training text only), as the brief asked:**

| group | n | γ min | γ median | share γ > 0.9 | share γ < 0.1 |
|---|---|---|---|---|---|
| droplets | 44 | 1.000 | 1.000 | 1.00 | 0 |
| glare | 40 | 1.000 | 1.000 | 1.00 | 0 |
| wet_road | 36 | 1.000 | 1.000 | 1.00 | 0 |
| low_contrast | 36 | 0.512 | 1.000 | 0.92 | 0 |
| every macro group (7 × 12) | 84 | 0.000 | 0.000 | 0 | **1.00** |
| generic OOV bank | 12 | max 0.009 | — | — | 1.00 |

All 60 atomic degradation phrasings score γ = 1.000. The only sub-0.9 cases
are 3 compound phrasings ending "…on an empty road".

## On "removing the test-informed caveats"

Every statistic v2.4 uses is now derived from training material: the text
mean, the bank, τ, PartNorm and ResNorm running statistics (the last two were
always computed on the 906 training scenes). That makes the **new gate** fully
clean. It does not erase the history of two **design choices** — that text
mean-centring was adopted after a BDD100K diagnostic (Milestone 10), and that
ResNorm was motivated by a decomposition on the human-labelled scenes (Milestone
14). Re-estimating a statistic on training data cannot undo the fact that a
decision was informed by a test set. What we can now add is train-only
evidence for both choices: mean-centring is required for the gate to separate
the training bank, and ResNorm's scale gap (raw residual std 6.31 vs base-part
stds 0.03–0.57) is visible from training statistics alone. The paper keeps a
one-line disclosure, now with this supporting evidence.

## Results

### (A) Human ground truth, 550 scenes

| ranker | macro mAP |
|---|---|
| v2.3 + calibration | 0.8282 |
| **v2.4 + calibration** | 0.8165 |
| v2.3 | 0.7897 |
| **v2.4** | **0.7879** |
| v2.4 base only | 0.7825 |
| v2.4 monocular | 0.7681 |
| v2.4 without ResNorm | 0.7668 |
| v2.3A | 0.7613 |
| logit VLM lookup | 0.7325 |
| zero-shot SigLIP2 | 0.5688 |

Per attribute: droplets 0.9450, glare 0.5836, wet_road 0.9782, low_contrast
**0.6448** (γ = 1.000; v2.3A had 0.5416). All four canonical queries have
γ = 1.000, so abstention, gate and no-gate give identical scores here. Glare
false-negative recovery is 6/20 (v2.3: 7/20).

### (B) BDD100K, 1000 images, 12 queries, no BDD tuning

| ranker | all | in-vocab (4) | OOV (8) | P@5 | NDCG@10 |
|---|---|---|---|---|---|
| **v2.4** | **0.6387** | **0.7087** | 0.6038 | **0.850** | **0.827** |
| v2.4 without gate scaling | 0.6342 | 0.7087 | 0.5970 | 0.833 | 0.820 |
| v2.4 without abstention | 0.6292 | 0.7087 | 0.5894 | 0.833 | 0.813 |
| zero-shot SigLIP2 | 0.6218 | 0.6002 | **0.6327** | **0.850** | 0.809 |
| v2.3 | 0.6130 | 0.6904 | 0.5743 | 0.800 | 0.813 |
| logit VLM lookup | 0.5871 | 0.6475 | 0.5569 | 0.717 | 0.705 |

Per query γ: "bright daytime driving" **0.000** (was 0.992), "driving after
dark" 0.000, "a night-time street scene" 0.000, "a snowy street scene" 0.000,
"driving in rain" 0.000, "adverse weather on the road" 0.000. The two remaining
false triggers are **"overcast grey sky" → low_contrast (γ = 0.999)** and
**"snow covering the road" → wet_road (γ = 0.485)**. They cost the OOV split
its last 0.029 against SigLIP2. No training phrasing describes overcast skies
or snow, so a train-derived gate has nothing to contrast them with. Adding such
foils now would be tuning on BDD100K, so they were not added.

### (C) Stratum 5, 100 % human ground truth

| | v2.4 | SigLIP2 | logit lookup | v2.3A | v2.3 |
|---|---|---|---|---|---|
| corpus atomic (20) | **0.3371** | 0.3370 | 0.2171 | 0.3277 | 0.2295 |
| corpus semantic × condition (20) | 0.2852 | 0.2865 | 0.2201 | 0.2894 | 0.2449 |
| clean val150 atomic (20) | **0.3827** | 0.3824 | 0.2200 | 0.3807 | 0.3128 |
| clean val150 × night/rain (4) | 0.3778 | 0.3959 | 0.2864 | **0.4543** | 0.3981 |

v2.4 vs logit lookup: atomic +0.1200 (p = 0.0001), compositional +0.0616
(p = 0.065, 12 predicates). v2.4 vs SigLIP2: +0.0001 atomic (p = 0.67),
−0.0011 compositional. The contrastive gate puts γ ≈ 0 on object queries, and
also on "…at night"/"…in the rain" conjunctions, because night and rain are
foil classes. So v2.4 *is* SigLIP2 on semantics. The val150 conjunction win of
v2.3A (4 queries) came from its looser gate letting part of the margin through.
v2.4 does not reproduce it.

### (D) 150-query suite, 602 eval scenes

| ranker | mAP | top-5 skew | S1 sensor | S2 macro | S3 compos. | S4 negated |
|---|---|---|---|---|---|---|
| v2.3 | **0.6313** | +17.0 | 0.760 | 0.497 | **0.734** | 0.462 |
| **v2.4** | 0.5941 | **+0.5** | 0.746 | 0.489 | 0.459 | **0.518** |
| v2.4 without abstention | 0.6438 | +20.3 | 0.752 | 0.531 | 0.735 | 0.493 |
| zero-shot SigLIP2 | 0.4446 | −16.2 | 0.444 | 0.487 | 0.370 | 0.355 |

Mean γ by stratum: sensor 0.984, macro 0.016, compositional **0.431**,
negated 0.749. The skew fix and the S3 loss have the same cause: abstention
moves compositional degradation × macro queries partly toward SigLIP2. On this
suite relevance is VLM-defined (circular), so the S3 number overstates what is
lost against human truth. It is still a real behavioural cost.

## What v2.4 settles and what it does not

* **Settled:** one model, no post-hoc variant, a gate with zero test-set input,
  the best BDD100K result in the project and the first above SigLIP2,
  in-domain accuracy kept, skew fixed, semantics no worse than SigLIP2.
* **Not settled:** compositional degradation × condition queries (suite S3,
  val150 S5b). A gate built on "concept vs foil" treats a query that names
  both as half-foil. Milestone 19 examines conjunctions directly.

## Artefacts

| path | contents |
|---|---|
| `src/scene_mining/filter/sceneminer_v2_4.py` | v2.4 gate + abstention |
| `scripts/check_v24_gate_train.py` | LOO gate verification on training text |
| `scripts/train_sceneminer_v24.py` | trainer (same recipe as v2.3) |
| `scripts/eval_v23.py` | now `--model v24` |
| `scripts/eval_semantic_edge_cases.py` | now includes v2.4 rows |
| `results/cvpr/milestone18_v2_4.json` | human GT, BDD100K, suite |
| `results/cvpr/milestone18_semantic_edge_cases.json` | Stratum 5 with v2.4 |
| `results/cvpr/milestone18_gate_train_check.json` | LOO τ grid and per-group γ |
