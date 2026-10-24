# SceneMiner — CVPR rebuild, master report
**Session: 18–19 Oct 2026.** All numbers produced in this session from the
verified August feature store; nothing is carried over from the lost September work.

---

## Headline

**SceneMiner v2 is the first configuration that beats every baseline against
human ground truth** — 0.7587 mean mAP vs 0.7325 for the strongest baseline and
0.4202 for v1. The gain comes from giving the head signals it never had
(SigLIP2 embeddings, continuous VLM margins), supervising all four attributes,
and down-weighting the VLM's borderline calls.

Against that: the 3D detector gate is **inconclusive**, and three separate
claims in the submitted paper **do not reproduce**.

---

## 1. Human ground truth (600 scenes, rater 1)

Your annotation file was corrupted by a stray `P` character (an IDE keystroke
while the server held the file open); all 600 scenes were recovered.
530 `flag=None`, 20 `skip` (all-False), 50 `unsure` (all-False) →
Mode A = 530 scenes, Mode B = 550. Results differ by <0.006 between modes.

### The VLM is not an oracle once humans grade it
| attribute | human+ | precision | recall | F1 | κ | FP | FN |
|---|---|---|---|---|---|---|---|
| droplets | 79 | 0.644 | 0.962 | 0.772 | +0.724 | 42 | 3 |
| glare | 41 | 0.583 | 0.512 | 0.545 | +0.511 | 15 | 20 |
| wet_road | 85 | 0.683 | 0.988 | 0.808 | +0.765 | 39 | 1 |
| low_contrast | 66 | 0.495 | 0.742 | 0.594 | +0.526 | 50 | 17 |

**Precision 0.50–0.68.** This is the circularity critique, quantified: grading
against human labels collapses the flat lookup from a tautological **1.000**
(predicate relevance) to **0.539**.

### Retrieval vs human GT (Mode B, 550 scenes)
| ranker | droplets | glare | wet_road | low_contrast | MEAN |
|---|---|---|---|---|---|
| **SceneMiner v2 (5 seeds)** | **0.9143** | **0.6416** | **0.9663** | 0.5126 | **0.7587** |
| logit VLM lookup | 0.8238 | 0.5887 | 0.9653 | **0.5521** | 0.7325 |
| v2 base only (zero-shot prior) | 0.7498 | 0.5994 | 0.8149 | 0.3585 | 0.6307 |
| metadata-matched | 0.7080 | 0.3104 | 0.6835 | 0.7440 | 0.6115 |
| zero-shot SigLIP2 | 0.9129 | 0.4144 | 0.8433 | 0.1045 | 0.5688 |
| flat VLM lookup | 0.6427 | 0.3883 | 0.6896 | 0.4369 | 0.5394 |
| SceneMiner v1 row E | 0.9077 | 0.2031 | 0.4813 | 0.0888 | 0.4202 |
| chance | 0.1436 | 0.0745 | 0.1545 | 0.1200 | 0.1232 |

Leakage: **0 of 600 human-labelled scenes are in the training split.**
Significance vs the logit lookup: droplets **+0.091** and glare **+0.053** both
exceed 2 sd; wet_road is a tie; low_contrast **−0.040** is a significant loss.

**Still unresolved:** fusion does not rescue VLM errors. On glare's 20 false
negatives, v2 recovers 10/20 in the top-20 %, its own base 11/20, plain
zero-shot SigLIP2 **14/20**.

---

## 2. The 3D gate — inconclusive, and here is why

`bevformer_base` validated at **mAP 0.4164 / NDS 0.5173** (published numbers).
A hand-rolled per-scene matcher produced an **inverted** signal (night 150×
easier); rebuilt on the official `accumulate`/`calc_ap` it behaves correctly
(night 0.4171 < day 0.4553).

**Contamination quantified** — all 278 of our held-out nuScenes scenes are in
the official *train* split:

| set | per-scene mAP | sd |
|---|---|---|
| our 278 (memorised) | 0.7864 | 0.068 |
| 150 official val (unseen) | 0.4514 | 0.100 |
| **inflation** | **+0.3349 (+74 %)** | variance compressed |

Gate result (top-50 hardness, higher = harder): flat lookup −0.7629 > logit
−0.7665 > metadata −0.7695 > SigLIP2 −0.7742 > **SceneMiner −0.7821** > random
−0.7877. Within the positive bucket SceneMiner's correlation with hardness is
**negative** (ρ = −0.21 to −0.24). The decile curve is flat.

**But all rankers sit within 0.02 mAP on a signal with sd 0.068 (~0.3σ)**, and
on *clean* val these conditions cost BEVFormer only −0.038 (night) and −0.006
(rain). The gate is not decisive; the premise that these conditions "break
perception" is weaker than the paper assumes.

---

## 3. Three paper claims that do not reproduce

| claim | paper | measured here |
|---|---|---|
| Table 1 dead risk token | 1 → 1035 distinct rows | **reproduces exactly** (std 0.0000) |
| Table 1 fused effective rank | 18–25 → 105–118 | 45.6 → 72.0 |
| Table 1 routing mass on risk | 0.315 → 0.205 | 0.117 → 0.137 (visual dominates at 0.583) |
| Table 1 score span | 213.7 → 9.8 | 7.35 → 9.96 |
| §4.2 learned τ "drifted below one" | τ < 1, saturation | **τ = 1.070 (3k), 1.249 (12k)** — drifts *up*; removing the clamp changes nothing |
| §4.3 BLIP-2 degeneracy | erank 41.2/256, 6 tokens <1 % mass | **180.7/256, 0 tokens <1 %** |
| §4.3 erank(L) null | 47.6 | 8.04 atomic / 22.39 over 233 queries (**matrix rank 38 < 48**, so 47.6 is unreachable with this vocabulary) |

Two bugs in my own first pass were found and fixed: BLIP-2 rank was measured on
a 32×768 concatenation (913, meaningless), and erank(L) on 11 atomic columns
(capped at 11 by arithmetic).

### The reframing you were right about
Row A_true, with the risk channel genuinely dead (1 distinct row, std 0.0000),
scores **0.311 ± 0.036** vs Row A's 0.307 ± 0.048. I previously called this a
refutation. **It is the paper's thesis demonstrated:** a modality can be 100 %
dead while the task metric is undisturbed.

### De-confounded Table 2 (5 seeds, scene-disjoint)
| Row | mAP | Δ vs A |
|---|---|---|
| A_true (true pre-repair) | 0.311 ± 0.036 | — |
| A (legacy_scalar) | 0.307 ± 0.048 | −0.004 |
| A1 (+ stratified negatives only) | 0.294 ± 0.011 | **−0.013** |
| A2 (+ deterministic labels only) | 0.526 ± 0.013 | **+0.219** |
| B (both) | 0.537 ± 0.022 | +0.230 |
| E (+ VICReg + bounded) | 0.505 ± 0.020 | −0.032 vs B |

**Deterministic labels are the entire effect.** Stratified negatives contribute
nothing; VICReg + bounded scoring is negative.

---

## 4. Assets built

| asset | state |
|---|---|
| `benchmark_cvpr/split_906_602.json` | frozen, segment-disjoint, 0 overlap |
| `binary_flags_with_logits.json` | 1,508 × 4 attributes + Yes/No log-prob margins |
| `lowcontrast_strict.json` | sharpened prompt: prevalence **55.6 % → 17.1 %**, margins 0.61 → 1.12 |
| `features_v2.npz` | + SigLIP2 image embeddings, + 4 continuous VLM margins |
| `benchmark_cvpr/queries_390.json` | 233 queries (126 train / 107 eval), 4 strata, disjoint |
| `human_annotations_rater1_600.json` | 600 scenes, repaired |
| BEVFormer full infos | 34,149 samples, private data root |

---

## 5. Status against the five milestones

| milestone | state |
|---|---|
| 1 — 3D gate + 3 diagnostic fixes | **complete** |
| 2 — SceneMiner v2 | **complete** |
| 3 — 390-query suite, domain skew, Tables 2/3 re-run | **partial**: suite built at 233/390; domain-skew z-scoring and the full Table 3 re-run **not done** |
| 4 — ACDC / BDD100K external dataset | **not started** (needs manual download; not feasible in the window) |
| 5 — repo polish + master report | **complete** |

### Highest-value next steps
1. Expand the suite to the full 150/240 and re-run Tables 2 and 3 on it.
2. Second human rater → inter-rater κ, which the paper needs.
3. Per-domain z-scoring for the 12.1-point nuScenes skew + balanced Figure 2.
4. ACDC for an external, human-annotated adverse-conditions holdout.
5. Decide the paper's frame: v2's win is real but modest (+0.026 over the best
   baseline, 2 wins / 1 tie / 1 loss). The diagnostic protocol remains the
   safer anchor, and Row A_true now demonstrates its central claim cleanly.


---

# Part II — Milestones 6–9 (19 Oct 2026)

## 6. SceneMiner v2.1 and what human calibration buys

v2's single global blend (~0.60 SigLIP2 / 0.40 VLM margin) is wrong whenever the
two signals disagree about which query they are good at — and they do
(SigLIP2 0.913 on droplets vs 0.105 on low_contrast). v2.1 replaces it with a
query-conditioned routing distribution over three base signals plus a
query-conditioned residual gate, all zero-initialised (routing exactly uniform,
residual inert at step 0 — verified).

| ranker (Mode B, 550 human-labelled scenes) | droplets | glare | wet_road | low_contrast | MEAN |
|---|---|---|---|---|---|
| **v2.1 + 5-fold human calibration** | 0.8545 | **0.7035** | 0.9616 | **0.6499** | **0.7924** |
| SceneMiner v2 | **0.9143** | 0.6416 | 0.9663 | 0.5126 | 0.7587 |
| v2.1 zero-leakage | 0.8991 | 0.5754 | 0.9639 | 0.5035 | 0.7355 |
| logit VLM lookup | 0.8238 | 0.5887 | **0.9653** | 0.5521 | 0.7325 |
| zero-shot SigLIP2 | 0.9129 | 0.4144 | 0.8433 | 0.1045 | 0.5688 |

**The routing collapsed.** Across all five seeds and all four attributes it puts
0.967–0.992 of its mass on the VLM margin — because the training labels *are*
VLM-derived, so copying the VLM is optimal. That is why v2.1 zero-leakage
(0.7355) is worse than v2: it discarded SigLIP2's droplets advantage.

**What human labels buy.** Refitting only the three routing weights on 4/5 of
the human scenes and scoring on the held-out fifth moves low_contrast
0.5035 → 0.6499 and glare 0.5754 → 0.7035. **A 440-scene calibration set is
worth more than the architecture change.** (Fold variance is large: ±0.118 on
low_contrast.) Neither version reaches 4/4 wins, and glare FN recovery is still
best for plain SigLIP2 (14/20 vs v2.1's 7/20).

## 7. The suite is exact; the skew fix is not recommended

`queries_390.json` — **240 train / 150 eval**, strata **62 / 58 / 18 / 12**,
100 % disjoint text. A leakage bug was caught: 8 compositional predicates had
entered the training pool, which would have falsified the "never supervised"
claim. Fixed and verified at 0 overlap.

**Per-domain z-scoring overcorrects and is not recommended.** SceneMiner's skew
goes from +8.7 points to **−23.8**, and mAP drops 0.5312 → 0.4925. The residual
skew on this larger suite is 8.7 points, not the 12.1 the paper reports.

Table 3, sensor-degradation stratum (the one the paper is about, and the one
where no tautology applies):

| ranker | mAP |
|---|---|
| **SceneMiner v2.1** | **0.6608** |
| logit VLM lookup (calibrated) | 0.6074 |
| flat VLM lookup | 0.5790 |
| zero-shot SigLIP2 | 0.4440 |

v2.1 also leads the held-out compositional stratum among non-tautological
rankers (0.5496 vs 0.3610). ⚠ The metadata prior scores **exactly 1.0000** on
the macro stratum — the tautology again, since macro predicates read the same
fields it stores. That figure must not appear as a baseline.

## 8. A third dataset, and the cleanest version of the gate

**BDD100K** (ungated Hub mirror, no login): 1,000 stratified images with native
human `weather`/`timeofday` labels — night 493, rainy 94, snowy 82.

| ranker | mAP (chance 0.262) |
|---|---|
| **zero-shot SigLIP2** | **0.6218** |
| logit VLM lookup | 0.5910 |
| SceneMiner v2.1 (portable routing) | 0.4177 |

**The learned component does not transfer; the zero-shot component does.** And
the full head cannot be evaluated on BDD at all — its Stage-1 signals need a
calibrated 6-camera rig with 3D boxes. **The architecture is not portable to the
kind of corpus the Area Chair asked for**, which belongs in the paper.

**Uncontaminated 150-scene nuScenes val 3D study** (BEVFormer never saw these):

| ranker | top-30 mAP | Δ vs corpus |
|---|---|---|
| logit VLM lookup | 0.4449 | −0.0065 |
| metadata-matched | 0.4477 | −0.0038 |
| random | 0.4524 | +0.0010 |
| zero-shot SigLIP2 | 0.4546 | +0.0031 |

**No ranker finds measurably harder scenes** — every delta is under 0.07 σ.
Retrieved sets score *higher* at the tight 0.5 m threshold, and random draws the
highest low-visibility fraction. **§6.8's −0.074 gap does not reproduce on
uncontaminated data in any direction.**

## 9. §4.2 reconciled, partially

CLIP's multiplicative `logit_scale` is the better account of saturation, but not
a sufficient one:

| configuration | score span | effective support (of 1,508) |
|---|---|---|
| learned τ (divide) | 7.35 | 655.26 |
| **CLIP logit_scale** | **15.08** | **299.47** |
| paper §4.2 | 213.7 | 1.06 |

Direction confirmed, magnitude not. The scale settled at 12.98, *below* its
14.29 init. Inside the 16-scene training batch support does collapse (2.0/16),
so saturation is real in kind but not in degree. **The 213.7 figure should be
re-derived or dropped.**

---

# Part III — Milestones 10–13 (19 Oct 2026)

## 10. SceneMiner v2.2 and the real cause of the transfer failure

The routing collapse was real and is fixed: mean routing weight on the VLM
margin falls from **0.980 to 0.535**, via channel dropout (50 % margin, 30 %
3D), an entropy floor capping any channel at 0.70, cross-modal agreement
distillation (37.4 % of pairs down-weighted for teacher disagreement), and
explicit modality availability. No human label enters training.

**The milestone's premise did not survive measurement, and is reported as
such.** A build with the routing fixes *only* moved BDD100K by **+0.0004**. On
BDD100K's phrasings v2.1's router was already distributed ([0.301, 0.392,
0.307]) — the collapse is query-specific, because the router is a linear map of
the query embedding.

The actual mechanism is **scale incommensurability**. v2.1's blend scored
*below both signals it blended* (0.4181 vs 0.6218 and 0.5871), which no genuine
convex combination can do. The margin part carries **28.1×** the spread of the
cosine part (std 0.5419 vs 0.0193), so a nominal 50/50 routing weight was ~96 %
margin. The gap is visible in training too — PartNorm's running stds come out
at [0.031, 0.564, 0.166] — so this was diagnosable without external data.

> **Correction (Milestone 23).** The BDD100K numbers in this section were computed
> with L2-normalised query features, while the heads were trained on raw ones. With
> raw features, v2.1 scores 0.486 and its blend is 0.485 (still below both signals);
> the spread ratio is 30.3× (std 0.583 vs 0.019); and v2.1's router puts 0.977 of its
> weight on the margin on BDD100K too, so the "already distributed" observation
> above was an artefact. Routing repair alone still moves BDD100K by −0.001.

**Three bugs, one in an already-reported number.**

| bug | impact |
|---|---|
| the BDD lookup baseline chose its margin channel by correlation **with the test labels** (ours, Milestone 8) | small: 0.5910 → 0.5871 |
| BDD margin columns permuted vs training order — droplets and wet_road **swapped** into the model | large: v2.2 0.5475 → 0.5952 |
| learned channel selector wrong on 8/12 external queries at p_max 0.26 | fixed by mean-centred text priming, 6/12 → 10/12 |

**Targets: none of the four were met.** Zero-leakage 0.7065 (target > 0.765),
calibrated 0.7656 (> 0.800), glare FN recovery 1/20 (≥ 13/20, and *worse* than
v2.1's 7/20), BDD 0.5952 (vs SigLIP2 0.6218). The one target met is beating the
logit lookup (0.5871).

**The defensible claim is a trade-off, not a win:** routing weight on a VLM
teacher is an in-domain asset and an out-of-domain liability. Capping it costs
0.029 mAP in domain and buys 0.176 externally.

## 11. The premise holds, once measured correctly

Milestone 8 found no link between degradation and detector error (< 0.07σ).
That test averaged scene mAP over six cameras and 360° while the degradation
signal comes from one camera. Only **23.4 %** of nuScenes GT lies in the
CAM_FRONT FOV (half-angle 32.6°, from the camera's own intrinsics).

Measuring degradation **and** error in the same sector, on the 150
uncontaminated val scenes, 42 degraded vs 108 clean:

| metric (front sector) | degraded | clean | Δ | σ | q (BH) |
|---|---|---|---|---|---|
| mAP | 0.3886 | 0.4792 | −0.0906 | −0.69 | **0.015** |
| FP rate @ score>0.3 | 0.3503 | 0.2775 | +0.0729 | +0.50 | **0.045** |
| mATE (m) | 0.8330 | 0.7553 | +0.0777 | +0.44 | 0.080 |
| recall > 30 m | 0.5312 | 0.6079 | −0.0767 | −0.42 | 0.080 |
| recall ≤ 30 m | 0.7703 | 0.8262 | −0.0559 | −0.44 | 0.080 |
| mAOE (rad) | 0.5008 | 0.4308 | +0.0701 | +0.22 | 0.489 |

4/30 tests survive Benjamini–Hochberg at q<0.05, 12 at q<0.10. Orientation does
not move — degradation costs detections and positions, not heading. Lens
degradations hit mAP hardest; low contrast hits far recall hardest (−0.0973,
q=0.087). **Glare has only 6 degraded scenes and is underpowered, not null.**

Milestone 8's result stands as measured and is quantitatively consistent with
this one: a −0.69σ front-sector effect diluted across a corpus where 76.6 % of
GT sits outside the front camera is the sub-0.07σ whole-scene signal observed.

**Negative-tail domain calibration.** Milestone 7's full per-domain z-score
overcorrects (skew +8.7 → −23.8, mAP −0.0388). Estimating the offset only from
each domain's negative tail, with (q, α) fit on the 240 train queries and
reported on the 150 held-out eval queries, dominates it for every ranker. Two
findings matter more: **v2.2's raw skew is already +1.1 points** (v2.1: +8.7),
so standardising the parts removed most of the skew as a side effect and v2.2
needs no calibration; and **the ±2.0 point target is below the resolution of a
150-query evaluation**, whose bootstrap CI widths are 9.7–12.5 points.

## 12. Figures and manuscript

Three figures generated directly from the result JSONs, and `paper/main_cvpr.tex`
at 6 pages, organised as three Audit Discoveries plus the v2.2 method plus the
targeted premise test. Every WACV claim that failed to reproduce is **removed,
not softened**: Table 1 magnitudes, the τ-drift claim, §4.3's BLIP-2 degeneracy,
the unreachable erank(L), and §6.8 downstream utility — the last replaced by the
CAM_FRONT analysis, which does reproduce.

---

# Part IV — Milestones 14–17 (23 Oct 2026)

## 14. SceneMiner v2.3 — the diagnosis was partly wrong, the model is better

Two mechanisms proposed for v2.2's trade-off were checked before building and
**did not hold**: on the 20 VLM-missed glare scenes the margin averages +0.80σ
(none below −0.5σ), so no negative margin vetoes SigLIP2; and v2.2 gave SigLIP2
0.21 of the low_contrast weight, not ~0.47. What did hold: **the gated residual
dominated the ranking** with 2–3× the base's spread (raw residual running std
6.31, visible without labels). The base alone scored 0.9609 vs 0.8813 on
droplets and recovered 12/20 vs 5/20 missed glare scenes.

v2.3 = specified concept gate + FN rescue, plus **ResNorm** (standardised,
bounded residual, α = 0.5 a priori).

| benchmark | v2.1 | v2.2 | **v2.3** | target |
|---|---|---|---|---|
| human GT zero-leakage | 0.7355 | 0.7065 | **0.7897** | > 0.76 ✓ |
| human GT calibrated | 0.7924 | 0.7656 | **0.8282** | > 0.795 ✓ |
| glare FN recovery | 7/20 | 1/20 | 7/20 | ≥ 12/20 ✗ |
| BDD100K | 0.4189 | 0.5952 | 0.6130 | ≥ 0.6218 ✗ |
| 150-suite mAP / skew | 0.5312 / +8.7 | 0.5178 / +1.1 | 0.6313 / **+17.0** | skew preserved ✗ |

BDD100K values in this table used L2-normalised queries (corrected in Milestone 23:
v2.1 0.486, v2.2 0.570, v2.3 0.589).

Ablations: gate −0.0123, ResNorm −0.0146, FN rescue −0.0003 (as predicted). On
BDD100K the gate picks the right column on 4/4 in-vocabulary queries and wins
there by +0.090, but loses 0.058 out of vocabulary where weather phrases are
semantically close to a concept ("bright daytime driving" → sun glare, γ 0.99).
No calibration within budget fixes the skew regression.

## 15. Stratum 5 — the expected headline does not hold

Human semantic ground truth from nuScenes 3D annotation (pedestrian, VRU,
construction, heavy vehicle, rare/child; front sector, visibility > 40 %), 40
queries, 12 compositional predicates with ≥ 5 positives.

**Contamination finding:** the 700 corpus nuScenes scenes are the official
nuScenes *train* split, so BEVFormer (e, r) and the nuPrompt-supervised tag head
(t, which includes pedestrian/bus/construction tags) saw labels for them. Those
rankers are flagged, and a fully clean evaluation runs on val150.

v2.3 is **worse than SigLIP2** on semantic queries (atomic 0.2295 vs 0.3370,
p = 0.0001; clean val150 0.3128 vs 0.3824). The fixed-column lookups and
metadata sit at exactly chance (0.2171). A **post-hoc** abstention variant
(v2.3A: `s = γ·s + (1−γ)·p_siglip`) matches SigLIP2 and beats the logit lookup
by +0.111 atomic (p = 0.0001) and +0.067 compositional (p = 0.03), and brings
suite skew to +0.9. It costs human GT 0.7897 → 0.7613 and BDD 0.6130 → 0.6073,
almost entirely through the low_contrast query (γ = 0.409). v2.3 stays the main
model.

## 16. Retrieval → 3D breakage on val150

Portable signals only; null permutes per-scene metrics with retrieved sets fixed.
For the 62 degradation queries, v2.3's top-25: front mAP −0.060 (p 0.001), VRU
recall −0.193 (p < 0.001), far recall −0.043 (p 0.048); harder than SigLIP2's
(mAP −0.028, p 0.004) but **indistinguishable from the VLM-margin lookup and from
night/rain metadata**, the latter being the only ranker with a significant
ghost-FP rise (+0.052, p 0.002). The effect is a threshold confined to the top
quartile. Semantic retrievals (pedestrians, cones, trucks) are not harder than
random for this detector.

## 17. Manuscript and supplementary

`paper/main_cvpr.tex`: 8 pages + references, 4 figures, 9 tables, a
related-work section and a bibliography; every figure inspected after rendering.
`paper/supp_cvpr.tex`: 13 pages, with the full hyperparameter table, both
annotation protocols and a real screenshot of the annotation UI, all 30 BH tests,
per-query results for all 190 evaluated queries, the 240 training queries
(listed, not scored), and extra figures. All tables are generated from JSON.

---

# Part V — Milestones 18–21 (23 Oct 2026)

## 18. One model: SceneMiner (v2.4)

v2.3 and v2.3A are replaced by a single model. Its concept gate is a
**contrastive posterior over 23 classes built only from training text**: four
concept classes (156 TRAIN degradation phrasings), seven foil classes (84 TRAIN
macro phrasings) and 12 generic OOV phrases. τ = 0.02 is chosen by
leave-one-out on that bank. There, every atomic degradation phrasing scores
γ = 1.000 and every foil ≤ 0.010. Abstention toward SigLIP2 is always on. Foils
suggested from BDD100K/Stratum-5 vocabulary (overcast, snow, pedestrian, …)
were **not** used, because that would have tuned on the test sets.

| | SceneMiner (v2.4) | v2.3 | v2.3A |
|---|---|---|---|
| human GT zero-leakage / calibrated | 0.7879 / 0.8165 | 0.7897 / 0.8282 | 0.7613 / — |
| low_contrast | 0.6448 | 0.6548 | 0.5416 |
| BDD100K | **0.6387** (> SigLIP2 0.6218) | 0.6130 | 0.6073 |
| Stratum 5 atomic | 0.3371 (= SigLIP2) | 0.2295 | 0.3277 |
| suite mAP / skew | 0.5941 / **+0.5** | 0.6313 / +17.0 | 0.5843 / +0.9 |

Costs: suite S3 (compositional degradation × macro) drops because those queries
name a foil (γ ≈ 0.43); v2.3A's val150 semantic × night/rain result (0.4543, 4
queries) is not reproduced (0.3778).

## 19. Where learned fusion helps — honestly

Three human-graded regimes against two no-learning controls:

| regime | SceneMiner | SigLIP2 | lookup | SigLIP2+lookup | SigLIP2+metadata |
|---|---|---|---|---|---|
| atomic degradation | **0.789** | 0.569 | 0.733 | 0.737 | 0.626 |
| atomic semantics | 0.337 | 0.337 | 0.217 | = SigLIP2 | = SigLIP2 |
| object × condition | 0.292 | 0.293 | 0.227 | 0.284 | **0.327** |

**The compositional-synergy claim does not hold.** SigLIP2 + metadata beats
SceneMiner on all 11 generated conjunctions (p = 0.0007). Grading val150
conjunctions with VLM labels was declined as circular.

Rater sensitivity: the 50 unsure scenes carry no labels. Removing
CLIP-ambiguous scenes widens the lead over the lookup. A Monte Carlo second rater
(κ 0.90/0.85/0.75) keeps SceneMiner ahead of SigLIP2 in 100 % of trials and
ahead of the lookup in 92–100 % (macro). Per attribute, glare and wet road are
near coin flips, and glare is lost to SigLIP2+lookup fusion.

## 20. Verified bibliography and the official CVPR kit

All 20 references are generated from fetched metadata (arXiv, Crossref,
Semantic Scholar, OpenAlex; `paper/bibcheck/`). Every arXiv id resolves to the
intended paper. Corrections forced by the metadata include the BEVFormer title
(also wrong in the brief's anchor list), nuPrompt at AAAI 2025, the 23-author
published Waymo list, and ACDC cited in its ICCV 2021 version. The paper uses
the official CVPR 2026 author kit. Main text fills pages 1–8 and references
start on page 9.

## 21. Final inspection

Every page of both PDFs was rasterised and inspected. Fixes made in this pass:
a "-12pt" artefact printed in every page header by the kit under TeX Live 2018
(a one-line `\dimexpr` compatibility edit in `cvpr.sty`, identical output on
current TeX); a non-breaking-space bug in two Crossref titles; overflowing
tables; floats drifting away from their sections; and stale v2.3 content in
the supplementary. All qualitative figures show the same number of examples
from each of nuScenes, Waymo and BDD100K.

---

# Part VI — Milestone 22 (23 Oct 2026)

## 22. Factored gate; paper reframed

The S3 / conjunction failures had two distinct causes: context × context queries
lost their metadata on abstention, and some degradation × context compounds were
gated off because one softmax made context compete with the concept. SceneMiner
v2.5 factors the gate: **γ_deg** is computed on context-projected embeddings, and
**γ_macro** maps context classes to recorded-metadata columns that enter the
abstention target. The variant and τ were selected on training text by
leave-group-out (half projection, τ = 0.02).

| | v2.4 | **v2.5 (final)** |
|---|---|---|
| human GT zero-leakage / calibrated | 0.7879 / 0.8165 | 0.7845 / **0.8263** |
| BDD100K (corrected in M23; reported 0.6387 / 0.6317) | **0.6283** | 0.6164 (SigLIP2 0.6218) |
| suite overall / skew / S3 | 0.594 / +0.5 / 0.459 | **0.685** / +2.2 / **0.603** |
| S5 object × condition | 0.292 | **0.310** (SigLIP2+metadata 0.327, gap n.s.) |
| S5 atomic corpus / val150 | 0.337 / 0.383 | 0.332 / **0.392** |

The paper is reframed as a first-principles design and diagnostic study. The
re-audit table of an earlier version is removed; the weaker designs appear as a
design-progression table; the width-1 LayerNorm dead-modality result is in
Section 6; and Equation 1, Figure 1 and Figure 2 are fixed. Main text ends on
page 8, references from page 9.

## 23. Published driving-retrieval paradigms; BDD100K evaluation bug

Twelve methods on identical splits and metrics, including re-implementations of
the BEV-TSR, RefAV, CARIM, Talk2BEV/nuPrompt and ReferGPT paradigms on our cached
signals (`milestone23_related_work_comparison.json`). SceneMiner leads on human
GT (0.785; next 0.735), ties SigLIP2 on BDD100K and Stratum 5, and loses to CLIP
and ReferGPT-style on the 4 val150 conjunctions (untestable, n = 4). The RefAV-style
program's 0.850 on the suite is circularity. Caption retrieval is weak partly
because our caption prompt makes 64 % of captions deny droplets or glare.

An evaluation bug of mine was found: the BDD100K benchmark fed L2-normalised query
features to learned heads trained on raw features. All BDD100K numbers were
recomputed (SceneMiner 0.632 → 0.616, now 0.005 below SigLIP2). The γ_macro
subspace refinement was rejected on training text. Four references were verified
(BEV-CLIP is published as BEV-TSR, AAAI 2025; RefAV is arXiv only).

## 24. Qualitative showcase

Figure 2 has degradation rows and safety-critical road-user rows, top-2 per corpus
for nuScenes, Waymo and BDD100K. Boxes come only from human 3D annotations projected
into the retrieved frame (source frames verified by pixel correlation, r ≥ 0.979);
retrievals without an annotated object of the class are labelled, not hidden. The
supplementary adds a page of semantic strips and a rule-selected failure strip.

## 25. Paper rewrite

Abstract and introduction rebuilt around three contributions (system; calibrated
fusion with diagnostic auditing; multi-dataset human evaluation); related work
covers the driving-retrieval systems; the master comparison table is Table 2.
Main text is 8 pages with references from the top of page 9.

## 27. Cached scene index; conditional context gate (v2.6); portable mode

**Cached index.** Scene tokens are projected once. Query-only work is done once per
query, and the decoder's key/value projections are folded onto the query side.

* **Identity:** identical to the plain forward pass in float64, with AP difference 0 on
  every benchmark.
* **Latency:** 1.24 ms ranking plus 2.27 ms query preparation for one model, vs 12.39 ms
  plain. The < 0.5 ms target was not met.

**v2.6 (final).** It removes only the context directions a query activates. The variant
was selected on training text by the Milestone-22 rule; the worst negative fell from
0.152 to 0.080.

* Human GT: 0.794, 0.829 calibrated.
* Suite: 0.694 / +2.6.
* BDD100K: 0.614. The 0.628 target was not reached: "snow" still activates a weather
  class.

**Portable mode.** It reads no BEV or tags and scores 0.762 on human GT, above every
other method.

## 28. Upgraded baselines; untouched holdout

**CARIM-style.** Neutral captions help semantics and hurt degradations, so both prompts are reported.

**RefAV-style.** Programs are now compiled by a local LLM.

**Holdout.** A pre-registered 36-query benchmark on 2000 BDD100K images never used before,
graded by BDD100K's own tags and 2D boxes:

* SceneMiner 0.400 vs SigLIP2 0.408 (p = 0.28), with identical rankings on object queries;
* above CLIP / ReferGPT-style, CARIM-style, the lookup, the naive router and RefAV-style
  (p ≤ 0.013);
* not significantly above SigLIP v1 (p = 0.42).

## 29. Figures

**Figure 3.** All 42 images are from distinct driving segments.

* Waymo trucks, buses and motorcycles come from labelled vehicle boxes, selected by size.
* BDD100K human 2D boxes are drawn.
* Misses are kept: 13/20 labelled cells show the object.

**Figure 1.** Redrawn as a two-tier diagram from measured, current components.

## 30. Second rater, anonymity, licences

**Second rater.** A 160-scene kit is ready to run; no second-rater data exists yet.

**Anonymity.** The leak checker covers decoded PDF content, metadata, EXIF and sources,
and passes. Figure and paper PDFs carry no creator/producer metadata.

**Licences.** The dataset licence terms were checked against the owners' pages, and the
Waymo attribution footnote was added.

## 31. Final paper

**Title.** "Scale-Aware Multi-Modal Fusion …".

**Tables.**

* Main tables are generated from the result files at 3 decimals.
* Table 2 gains the holdout column and both SceneMiner rows.

**Number check.** `verify_paper_numbers.py` passes 48 registry claims. It caught three
values I had derived from rounded numbers.

**Layout.** Main text is 8 pages, with references from the top of page 9; the
supplementary is 19 pages.

## 32. Figure legibility; object×condition gate cap (rejected)

**No reported number changes.**

**Figures.**

* Figure 3 now spans the full text width. Its thumbnails are 25% larger on the page
  (0.757 → 0.950 in).
* Its labels are 8.5 pt bold condensed; its boxes are lw 2.0 with a dark stroke.
* Its insets cover 45% of the thumbnail height with a 2 pt border; its badges are
  7.5 pt bold.
* The supplementary strips get the same treatment.
* Figure 1 is set in mathtext, with a 7.0 pt minimum.

**Reported typos.** None of the three exists in the source or the rendered PDF.

**Capping γ_deg on object×condition queries.** Not adopted.

* Diagnosis: these queries route almost entirely to the degradation margins.
* The metadata vector is already unit-variance, so normalising it changes nothing.
* The best training-only detector of an extra object component finds 45% of
  synthetic conjunctions at a 1.3% false-flag rate. The pre-declared bar was 80%.
* v2.6 is unchanged, and the conjunction loss to SigLIP2 + metadata stands.

**Layout.**

* Figure 4 moved to the supplementary.
* The ablation and BDD100K summary tables are back in the main paper.
* Main text still fills 8 pages, with the references at the top of page 9.
* The verifier passes 50 claims; the anonymity check passes.

---

# Where this leaves the paper

**Defensible claims (final):**

1. **The premise holds, and retrieval closes the loop.** Front-camera
   degradation costs 0.0906 front-sector mAP and 7.3 points of ghost-FP rate on
   uncontaminated data (q < 0.05). SceneMiner's top retrievals for degradation
   queries lose 0.057 mAP and 0.197 VRU recall.
2. **Two structural defects:** circular predicate benchmarks (lookup = 1.000) and
   certifiers trained on the mined scenes (+0.3349 detector mAP; tag head
   supervised on the same objects).
3. **Three levels of silent modality failure:**
   parts (30.3×), residual (~10×), and a width-1 LayerNorm that kills a modality
   outright while mAP moves 0.004.
4. **One model**, best of twelve methods on human-graded degradation (0.794 / 0.829;
   0.762 in its portable mode, which reads no corpus-supervised signal), level with
   zero-shot SigLIP2 externally (BDD100K 0.614 vs 0.622; untouched pre-registered
   holdout 0.400 vs 0.408, p = 0.28) and on semantic edge cases. Its gate never saw
   evaluation data; the holdout was run once, after every design decision.

**What does not survive:** SceneMiner beating SigLIP2 on BDD100K (an evaluation
bug, fixed in M23); beating SigLIP2 on an untouched external holdout (M28: level);
beating every published paradigm everywhere; learned fusion beating simple SigLIP2 +
metadata fusion on object × condition conjunctions (that fusion is significantly better,
p = 0.016); SceneMiner finding breakage scenes better
than the VLM-margin lookup on val150; a uniform per-attribute lead.

**Before submission, for the authors:** fill in `\paperID`; `cvpr.sty` carries one
marked TeX Live 2018 compatibility edit that can be reverted on Overleaf.
