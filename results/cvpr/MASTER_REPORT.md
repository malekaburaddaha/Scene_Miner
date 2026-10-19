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

# Where this leaves the paper

**The strongest defensible claim is now:** SceneMiner v2.1 leads the
sensor-degradation stratum (0.6608 vs 0.6074) and the held-out compositional
stratum, and with a small human calibration set reaches 0.7924 mean mAP against
human ground truth — ahead of every baseline.

**The claims that do not survive:** the downstream detector-utility argument
(§6.8) does not reproduce on uncontaminated data; the collapse magnitudes in
Table 1 and §4.2 do not reproduce; §4.3's BLIP-2 degeneracy does not reproduce;
and the system does not transfer to a monocular external corpus.

**The honest frame:** this is a paper about *diagnosing* a retrieval index and
about what human-calibrated supervision buys — not about a detector-grounded
mining system. Row A_true demonstrates the central thesis cleanly (a modality
100 % dead, 1 distinct row, std 0.0000, while mAP moves 0.004), and the
de-confounded Table 2 shows deterministic labels are the entire effect.
