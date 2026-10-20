# Milestone 10 — SceneMiner v2.2: eliminating teacher-shortcut routing collapse

## Verdict up front

v2.2 **does eliminate the routing collapse** and **recovers most of the external
transfer gap** (+0.1763 mAP, +0.2368 NDCG@10 on BDD100K over v2.1). It does so
at a measured cost of **−0.029 mAP in domain**. **None of the four numeric
targets set for this milestone were met.** The targets assumed the collapse was
the *cause* of the BDD100K failure; it was not the main cause, and removing it
buys external robustness rather than in-domain accuracy. The details below are
reported as measured, including three bugs found along the way — one of them
mine, in a number already written into Milestone 8.

| target | measured | met |
|---|---|---|
| zero-leakage human GT > 0.765 mAP | **0.7065** (v2.1: 0.7355) | no, −0.059 |
| human-calibrated > 0.800 mAP | **0.7656** (v2.1: 0.7924) | no, −0.034 |
| glare FN recovery ≥ 13/20 | **1/20** (v2.1: 7/20) | no |
| beat SigLIP2 (0.6218) on BDD100K | **0.5952** | no, −0.027 |
| beat logit lookup (0.5871) on BDD100K | **0.5952** | **yes** |

## What v2.2 changes

Five mechanisms, in `src/scene_mining/filter/sceneminer_v2_2.py` (4.68 M params):

1. **Channel dropout.** The VLM margin is removed from both the base score and
   the fusion memory on 50 % of steps (measured 50.3 %); the BEV embedding, tag
   vector and risk scalar are removed on 30 % (measured 28.6 %). A router that
   has bet everything on one channel pays whenever that channel vanishes, so
   dropout *prices* the shortcut rather than forbidding it.
2. **Entropy floor.** `λ_ent · relu(max_k w_k − 0.70)²`, λ_ent = 2.0. Exactly
   zero once the router is distributed, so it never fights a healthy solution.
3. **Cross-modal agreement distillation.** Pairs are weighted by whether the
   SigLIP2 cosine and the VLM margin corroborate each other
   (`0.5·(1 + tanh(cos_z)·tanh(margin_z))`), floored at 0.30. 13 929 / 37 208
   pairs (37.4 %) were down-weighted for teacher disagreement. **No human label
   enters training** — agreement is between two frozen Stage-1 signals, so the
   550 human scenes stay a clean test set.
4. **PartNorm** — running standardisation of the three base parts (see
   Discovery 2).
5. **Explicit modality availability.** `has_3d_rig=False` masks the BEV, tag and
   risk tokens out of cross-attention via `memory_key_padding_mask` **and**
   renormalises the routing distribution over the channels that remain. Feeding
   zeros, as v2.1 did, is not the same as declaring a modality absent.

Identity-preserving start is retained and verified exactly: `max|s − s_base| = 0`
at step 0, uniform routing (0.3333 each), mass renormalised over 2 channels in
monocular mode.

## Discovery 1 — the collapse is real, and fixing it is not what fixes transfer

Routing weight on the VLM margin, on the four canonical human-GT queries:

| attribute | v2.1 | v2.2 |
|---|---|---|
| droplets | 0.987 | 0.560 |
| glare | 0.975 | 0.625 |
| wet_road | 0.992 | 0.607 |
| low_contrast | 0.967 | 0.692 |
| **mean** | **0.980** | **0.535** |

The collapse is eliminated. But the premise behind the milestone — that the
BDD100K failure was *caused* by it — does not survive measurement: on BDD100K's
query phrasings v2.1's router was **already distributed** ([0.301, 0.392, 0.307],
not collapsed), and it still scored 0.4189. The collapse is query-specific,
because `W_route` is a linear map of the query embedding and BDD's phrasings are
unseen. An intermediate build with the routing fixes *only* (dropout + entropy
floor + availability masking, kept as `work_dirs/sceneminer_v22_routingonly`)
moved BDD100K from 0.4177 to **0.4181** — a gain of 0.0004. The routing fix
alone does essentially nothing for transfer.

## Discovery 2 — a convex router over incommensurate parts is not a router

v2.1's blend of two signals scored **below both signals it blended**: 0.4181,
against 0.6218 for the SigLIP2 cosine alone and 0.5871 for the margin alone.
That is impossible for a genuine convex combination, so the parts cannot have
been commensurate. Measured on BDD100K:

| base part | std | mean abs |
|---|---|---|
| SigLIP2 cosine | 0.0193 | 0.0840 |
| VLM margin (tanh of z-scored logit) | 0.5419 | 0.4719 |

The margin part carries **28.1× the spread** of the cosine part, so a nominal
50/50 routing weight is roughly 96 % margin in effect. The same gap is visible
*in training*, with no test data involved — PartNorm's running statistics come
out at std **[0.031, 0.564, 0.166]**. This is the real mechanism behind the
transfer failure, and it is a general lesson for any gated multi-signal ranker:
**a routing weight is only an influence weight if the routed quantities are
standardised.** PartNorm fixes it, and is validated in domain (below).

## Discovery 3 — three bugs, one of them in a reported number

**(a) Oracle channel selection in my own baseline (mine, in Milestone 8).**
`scripts/eval_external_bdd100k.py` chose the `logit_VLM_lookup` margin channel by
maximising correlation **with the test labels** — an oracle no learned model had.
The channel is now fixed a priori per query from its wording alone (`CHANNEL`),
and the old rule is retained as a clearly labelled `*_ORACLE` upper bound.
Honest accounting of the impact: **it barely mattered** — 0.5910 → 0.5871, a
0.0039 difference. I flagged this as a third cause of the v2.1 drop; it was a
real leak and an immaterial one.

**(b) Permuted margin channels on BDD100K (material).** The margin columns were
produced in `QUESTIONS` order, `[wet_road, glare, droplets, low_contrast]`, but
the head's channels were trained in the feature cache's order,
`[droplets, glare, wet_road, low_contrast]`. **Channels 0 and 2 were swapped on
the way into the model**, so every SceneMiner row in Milestone 8 was scored with
the droplets and wet_road margins exchanged. Fixed by an explicit `PERM` applied
before anything consumes the columns. Effect on v2.2: 0.5475 → 0.5952.

**(c) Non-transferable channel selection.** `margin_select` is a learned linear
map fit on 72 nuScenes phrasings; on BDD100K it picked the wrong channel on
**8/12** queries at p_max ≈ 0.26 — a near-uniform average of all four margins.
It is now primed by text-text similarity to the four attribute prompts, on
**mean-centred** embeddings: SigLIP2 is trained for text-*image* alignment, so
its text space is strongly anisotropic and a raw cosine collapses onto one
generic direction (agreeing with a held-out a-priori channel map on 6/12
queries, versus **10/12** after centring).

## Results

### BDD100K — external, human-labelled, 1000 images, 12 queries, chance 0.2623

| ranker | mAP | P@5 | NDCG@10 |
|---|---|---|---|
| zeroshot SigLIP2 | **0.6218** | **0.8500** | **0.8088** |
| **SceneMiner v2.2 (monocular)** | **0.5952** | 0.7500 | 0.7892 |
| logit VLM lookup — *ORACLE, not a baseline* | 0.5910 | 0.6667 | 0.6722 |
| logit VLM lookup (a-priori channel) | 0.5871 | 0.7167 | 0.7054 |
| SceneMiner v2.1 (portable) | 0.4189 | 0.5500 | 0.5524 |
| flat VLM lookup | 0.3817 | 0.5500 | 0.5114 |

v2.2 beats both lookup baselines (including the oracle one) on all three
metrics, and improves on v2.1 by **+0.1763 mAP / +0.2000 P@5 / +0.2368 NDCG@10**.
It remains **below zero-shot SigLIP2 on mAP**, which is the honest headline for
external transfer. Routing on BDD, with the meta channel declared absent:
[siglip 0.489, margin 0.511, meta 0.000].

### Human ground truth — 550 scenes, rater 1, macro-mAP over 4 attributes

| ranker | mAP |
|---|---|
| SceneMiner v2.1 + human calibration | **0.7924** |
| **SceneMiner v2.2 + human calibration** | 0.7656 |
| SceneMiner v2.1 zero-leakage | 0.7355 |
| logit VLM lookup | 0.7325 |
| SceneMiner v2.2 monocular | 0.7167 |
| SceneMiner v2.2 zero-leakage | 0.7065 |
| zeroshot SigLIP2 | 0.5688 |
| metadata prior | 0.5297 |

Calibration refits **routing weights only** on 4/5 of the human scenes and
scores strictly on the held-out 1/5; the residual and all adapters stay frozen.

**Glare false-negative recovery got worse, not better: 1/20 versus v2.1's 7/20
(target ≥ 13/20).** This is the clearest cost of the intervention. v2.1 found
VLM-missed glare scenes *because* it leaned on the glare margin at 0.975 weight
and the margin's near-threshold tail still ranks them above the bulk; capping
that weight at 0.625 discards the signal that was doing the work. Recovering
VLM false negatives and refusing to over-trust the VLM are, on this evidence,
in direct tension.

### 150-query suite — 602 eval-split scenes, no evaluated scene ever trained on

| ranker | mAP | P@5 | NDCG@10 |
|---|---|---|---|
| attribute index — **TAUTOLOGICAL CEILING** | 1.0000 | 0.9920 | 1.0000 |
| SceneMiner v2.1 | **0.5312** | 0.5267 | 0.5368 |
| SceneMiner v2.2 | 0.5178 | 0.5093 | 0.5246 |
| SceneMiner v2.2 monocular | 0.5019 | 0.4720 | 0.5056 |
| zeroshot SigLIP2 | 0.4446 | **0.5387** | 0.5278 |
| logit VLM lookup | 0.3903 | 0.3467 | 0.3734 |

Relevance here is a predicate over the cached VLM attributes, so an index
reading those same attributes scores exactly 1.000 by construction — it is a
ceiling, not a baseline (Milestone 4). Per stratum, v2.1 leads v2.2 in 3 of 4;
v2.2 leads only on `negated` (0.4344 vs 0.4306).

### In-domain validation of the two new mechanisms

Both were ablated at inference on the 150-query suite, so neither rests on a
BDD100K measurement:

| mechanism | Δ mAP overall | strata helped |
|---|---|---|
| PartNorm | **+0.0288** | 4/4 |
| text mean-centring | +0.0018 | 4/4 |

PartNorm is independently justified in domain. Mean-centring is only neutral in
domain — which is expected, since in-domain phrasings were seen in training —
and its large benefit (6/12 → 10/12 channel agreement) was measured on BDD100K.
**Disclosure:** that one design choice was therefore informed by the external
set. It is a selector prior, fixed at train time from label-free text, and it
helps in 4/4 in-domain strata too, but it is not a fully blind choice and should
not be presented as one. **No further tuning against BDD100K was performed**,
to keep it an external test set rather than a validation set.

## Interpretation for the paper

The defensible claim is a **trade-off, not a win**: the routing weight a learned
head places on a VLM teacher is an in-domain asset and an out-of-domain
liability. Capping it costs 0.029 mAP in domain and buys 0.176 mAP in external
transfer. That is a real, measured, mechanistically explained result, and it is
more interesting than a tuned leaderboard number — but it does **not** support a
claim that SceneMiner beats zero-shot SigLIP2 on an external corpus. On
BDD100K's own human labels it does not.

## Artefacts

| path | contents |
|---|---|
| `src/scene_mining/filter/sceneminer_v2_2.py` | v2.2 head, PartNorm, self-test |
| `scripts/train_sceneminer_v22.py` | trainer with dropout / entropy floor / agreement |
| `scripts/eval_v22_human_gt.py` | 550-scene human-GT evaluation |
| `scripts/eval_v22_query_suite.py` | 150-query suite + mechanism ablations |
| `scripts/eval_external_bdd100k.py` | BDD100K, all three bugs fixed |
| `scripts/diag_bdd_blend.py` | the scale-incommensurability diagnostic |
| `results/cvpr/milestone10_sceneminer_v2_2.json` | human-GT results |
| `results/cvpr/milestone10_query_suite_150.json` | suite results + ablations |
| `results/cvpr/milestone10_external_bdd100k_fixed.json` | corrected BDD100K |
| `results/cvpr/milestone10_v22a_routingonly_*.json` | routing-fixes-only build |

5 seeds, 3000 steps each, 0.7 min per seed on one A40.
