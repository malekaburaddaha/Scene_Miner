# Milestone 1 — BEVFormer 3D gate + corrected diagnostics

## 1. The 3D gate

### Pipeline validation
`bevformer_base.pth` run through `conda_envs/envs/bevformer` (torch 1.10.0,
mmcv 1.4.0, mmdet 2.14.0, mmdet3d 0.17.1 — a complete stack that already
existed; no environment build was needed). Full nuScenes infos regenerated
(34,149 samples, 850 scenes) into a private data root, leaving the shared
`BEVFormer/data` (v1.0-mini, owned by another user) untouched.

**Official nuScenes eval on 6,019 val samples: mAP 0.4164, NDS 0.5173** —
matching the published checkpoint. The inference is sound.

### A per-scene metric that had to be rebuilt
A hand-rolled per-scene matcher gave an **inverted** signal: night scenes
scored 150× *higher* than day (r(night) = +0.96, surviving a control for object
count). Class-agnostic matching let ~300 predictions/sample trivially cover a
small GT set. Rebuilt on the official `accumulate`/`calc_ap`, per scene:

| signal | night | day | r(night) | partial r \| n_gt |
|---|---|---|---|---|
| per-scene mAP (official) | 0.4171 | 0.4553 | **−0.115** | −0.075 |
| per-scene mAP, rain vs dry | 0.4462 | 0.4526 | −0.025 | −0.021 |

Correct direction, weak confound. **But the effect is small: night costs
−0.038 mAP and rain −0.006.** On clean nuScenes val, these conditions barely
move BEVFormer at all.

### Training-set contamination, quantified
All 278 of our held-out nuScenes scenes are in the official **train** split,
which `bevformer_base` was trained on.

| set | mean per-scene mAP | sd |
|---|---|---|
| our 278 (train split, memorised) | **0.7864** | 0.068 |
| 150 official val (never seen) | **0.4514** | 0.100 |
| **inflation** | **+0.3349** (+74 % relative) | variance compressed |

This is a concrete number for Reviewer ETJq's objection. Memorisation also
*compresses* the difficulty spread, shrinking the signal the gate depends on.

### Gate result (on the 278 contaminated scenes, top-50)
Hardness = negated per-scene mAP; higher = harder scenes retrieved.

| ranker | droplets | wet_road | low_contrast | MEAN |
|---|---|---|---|---|
| **flat VLM lookup** | −0.7602 | −0.7607 | −0.7679 | **−0.7629** |
| logit VLM lookup | −0.7671 | −0.7644 | −0.7679 | −0.7665 |
| metadata-matched | −0.7676 | −0.7676 | −0.7733 | −0.7695 |
| zero-shot SigLIP2 | −0.7691 | −0.7626 | −0.7909 | −0.7742 |
| **SceneMiner** | −0.7634 | −0.8007 | — | **−0.7821** |
| random | −0.7834 | −0.7901 | −0.7894 | −0.7877 |

**SceneMiner is second-worst, barely above random.** Within the positive bucket
— where the lookup is tied and blind — SceneMiner's correlation with hardness is
**negative** (droplets ρ = −0.206, p = 0.086; wet_road ρ = −0.237, p = 0.049):
it ranks *easier* scenes higher. The decile curve is flat (D1 −0.797 →
D10 −0.808). The glare bucket (9 scenes) was too small to test.

**Caveat, stated plainly:** all rankers sit within 0.02 mAP of each other on a
signal with sd 0.068 (~0.3σ), and contamination compressed that spread. This is
weak evidence, not a decisive refutation. The honest reading is that the gate is
**inconclusive on this corpus** — and that the conditions themselves cost
BEVFormer little even on clean val (−0.038 for night).

## 2. The three diagnostic bugs

### Fix 1 — BLIP-2 effective rank
Concatenating 32 query tokens into (N, 32×768) gave erank 913, which cannot be
compared to "41.2 of 256". Measured properly:

| construction | erank |
|---|---|
| stacked tokens, 768-d | 443.11 |
| **PCA → 256-d (comparable)** | **180.72** of 256 |
| concat 32×768 (the bug) | 913.38 |

erank ratio vs label null = 22.1; **0 of 32 query tokens carry <1 % routing
mass**. Measured correctly, BLIP-2 still shows **no degeneracy** — §4.3 does not
reproduce.

### Fix 2 — label-entropy null
A multi-hot matrix over *k* atomic binary attributes has rank ≤ *k*, so
erank(L) ≤ k by arithmetic.

| construction | columns | erank(L) | matrix rank |
|---|---|---|---|
| atomic attributes | 13 | 8.96 | ≤13 |
| query-predicate (22 queries) | 22 | 8.98 | **14** |

§4.3's 47.6 needs ≥48 *linearly independent* supervision columns. Our current
suite has rank 14; recompute once the 390-query suite exists.

### Fix 3 — the unbounded-score hypothesis is disconfirmed
The clamp was never the cause. τ drifts **upward** (softening), never toward
saturation:

| run | steps | clamp | learned τ | 1/τ |
|---|---|---|---|---|
| Atrue_s0 | 3,000 | on | 1.0700 | 0.93 |
| AtrueUT_s0 | 3,000 | **off** | **1.0700** (identical) | 0.93 |
| AtrueUT_long | **12,000** | off | **1.2487** | 0.80 |

Removing the 1e-2 floor changed nothing because the floor was never binding, and
a 4× longer budget moved τ *further* from saturation. The predecessor's
"τ drifted below one, score span 213.7" does not reproduce in any configuration.

## 3. Correction to last night's framing
Row A_true (risk token genuinely dead: 1 distinct row, std 0.0000) scores
mAP 0.311 ± 0.036 vs Row A's 0.307 ± 0.048. I previously called this a
refutation. **It is not — it is the paper's thesis demonstrated.** A modality
can be 100 % dead while the task metric is undisturbed, which is exactly the
claim that retrieval scores do not reveal these defects.
