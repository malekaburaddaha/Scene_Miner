# Milestone 6 — SceneMiner v2.1, query-conditioned modality routing

## What changed
v2 blended the SigLIP2 cosine and the VLM margins with a **single global scalar**
(~0.60) applied to every query. That is wrong whenever the two disagree about
which query they are good at, and they do:

| attribute | zero-shot SigLIP2 | logit VLM lookup |
|---|---|---|
| droplets | **0.913** | 0.824 |
| low_contrast | 0.105 | **0.552** |

Forcing 60 % onto SigLIP2 dragged the v2 base to 0.3585 on `low_contrast`.

v2.1 replaces the scalar with a routing distribution computed from the query,
`w(q) = softmax(W_route q̂ / τ_r)` over three base signals — SigLIP2 cosine,
standardised VLM margins, and a deterministic tag/metadata prior (20 z-scored
tag logits + 6 metadata one-hots) — plus a **query-conditioned** residual gate.
`W_route`, both channel selectors and the gate head are zero-initialised, so at
step 0 routing is exactly uniform (0.3333 each) and the residual is inert.
Verified. 4.68 M parameters.

## Result (Mode B, 550 human-labelled scenes)

| ranker | droplets | glare | wet_road | low_contrast | **MEAN** |
|---|---|---|---|---|---|
| **v2.1 + human calibration (5-fold)** | 0.8545 | **0.7035** | 0.9616 | **0.6499** | **0.7924** |
| SceneMiner v2 (Milestone 2) | **0.9143** | 0.6416 | 0.9663 | 0.5126 | 0.7587 |
| v2.1 zero-leakage | 0.8991 | 0.5754 | 0.9639 | 0.5035 | 0.7355 |
| logit VLM lookup | 0.8238 | 0.5887 | **0.9653** | 0.5521 | 0.7325 |
| zero-shot SigLIP2 | 0.9129 | 0.4144 | 0.8433 | 0.1045 | 0.5688 |
| metadata prior alone | 0.1960 | 0.1318 | 0.2550 | 0.2942 | 0.2193 |
| chance | 0.1436 | 0.0745 | 0.1545 | 0.1200 | 0.1232 |

## The two goals, answered honestly

**"Wins 4 of 4 attributes" — NO.** The calibrated model wins `glare` (+0.115 over
the best baseline) and `low_contrast` (+0.098), ties `wet_road` (−0.004), and
**loses `droplets`** to plain zero-shot SigLIP2 (0.8545 vs 0.9129).

**"Improves glare FN recovery" — NO, it got worse.** On the 20 glare scenes the
VLM misses:

| ranker | recovered in top-20 % | median rank |
|---|---|---|
| **zero-shot SigLIP2** | **14/20** | 14 % |
| v2.1 | 7/20 | 31 % |
| logit VLM lookup | 7/20 | 24 % |
| *(v2, for reference)* | *10/20* | *20 %* |

Plain SigLIP2 remains the best recoverer of single-VLM errors across every
version we have built.

## The finding that matters most: routing collapsed

The learned routing is **not** query-specific. Across all five seeds and all
four attributes it puts 0.967–0.992 of its mass on the VLM margin:

| attribute | siglip | margin | meta |
|---|---|---|---|
| droplets | 0.005 | **0.987** | 0.008 |
| glare | 0.010 | **0.975** | 0.015 |
| wet_road | 0.003 | **0.992** | 0.005 |
| low_contrast | 0.016 | **0.967** | 0.018 |

This is circularity resurfacing in a very clean form: **trained on VLM-derived
labels, the optimal routing is to copy the VLM margin.** The architecture gave
the model the freedom to trust SigLIP2 on `droplets`; the supervision removed
any reason to. That is why v2.1 zero-leakage (0.7355) is *worse* than v2
(0.7587) — it threw away SigLIP2's droplets advantage.

## What the human labels buy
Refitting **the three routing weights only** on 4/5 of the human scenes and
scoring on the held-out 1/5 (residual and adapters frozen) moves:

- `low_contrast` 0.5035 → **0.6499** (+0.146)
- `glare` 0.5754 → **0.7035** (+0.128)

**A 440-scene human calibration set is worth more than the architecture change.**
That is a usable paper claim, and it points at human-labelled calibration rather
than more fusion capacity as the next investment.

Caveat: the calibrated numbers come from a 3-weight grid search per fold with
110 held-out scenes, so fold variance is non-trivial (see `mAP_std` in the JSON).
