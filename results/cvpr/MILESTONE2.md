# Milestone 2 — SceneMiner v2

## Diagnosis driving the design
Against the 600 human labels, v1 lost to signals it could not see:

| ranker | mean mAP (human GT) |
|---|---|
| logit VLM lookup | 0.733 |
| zero-shot SigLIP2 | 0.569 |
| SceneMiner v1 (row E) | 0.420 |

Three causes, each addressed:
1. **Missing signals.** `features_overnight.npz` carried no SigLIP2 embedding and
   no continuous VLM margin. Both are now cached (`features_v2.npz`:
   `siglip_img` 768-d, `vlm_margin` 4-d + standardised).
2. **Noisy binary supervision.** v1 imitated hard 0/1 VLM flags whose precision
   against human labels is 0.50–0.68. v2 weights every positive and negative by
   `tanh(|margin|)`, so borderline VLM calls contribute less.
3. **An unsupervised attribute.** v1 never trained on `low_contrast` (0.089 mAP).
   v2 supervises all four, with 4 phrasings × 5 macro contexts → 72 queries.

## Identity-preserving growth (Sec. 3.4)
```
s(q,x) = s_base(q,x) + tanh(gate) · s_res(q,x),    gate initialised to 0
```
`s_base` blends the SigLIP2 image–text cosine with a query-selected VLM margin;
`s_res` is a 6-token fusion head (BEV embedding, tags, risk, Qwen visual,
SigLIP2 image, VLM margins). **Verified: score == base exactly at init**
(max |s − base| = 0.00e+00), so every departure is something training paid for.
4.66 M parameters; 3,000 steps in 0.6 min/seed.

Learned values: gate → ±0.12 (seed 4 went **negative**, −0.119), blend → 0.60.

## Result (Mode B, 550 human-labelled scenes, 5 seeds)

| ranker | droplets | glare | wet_road | low_contrast | **MEAN** |
|---|---|---|---|---|---|
| **SceneMiner v2** | **0.9143** | **0.6416** | **0.9663** | 0.5126 | **0.7587** |
| logit VLM lookup | 0.8238 | 0.5887 | 0.9653 | **0.5521** | 0.7325 |
| v2 base only (zero-shot prior) | 0.7498 | 0.5994 | 0.8149 | 0.3585 | 0.6307 |
| metadata-matched | 0.7080 | 0.3104 | 0.6835 | 0.7440 | 0.6115 |
| zero-shot SigLIP2 | 0.9129 | 0.4144 | 0.8433 | 0.1045 | 0.5688 |
| flat VLM lookup | 0.6427 | 0.3883 | 0.6896 | 0.4369 | 0.5394 |
| SceneMiner v1 row E | 0.9077 | 0.2031 | 0.4813 | 0.0888 | 0.4202 |
| SceneMiner v1 D_final | 0.8463 | 0.2617 | 0.2249 | 0.1563 | 0.3723 |
| chance | 0.1436 | 0.0745 | 0.1545 | 0.1200 | 0.1232 |

**Leakage check: 0 of 600 human-labelled scenes are in the training split.**

### What survives seed noise
| attribute | v2 | sd | logit lookup | delta | > 2 sd |
|---|---|---|---|---|---|
| droplets | 0.9143 | 0.0188 | 0.8238 | **+0.0905** | yes |
| glare | 0.6416 | 0.0090 | 0.5887 | **+0.0530** | yes |
| wet_road | 0.9663 | 0.0070 | 0.9653 | +0.0010 | no (tie) |
| low_contrast | 0.5126 | 0.0164 | 0.5521 | **−0.0395** | yes (loses) |

## Honest reading
- **v2 is a real improvement over v1: +0.339 mean mAP** (0.759 vs 0.420), and it
  is the first configuration to beat every baseline on average.
- **The learned residual genuinely contributes +0.128** over the zero-shot prior
  it was initialised from (0.759 vs 0.631) — this is not inheritance.
- But the margin over the best baseline is **+0.026 overall**: two significant
  wins, one tie, one significant loss. It is not a across-the-board victory.
- **The fusion-recovery claim still does not hold.** On glare (20 VLM false
  negatives) v2 recovers 10/20 in the top-20 %, its own base recovers 11/20, and
  plain zero-shot SigLIP2 recovered 14/20. Multimodal fusion is still not the
  thing that rescues single-VLM errors.
- Scope: four canonical queries, one rater. The 390-query suite and a second
  rater are needed before this is a paper claim.
