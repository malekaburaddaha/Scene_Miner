# Milestone 9 — reconciling §4.2, and final repository state

## The CLIP logit-scale hypothesis: partially confirmed

Your proposal was that the predecessor's saturation came not from dividing by a
learned τ but from CLIP's **multiplicative** parameterisation —
`logit_scale = Parameter(log(1/0.07))`, `score = exp(logit_scale) · raw_dot`,
applied to unnormalised projections. Implemented as `--clip-logit-scale` and
trained for 3,000 steps × 3 seeds alongside `--force-layernorm1d`:

| configuration | score span | listwise effective support (of 1,508) |
|---|---|---|
| A_true, learned τ (divide) | 7.35 | 655.26 |
| **A_true + CLIP logit_scale** | **15.08** | **299.47** |
| paper predecessor (§4.2) | **213.7** | **1.06** |

**Direction confirmed, magnitude not.** The multiplicative form roughly doubles
the score span and halves the effective support relative to the divide-by-τ
form. It is the better account of the two. But it does not reach 213.7 / 1.06,
and the learned scale did not run away — it settled at
exp(2.563) = **12.98**, slightly *below* its 14.29 initialisation, consistently
across all three seeds.

One caveat worth recording: measured on the **16-scene training batch** rather
than the full corpus, effective support does collapse — 2.0/16 and 1.9/16 on
seeds 1 and 2 at step 3,000. The saturation signature is real inside the
listwise objective; it simply does not produce a 213-unit span over the whole
1,508-scene bank at this budget.

**Conclusion for the paper.** Of the three mechanisms tested for §4.2's claim —
the 1e-2 clamp, a longer budget, and the CLIP multiplicative scale — only the
third moves the numbers, and only part of the way. The honest statement is that
score saturation is reproducible *in kind* but not *in degree*, and the 213.7
figure should either be re-derived from the original run or dropped.

## Final repository state
`/nfs/turbo/umd-srawa/scene_mining/Scene_Miner`, remote
`git@github.com:malekaburaddaha/Scene_Miner.git`, all commits under
`malekaburaddaha <am.aburaddaha@gmail.com>` with no AI attribution anywhere.
