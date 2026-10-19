# Milestone 8 — external dataset + uncontaminated 3D study

## 1. BDD100K: a third dataset, with native human labels, no login

`datasets/argoverse` turned out to be 62 GB of three unextracted tars **with no
weather or lighting annotations** — which is precisely why §7 names Argoverse as
the *metadata-free* corpus. It cannot serve as a human-labelled adverse-weather
benchmark.

Instead: **BDD100K** via the ungated `dgural/bdd100k` Hub mirror. It carries
per-image human annotations for `weather` (clear / overcast / rainy / snowy /
foggy / partly cloudy) and `timeofday` (daytime / night / dawn-dusk) — a label
channel that shares nothing with our VLM pipeline.

Downloaded a stratified **1,000 images** oversampling adverse conditions:

| field | distribution |
|---|---|
| weather | clear 628, overcast 123, **rainy 94**, **snowy 82**, partly cloudy 73 |
| timeofday | **night 493**, daytime 431, dawn/dusk 76 |

Extracted SigLIP2 embeddings and all four Qwen2.5-VL logit margins, then
evaluated 12 queries whose relevance comes from BDD's own human labels.

### Result (chance 0.2623)
| ranker | mAP | P@5 | NDCG@10 |
|---|---|---|---|
| **zero-shot SigLIP2** | **0.6218** | **0.8500** | **0.8088** |
| logit VLM lookup | 0.5910 | 0.6667 | 0.6722 |
| SceneMiner v2.1 (portable routing) | 0.4177 | 0.5500 | 0.5453 |
| flat VLM lookup | 0.3899 | 0.5500 | 0.5326 |

**The learned component does not transfer; the zero-shot component does.** The
routing head was trained on nuScenes/Waymo where it learned to put 0.97+ of its
mass on the VLM margins (Milestone 6). On BDD100K the margins are weaker than
SigLIP2 for these conditions, so that learned preference actively hurts —
0.4177 against SigLIP2's 0.6218.

### A structural limit, reported rather than worked around
The full SceneMiner head consumes four Stage-1 signals — BEVFormer embedding,
tag logits, physical risk, Qwen visual token — that require a **calibrated
6-camera rig with 3D boxes**. BDD100K is monocular dashcam video. Only the
portable part of the model (SigLIP2 cosine + VLM margins + learned routing)
can cross to a new corpus at all. **The architecture is not portable to the kind
of corpus the Area Chair asked us to add**, and that is worth stating plainly in
the paper rather than hiding behind a partial evaluation.

## 2. Uncontaminated 150-scene nuScenes val 3D study

The cleanest version of the gate: `bevformer_base` never trained on these
scenes (per-scene mAP 0.4514 vs 0.7864 on the memorised train scenes). Scenes
ranked by the portable signals, top-30 evaluated against official per-scene mAP.

**Lower top-k mAP = found harder scenes.** Corpus mean 0.4514.

| ranker | top-30 mAP | tight mAP (0.5 m) | low-vis frac | Δ vs corpus |
|---|---|---|---|---|
| logit VLM lookup | 0.4449 | 0.1744 | 0.3425 | **−0.0065** |
| metadata-matched | 0.4477 | 0.1553 | 0.3216 | −0.0038 |
| random | 0.4524 | 0.1556 | 0.4052 | +0.0010 |
| zero-shot SigLIP2 | 0.4546 | 0.1724 | 0.3247 | +0.0031 |

**No ranker finds measurably harder scenes.** Every delta is ≤ 0.0065 against a
per-scene standard deviation of ~0.10 — under 0.07 σ. The best performer,
the logit lookup, beats random by 0.0065; zero-shot SigLIP2 is actually *worse*
than random.

Two further checks point the same way:
- On the **tight 0.5 m** threshold (where precise localisation matters most),
  the retrieved sets score *higher* (0.1744, 0.1724) than random (0.1556) —
  i.e. easier, not harder.
- On the **human-annotated low-visibility fraction**, random draws the highest
  value (0.4052); every ranker selects scenes with *fewer* low-visibility
  annotations.

### What this means for §6.8
The submitted paper reports retrieved scenes at mAP 0.287 against 0.361 for
metadata-matched and 0.412 random — a −0.074 gap. **On uncontaminated data I
cannot reproduce any gap of that size in any direction.** The contaminated
278-scene study (Milestone 1) was also inconclusive, with all rankers inside
0.02 mAP. Combined with the clean-val finding that night costs BEVFormer only
−0.038 mAP and rain −0.006, the evidence says these conditions **do not
meaningfully degrade this detector on nuScenes**, and the downstream-utility
claim as written is not supported.
