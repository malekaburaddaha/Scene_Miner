# Milestone 16 — from retrieval to 3D perception breakage on uncontaminated val150

## Verdict up front

**Retrieval closes the loop with the detector, but SceneMiner is not the best
way to close it.** On the 150 nuScenes val scenes that BEVFormer never trained
on, SceneMiner v2.3's top-25 scenes for degradation queries are significantly
harder for the detector than random: front-sector mAP −0.060 (−0.45σ,
p = 0.001), VRU recall −0.193 (−0.80σ, p < 0.001), far recall −0.043
(p = 0.048). They are also significantly harder than zero-shot SigLIP2's
retrievals (mAP −0.028, p = 0.004; VRU recall −0.095, p < 0.001). They are
**statistically indistinguishable** from the 4-column VLM-margin lookup and from
simple night/rain metadata, and the metadata baseline is the only ranker whose
ghost-FP increase is significant.

On semantic queries, no ranker retrieves harder-than-random scenes except plain
v2.3. That is not a virtue: it is the margin leakage diagnosed in Milestone 15.
Its "pedestrian" retrievals are degraded scenes, not pedestrian scenes.

## Setup

* **Scenes.** The 150 official nuScenes val scenes. BEVFormer never saw them
  (per-scene mAP 0.4514 here vs 0.7864 on its train scenes).
* **Signals.** Only portable ones exist for val150: SigLIP2 on the CAM_FRONT
  keyframe (computed in Milestone 15) and the CAM_FRONT Qwen margins (Milestone
  11). v2.3 therefore runs in portable mode (meta channel absent, no residual).
* **Queries.** The 62 sensor-degradation queries of the frozen 150-query suite
  (not chosen for this study) and the 20 atomic Stratum-5 semantic queries.
* **Rankers.** v2.3, v2.3A (post-hoc abstention), zero-shot SigLIP2, logit VLM
  lookup (the query attribute's column), metadata matched (human scene
  description: night for glare/low contrast, rain for droplets/wet road), and
  random. A ranker with nothing to say about a query is random for it.
* **Metrics**, all inside the CAM_FRONT sector: mAP (official
  `accumulate`/`calc_ap`), ghost FP rate at score > 0.3, recall beyond 30 m,
  mATE (official `calc_tp`), and — new here — **VRU recall** and **VRU
  translation error** for pedestrians, bicycles and motorcycles at score > 0.3
  (`scripts/val150_ped_vru_metrics.py`; 126/150 scenes have front-sector VRU GT).
* **Null.** Similar queries retrieve overlapping scenes, so comparing each
  query's top-25 with independent random subsets would be anti-conservative.
  Instead the per-scene metric vector is **permuted across scenes** (2000
  times) with every ranker's retrieved sets held fixed. Ranker-vs-ranker tests
  apply the same permutation to both rankers at once.

## Top-25 vs the val150 mean — 62 degradation queries

Δ from the corpus mean (σ = scene-level SD), one-sided p in the worse direction.

| ranker | front mAP | ghost FP rate | recall > 30 m | mATE (m) | VRU recall | VRU ATE (m) |
|---|---|---|---|---|---|---|
| corpus mean | 0.4547 | 0.2979 | 0.5874 | 0.7763 | 0.5937 | 0.7998 |
| **v2.3** | **−0.0599** (p 0.001) | +0.0289 (p 0.090) | −0.0434 (p 0.048) | +0.0340 (p 0.102) | **−0.1931** (p <0.001) | +0.0578 (p 0.122) |
| v2.3A | −0.0574 (p 0.001) | +0.0300 (p 0.074) | −0.0421 (p 0.049) | +0.0333 (p 0.099) | −0.1857 (p <0.001) | +0.0560 (p 0.112) |
| logit VLM lookup | −0.0537 (p 0.002) | +0.0257 (p 0.114) | −0.0410 (p 0.058) | +0.0276 (p 0.140) | −0.1910 (p <0.001) | +0.0495 (p 0.158) |
| **night/rain metadata** | **−0.0658** (p <0.001) | **+0.0521** (p 0.002) | **−0.0551** (p 0.008) | **+0.0541** (p 0.008) | −0.1668 (p <0.001) | +0.0546 (p 0.101) |
| zero-shot SigLIP2 | −0.0317 (p 0.016) | +0.0188 (p 0.121) | −0.0268 (p 0.092) | +0.0234 (p 0.107) | −0.0979 (p <0.001) | +0.0347 (p 0.179) |
| random | +0.0003 | +0.0005 | +0.0006 | −0.0001 | −0.0004 | −0.0003 |

### Ranker vs ranker (two-sided, same permutation null)

| comparison | front mAP | ghost FP | VRU recall |
|---|---|---|---|
| v2.3 − SigLIP2 | **−0.0282 (p 0.004)** | +0.0101 (p 0.39) | **−0.0953 (p < 0.001)** |
| v2.3 − VLM lookup | −0.0062 (p 0.17) | +0.0032 (p 0.49) | −0.0021 (p 0.80) |
| v2.3 − metadata | +0.0060 (p 0.59) | −0.0232 (p 0.054) | −0.0263 (p 0.22) |
| metadata − SigLIP2 | −0.0341 (p 0.001) | +0.0332 (p 0.002) | −0.0690 (p 0.001) |

### Quartiles (`paper/figures/fig_val150_quartiles.pdf`)

| ranker | Q1 mAP | Q4 mAP | Q1 ghost FP | Q1 VRU recall |
|---|---|---|---|---|
| v2.3 | **0.3949** | 0.4750 | 0.3334 | **0.4415** |
| VLM lookup | 0.3983 | 0.4660 | 0.3240 | 0.4515 |
| metadata | 0.4097 | 0.4681 | **0.3353** | 0.4970 |
| SigLIP2 | 0.4199 | 0.4620 | 0.3201 | 0.5157 |
| random | 0.4553 | 0.4553 | 0.2981 | — |

The damage is concentrated in the top-scored quartile for every informative
ranker. Q2–Q4 sit at or slightly above the corpus mean, so the relationship is a
**threshold, not a gradient**: retrieval finds a hard head, and below it the
ranking carries no detector-relevant information. v2.3 has the lowest Q1 mAP and
the lowest Q1 VRU recall of all rankers, but its margin over the lookup is
within noise.

## Semantic queries (20 atomic)

| ranker | front mAP Δ | VRU recall Δ |
|---|---|---|
| v2.3 | −0.0251 (p 0.020) | −0.0868 (p 0.001) |
| v2.3A | +0.0004 (p 0.52) | −0.0167 (p 0.20) |
| SigLIP2 | +0.0033 (p 0.65) | −0.0066 (p 0.38) |
| lookups / metadata / random | ≈ 0 | ≈ 0 |

Scenes that contain pedestrians, cones or trucks are **not** harder for this
detector than other scenes. Plain v2.3's apparent effect comes from its margin
leaking into semantic queries (Milestone 15): it returns degraded scenes
whatever the query asks for. The abstaining variant, which actually retrieves
the semantic content, shows no effect.

## Interpretation

1. The loop between language-guided retrieval and 3D breakage **is real on
   uncontaminated data**: the top quartile of a degradation query is 0.06 mAP and
   0.19 VRU recall worse than the rest.
2. The signal driving it is **the degradation itself**, which human night/rain
   metadata and the VLM margin capture as well as SceneMiner does on these 150
   scenes. SceneMiner adds value over raw SigLIP2, not over the VLM column it
   routes to. This matches Milestone 14: in vocabulary, v2.3's base is mostly
   the margin.
3. VRU recall is the largest effect (−0.80σ) and the most safety-relevant one.
   Part of it may be compositional — fewer and smaller pedestrians are visible at
   night — and this design cannot separate that from detector failure.

## Artefacts

| path | contents |
|---|---|
| `scripts/val150_ped_vru_metrics.py` | per-scene front-sector VRU recall / ATE |
| `scripts/val150_retrieval_to_3d.py` | top-k, quartiles, permutation and pairwise tests |
| `scripts/make_figures.py` | adds `fig_val150_quartiles` |
| `results/cvpr/val150_per_scene_front.json` | per-scene front-sector metrics, 150 scenes |
| `results/cvpr/milestone16_val150_retrieval_to_3d.json` | all results |
| `paper/figures/fig_val150_quartiles.pdf` | quartile figure |
