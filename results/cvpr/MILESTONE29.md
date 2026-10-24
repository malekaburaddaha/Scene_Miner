# Milestone 29 — De-duplicated showcase, Waymo/BDD100K callouts, two-tier Figure 1

**Scripts**

* `scripts/make_qualitative_showcase.py`
* `scripts/make_fig1_architecture.py`

**Outputs**

* `paper/figures/fig2_qualitative.pdf`
* `paper/figures/supp_semantic_strips.pdf`
* `paper/figures/supp_failure_strip.pdf`
* `paper/figures/fig1_architecture.pdf`
* `paper/figures/supp_stage1.pdf`
* `paper/figures/supp_stage2.pdf`
* the log `results/cvpr/milestone24_qualitative_log.json`

## 1. No repeated image or segment

* **One lock per figure.** `Ranker.top` takes a set of used driving segments that is shared
  by every row of a figure:
  * a nuScenes scene;
  * a Waymo segment, so all its chunks count as one;
  * a BDD100K video, the image-name prefix.
* **How a row fills.** It takes each corpus's best-ranked segment not yet shown.
* **Logging.** The rank of every shown image within its corpus is logged.
* **Hard check.** `render` asserts that every image id in the figure is distinct.

| figure | cells | distinct images | deepest rank shown |
|---|---|---|---|
| Figure 3 (main) | 42 | 42 | 18 |
| supplementary semantic strips | 60 | 60 | 8 |

## 2. Callouts from human annotations, no disclaimer stamps

**Waymo**

* Trucks, buses and motorcycles are labelled `TYPE_VEHICLE`, so they are selected by
  labelled 3D size.
* Two-wheelers: `TYPE_CYCLIST`, or a vehicle with length < 2.8 m and width < 1.2 m.
* Heavy vehicles: length ≥ 6.0 m or height ≥ 2.8 m, within 50 m.
* **Deviation from the requested rule** (length ≥ 6.0 m **or height ≥ 2.3 m**). The 2.3 m
  height bound selected a 5.3 × 2.5 × 2.45 m van 58 m away. I inspected it at full
  resolution: it is not a truck or a bus. So I raised the height bound to 2.8 m (box
  trucks, buses) and kept boxes within 50 m.
* The pedestrian projection was verified at full resolution in Milestone 24.

**BDD100K**

* The mirror's `samples.json` carries BDD100K's human 2D boxes for all 10,000 images.
* `build_bdd_holdout.py` writes them for the 1000 evaluation images to
  `datasets/external_adverse/bdd100k/boxes.json`.
* They are now drawn: pedestrian / rider / bicycle / motorcycle / truck / bus.
* One cell shows 15 motorcycle boxes. Checked against the labels, that image does label 15
  parked motorcycles.

**Stamps**

* Grey stamps ("no box labels on disk", "class not in Waymo labels") are removed.
* A red badge now appears only where the dataset labels the queried class and the frame
  has none of it.
* What each dataset does not label (cones and barriers in Waymo and BDD100K) is stated once
  in the caption.

**Hit rate.** These are cells whose dataset labels the class and that show at least one
annotated object. The rows are the top-ranked unique segments; I did not choose retrievals
by whether they contain the object, so the misses are shown.

| | nuScenes | Waymo | BDD100K | total |
|---|---|---|---|---|
| Figure 3 | 7/8 | 2/6 | 4/6 | **13/20** |
| supplementary strips | 14/20 | 7/14 | 7/14 | **28/48** |

## 3. Figure 1: two-tier diagram with current, measured content

**Layout**

* A colour legend bar.
* A full-width Stage-1 tier with two rows:
  * the multi-camera rig path: SigLIP2-base patches, BEVFormer 50×50, motion head, BLIP-2
    Q-Former (32 queries), Qwen2.5-VL-3B with LoRA r = 16 on q/v, and the e / t / r heads;
  * the monocular path for any single camera: SigLIP2 ViT-B/16, zero-shot Qwen2.5-VL
    margins and token, recorded metadata.
  * Both rows feed the cached feature bank.
* A full-width Stage-2 tier: six modality adapters (with r standardised, not LayerNormed),
  FiLM, the 2-layer cross-attention decoder, the scorer, ResNorm, PartNorm, the router, the
  factored concept gate, the score equation, and the query wiring.

**Where the content comes from**

* **Stage-1 configuration:** `configs/experiment/nuscenes_scenemining_v8_unified.py`.
* **Stage-2 parameter counts:** the v2.6 state dict, 4,678,736 trainable.
* **Latency:** `milestone27_latency.json`, 3.5 ms per query.

**Removed**

* Nothing is copied from the v1 reference diagram (CLIP, 24 M LoRA, 0.864 mAP).
* Training-run statistics I could not verify from logs are left out (s/epoch, step count;
  the config says 2 epochs, the reference said 3).

**Supplementary S1** gains two detailed diagrams (`supp_stage1.pdf`, `supp_stage2.pdf`) with
measured parameter counts:

* SigLIP2 vision tower: 92.9 M.
* BLIP-2 Q-Former: 105.1 M.
* Qwen2.5-VL-3B: 4.07 B total, hidden size 2048 (meta-device instantiation from the
  configs).
* Every Stage-2 module.

## Correction to the Milestone 28 commit message

It says SceneMiner is above every other baseline on the holdout with p ≤ 0.013. That is not
true for SigLIP ViT-B/16 (v1): +0.012, p = 0.42. `MILESTONE28.md` lists the significant
comparisons correctly (CLIP / ReferGPT-style, CARIM-style, RefAV-style, lookup, naive
router). The commit history was not rewritten.
