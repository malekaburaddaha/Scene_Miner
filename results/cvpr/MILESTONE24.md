# Milestone 24 — Two-part qualitative showcase with human-annotation callouts

**Script:** `scripts/make_qualitative_showcase.py` (`--only main|supp|fail`)
**Log:** `results/cvpr/milestone24_qualitative_log.json`. It records every retrieved
image id, its frame-match correlation, and its box count.

## Main Figure 2 (`paper/figures/fig2_qualitative.pdf`)

Every row shows SceneMiner's top-2 retrievals from each of three corpora, ranked
within that corpus. Each corpus gets two columns, so no dataset dominates.

* **nuScenes and Waymo:** the 602-scene evaluation split, one image per driving segment.
* **BDD100K:** 1000 external images, scored with portable signals.

**(a) Degradations** (no boxes; these attributes are not box-annotated)

* water droplets smearing the camera lens
* blinding headlight bloom from oncoming traffic
* a low contrast scene on wet reflective asphalt

**(b) Safety-critical road users and objects**

* pedestrians crossing the road at night
* cyclists or motorcyclists on the road
* a construction zone with traffic cones and barriers
* a large truck or bus on a wet road

### Callouts come only from human annotations

**nuScenes**

* The thumbnail is the scene's first keyframe; it matches the full-resolution
  CAM_FRONT image at r ≥ 0.98.
* Boxes are `sample_annotation` boxes of the queried classes at visibility ≥ 40%,
  projected with the devkit.

**Waymo**

* The chunk thumbnail is matched to its source frame by pixel correlation, after
  reproducing the loader's aspect-preserving 0.35× resize and centre padding.
* All 42 nuScenes and Waymo cells matched at r ≥ 0.979. Below r = 0.95, no box would
  have been drawn.
* Boxes are the frame's labelled 3D boxes within 60 m, projected with the frame's
  calibration.
* I verified the projection visually at full resolution: boxes sit on the pedestrians.
* Waymo labels lack truck, bus, cone, barrier, child and emergency classes, and label
  motorcycles as vehicles. Those cells say so instead of drawing boxes.

**BDD100K**

* No box labels are on disk, so no callouts are drawn, and the cell says so.

### Rendering details

* The inset enlarges the smallest annotated box (< 2% of the image). The inset window
  is clamped inside the image.
* A red label marks a retrieval whose frame has **no annotated object** of the
  queried class.
  * Main figure: one nuScenes "pedestrians at night" retrieval, and both Waymo
    "cyclist" retrievals. One of those Waymo frames shows a motorcycle, which Waymo
    labels as a vehicle.
  * These are shown, not filtered out.
* Waymo frames (3:2) are displayed with a 16:9 crop so rows align. Boxes stay in
  pixel coordinates.

## Supplementary

### `supp_semantic_strips.pdf` (full page)

Ten object, rare-road-user and object×condition queries, using the same layout and
callout rules as the main figure. These include:

* "a child or an unusual road user near the road" (nuScenes rare classes)
* "a police car or an emergency vehicle"
* "pedestrians on a rainy night"
* "a truck ahead in heavy rain"

The rare-road-user and emergency rows mostly show red "no annotated …" labels. That
agrees with these categories being at chance for every ranker.

### `supp_failure_strip.pdf`

This strip shows BEV-TSR-style vs Qwen2.5-VL lookup vs SceneMiner, top-3 on the 258
nuScenes scenes with human semantic labels, with ✓/✗ from the ground truth.

Rows are chosen by a **fixed rule** applied to Milestone 23's per-query APs:

| row | rule | AP margin |
|---|---|---|
| "road construction zone with traffic cones and barriers" | largest SceneMiner margin (atomic) | +0.349 |
| "people crossing the street after dark" | largest SceneMiner margin (compositional) | +0.265 |
| "a bus or truck seen through rain droplets on the lens" | largest margin **against** SceneMiner | −0.148 |

The third row is included so the strip is not one-sided. When the lookup has no
column (object-only queries), its score is constant. Its order is then arbitrary and
labelled as such.

## Paper changes

**Main paper**

* New Figure 2 at full text width, with a new caption.
* To fit it within 8 pages, the BDD100K per-query table moved to the supplementary.
  The text now points there.

**Supplementary**

* The old strip figure was replaced by the semantic page and the failure strip.
* The BDD100K per-query table was added.
* Compiles to 15 pages with no undefined references.
