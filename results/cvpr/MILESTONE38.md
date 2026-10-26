# Milestone 38 — Thinner annotation boxes

**Problem.** The human-annotation boxes in Figure 3 and the supplementary semantic
strips were drawn at 2.0 pt with a 3.6 pt dark halo, which hid small objects
(pedestrians, cones, two-wheelers).

**Change** (`scripts/make_qualitative_showcase.py`):

* Boxes are now 0.5 pt hairlines in the same yellow, with no halo. This applies both in
  the thumbnails and in the zoom insets.
* The inset frame went from 2.0 pt to 1.0 pt.
* Nothing else changed: the selection, badges and labels are as before.
  * Figure 3: 42 cells, 13 of 20 labelled cells show the class.
  * Supplementary strips: 60 cells, 28 of 48.

**Checks.**

* Pages 7 (main) and 15 (supplementary) were inspected at print size; the objects are
  visible inside their boxes.
* Layout is unchanged: the main text ends on page 8 and the references start on page 9.
* `verify_paper_numbers.py` passes; `check_anonymity.py` passes, including the rebuilt
  code bundle.
