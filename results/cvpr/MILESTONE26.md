# Milestone 26 — Final compile, page-by-page inspection, documentation

## Compile

* **`paper/main_cvpr.pdf`**
  * Official CVPR 2026 kit (review mode).
  * 10 pages: **8 pages of main text, references beginning at the top of page 9**.
  * No undefined references or citations. One 3 pt overfull line remains; it is not visible.
* **`paper/supp_cvpr.pdf`**
  * 16 pages.
  * No undefined references.
  * Two main-paper cross-references were updated after the M25 table renumbering
    (S3: human-graded evidence is in Tables 2, 4, 6, 7, 8; the rare-category remark
    now points to Sec. 9).

## Inspection

Every page of both PDFs was rasterised and read.

**Main paper**

* Page 1: the abstract and the three contributions.
* Page 2: Figure 1 and related work, including the new driving-retrieval paragraph.
* Pages 3–4: Sections 4–7 and the audit figure.
* Pages 5–6:
  * the design table and Results;
  * Table 2, the master comparison, with † and ‡ marks;
  * the ablation, BDD100K, suite and regimes tables. The regimes caption explains
    0.789 (ensemble) vs 0.785 (per-seed mean).
* Page 7: Figure 3, the two-part showcase. Box callouts and red "no annotated"
  labels are legible.
* Page 8: Figure 4, the premise and loop tables, and the Discussion, ending at the
  bottom of the right column.
* Pages 9–10: the 24 references.

**Supplementary**

* Hyperparameters and the gate grid.
* The new "Re-implemented baselines" section.
* Annotation protocol and the significance tests.
* Per-query tables, including the BDD100K per-query table and both rater tables
  moved from the main paper.
* The semantic strip page and the failure strip.
* The training queries and the reproduction commands.

## Documentation

**README.md**

* The BDD100K table is corrected, with a note on the bug.
* New master comparison table.
* Reproduction commands for Milestones 23–25.

**MASTER_REPORT.md**

* Correction notes added to the historical Milestone 10/14 BDD100K passages, which
  are kept as written.
* The Milestone 22 table is corrected.
* New sections 23–25.
* Updated final claims: SceneMiner is level with SigLIP2 on BDD100K, not above it,
  and does not beat every published paradigm everywhere.

## Repository checks

* No AI attribution in any tracked file or commit message.
* The user's email address appears only in:
  * commit metadata;
  * two earlier milestone notes that record commit authorship (MILESTONE9, MILESTONE13).
  * It is not sent to any service.
* The one commit message matching "claude/anthropic" (a08fad1) describes the RefProg
  LLM-backend selector in code; it is not attribution.
* No file over 5 MB.
* No checkpoints, datasets or caption dumps are tracked.
* Every commit is by `malekaburaddaha`.
* The tree is clean after the push.
