# Milestone 31 — Title, generated tables, number verification, final compile

## 1. Title and readability

**Title:** *SceneMiner: Scale-Aware Multi-Modal Fusion and Diagnostic Auditing for
Language-Guided Driving Scene Retrieval* (main and supplementary).

**Wording:**

* "calibrated fusion" became "scale-aware fusion".
* "Calibrated" now refers only to the 5-fold human calibration (0.829).
* Remaining lab-diary references ("Milestone 27", "v2.5") were removed from the
  supplementary.

**Tables are now generated from the result files at 3 decimals.**

* Generators:
  * `scripts/make_main_tables.py` → `tab_design`, `tab_abl`, `tab_bdd`, `tab_suite`,
    `tab_regimes`, `tab_loop`, `supp_tables/holdout_table`, `supp_tables/rater_table`,
    `supp_tables/rater_attr_table`;
  * `make_tab_master.py`;
  * `make_tab_bddq.py`.
* No main-paper table is hand-typed any more.
* The loop table and the two rater tables in the supplementary had been hand-typed from
  earlier models and were stale. They are now regenerated from v2.6.

## 2. Number verification (`scripts/verify_paper_numbers.py`)

**Registry: 48 claims, all pass.** Each one is checked against an exact JSON path,
including values derived from the JSON:

* the abstract;
* latency;
* per-attribute human GT;
* BDD100K gains and losses;
* the holdout and its six paired deltas;
* the suite and its strata;
* the regimes and their tests;
* the rater leads;
* the loop;
* the design-table trade-offs;
* Table 2 cells.

**It caught three numbers I had derived from rounded values**, now corrected:

| claim | was printed | exact | now |
|---|---|---|---|
| BDD100K OOV loss vs SigLIP2 | 0.061 | 0.0603 | 0.060 |
| degradation-regime lead | +0.066 | +0.0652 | +0.065 |
| rater lead (all 550) | +0.071 | +0.0703 | +0.070 |

**Limitation:** registry values are matched as substrings of the LaTeX. The rater lead was
initially "found" inside 0.0701 in another table, so I recomputed every hand-derived
difference by hand as well.

**Provenance scan.**

* Every decimal in `main_cvpr.tex` and the `tab_*.tex` files has some numerically matching
  value in `results/cvpr/*.json`, at its displayed precision.
* This is weak evidence for 3-decimal numbers, since with thousands of JSON values a chance
  match is likely. The registry is the real check.

## 3. Final compile and page-by-page inspection

### Main paper (`main_cvpr.pdf`)

* **Layout:** 10 pages; **the main text ends at the bottom of page 8 and the references
  start at the top of page 9**.
* **What had to move to fit the new content** (Table 2 holdout column, holdout paragraph,
  latency text):
  * the ablation and BDD100K summary tables went to the supplementary (their numbers stay
    in the text);
  * Figures 3 and 4 shrank to 86% and 75% of the text width;
  * the Section 6 aphorism and the "For the community" heading were folded into other
    text;
  * several paragraphs were tightened.
* **Inspection, page by page, at 100 dpi:**
  * Page 2: the two-tier Figure 1.
  * Page 6: Table 2 with both SceneMiner rows, the holdout column, latency 3.51 / 2.47 ms,
    neutral and listing CARIM-style rows, and the LLM-compiled RefAV-style row.
  * Page 7: Figure 3 with 42 distinct images and no grey stamps.
  * Page 8: Figure 4.
* **Figure 1 caveat:** its smallest fonts are 4.4–6.8 pt. It is legible zoomed in, but
  small in print.

### Supplementary (`supp_cvpr.pdf`)

19 pages, checked at 40 and 75 dpi. New material:

* detailed Stage-1/2 diagrams;
* the re-implemented-baseline specifications;
* the gate grid with the conditional variant marked as chosen;
* the ablation and BDD100K summary tables;
* the holdout section and table (small p-values printed as "< 0.001");
* the per-query BDD100K table from v2.6;
* the regenerated rater tables;
* the licence section;
* reproduction commands for v2.6.

### Both

No undefined references or citations, and no LaTeX errors.

## 4. Anonymity, final PDFs

* **`check_anonymity.py`: PASSED.**
* **Paper PDFs:** the info dictionary holds only creation and modification dates.
  `\hypersetup` blanks the author, title, creator and producer fields, and
  `\pdfsuppressptexinfo=-1` removes the pdfTeX banner.
* **Figure PDFs:** the info dictionary is empty (`strip_pdf_metadata.py`).

## 5. Documentation

* **`README.md`:**
  * the headline section is rewritten for v2.6 (portable mode, holdout, latency, the
    honest external picture);
  * reproduction commands for Milestones 27–31 are added.
* **`MASTER_REPORT.md`:**
  * sections 27–31 are added;
  * the final claims are updated: SceneMiner is level with SigLIP2 externally, and the
    conjunction fusion is significantly better.
