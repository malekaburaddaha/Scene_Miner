# Milestone 37 — Final verification, visual audit, push

## Number verification (`scripts/verify_paper_numbers.py`)

**84 registered claims, all pass.** Since M32, these were added:

* rater-2 κ, its range and stratum, VLM–human κ, SceneMiner graded by rater 2, and the
  three bootstrap leads;
* the reranker's BDD100K score, its holdout and Stratum-5 leads and their p-values, the
  8.2 s latency and the 2300× ratio;
* the human-GT bootstrap and its CI, the low-contrast APs, and "3 of 66";
* the backbone numbers (0.569 → 0.543, 0.408 → 0.474, 0.784, 0.468, 0.650);
* the abstract's 0.737 and "3000", derived from the result file;
* the SceneMiner seed sd, checked as the exact table cell in Tables 1 and 2.

**Matching** is whole-number (from M33), so a value cannot "pass" as a substring of a
longer number.

**Provenance scan:** zero unsourced numbers in `main_cvpr.tex` and every `tab_*.tex`.

## Compile

**Main paper (`main_cvpr.pdf`):**

* 10 pages; the header reads "CVPR 2027 Submission #*****".
* The main text ends on page 8, with the last column stopping about **10 lines** above
  the bottom of the page.
* The references start at the top of page 9.

**Supplementary (`supp_cvpr.pdf`):** 22 pages.

**Both:** no undefined references, no overfull boxes, no LaTeX errors.

## Visual audit

**Pages inspected:** all 10 main pages at 80 dpi, Tables 1 and 2 and Figure 2(a) at 200
dpi, and an overview of all supplementary pages plus pages 5–7 at 60–85 dpi.

**Problems found and fixed:**

* **Figure 2(a), lookup bar:** it showed 0.390, from the Milestone 10 implementation,
  while Table 2 shows 0.414. The four non-index bars now come from the Table 2 run.
* **Supplementary suite table, lookup row:** same fix. It now uses the Table 2 run's
  per-query APs grouped by stratum (all 0.414, skew −10.1).
* **`\best{$…$}` did not render math in bold:** it is now `\textbf{\boldmath #1}` in
  both documents.
* **Table 1 was small (about 6 pt):** two row labels were shortened ("routing repair",
  matching the text's wording; "conditional removal").
* **Reference [3] read "The fishyscapes benchmark":** "Fishyscapes" is now protected in
  `build_bib.py`.
* **Section S6's heading landed after its three tables:** the tables are pinned
  (`[H]`) and the section has a lead-in sentence.

Figure PDFs were re-stripped; every info dictionary is empty.

## Anonymity (`scripts/check_anonymity.py`)

**PASSED** on:

* `main_cvpr.pdf`, `supp_cvpr.pdf` and every figure PDF and image;
* every `.tex` and `main.bib`;
* all **255 members** of `paper/supplementary_code.zip`, rebuilt after the last script
  changes. The M36 build had 254: `scripts/paths.py` was not yet tracked then, so that
  bundle lacked a module the scripts import. It was never committed; the committed
  bundle includes it.

## Repository

* **Committed:** `paper/supplementary_code.zip`, about 1.2 MB.
  * **Correction to M36:** that commit's message says the bundle was included, but the
    blanket `*.zip` ignore rule kept it out. An exception (`!paper/supplementary_code.zip`)
    now tracks it.
* **README:** points to `scripts/paths.py`, the bundle, and MILESTONE27–37.
* **MASTER_REPORT:** sections 34–37 added. The final claims now include the reranker
  trade-off and a note on the CRB venue.
* **Commit checks:** no tracked file over 5 MB, and no AI attribution.

## Left for the authors

* Fill in `\paperID`.
* Switch to the official CVPR 2027 author kit when it is released, and re-check the
  page-8 margin.
* Decide whether CRB should be cited as ICLR 2023 (currently arXiv 2301.09249).
