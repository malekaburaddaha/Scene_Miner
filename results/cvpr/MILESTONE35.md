# Milestone 35 — Related work (35 fetched citations) and the pre-submission fixes

## 1. Related work: 11 new entries, all fetched (`scripts/fetch_related_work_m35.py`)

### How the entries were fetched

* **Identity:** each paper is identified by an **exact normalised-title match on
  OpenAlex**.
* **Published record:** authors, venue and pages come from **Crossref, by the DOI
  OpenAlex links**.
* **Raw responses:** `paper/bibcheck/related_work_check_3_raw.json`.
* **Parsed records:** `related_work_check_3.json`.
* **Generation:** `scripts/build_bib.py` writes the entries. None is typed by hand.
* **Why not arXiv or Semantic Scholar:** the arXiv API returned HTTP 429 from this
  host, and Semantic Scholar's unauthenticated batch API also returned 429.

### The entries

| key | venue (from the fetched record) | DOI |
|---|---|---|
| `yogamani2019woodscape` | ICCV 2019 | 10.1109/iccv.2019.00940 |
| `sakaridis2019darkzurich` | ICCV 2019 (Dark Zurich paper, "Guided Curriculum Model Adaptation …") | 10.1109/iccv.2019.00747 |
| `sakaridis2018foggy` | IJCV 126, 2018 | 10.1007/s11263-018-1072-8 |
| `li2022coda` | ECCV 2022 (Crossref container "Computer Vision – ECCV 2022") | 10.1007/978-3-031-19839-7_24 |
| `blum2021fishyscapes` | IJCV 129, 2021 | 10.1007/s11263-021-01511-6 |
| `sima2024drivelm` | ECCV 2024 | 10.1007/978-3-031-72943-0_15 |
| `qian2024nuscenesqa` | AAAI 2024 | 10.1609/aaai.v38i5.28253 |
| `deruyttere2019talk2car` | EMNLP-IJCNLP 2019 | 10.18653/v1/d19-1215 |
| `marcu2024lingoqa` | ECCV 2024 | 10.1007/978-3-031-72980-5_15 |
| `feng2019activelidar` | IEEE IV 2019 | 10.1109/ivs.2019.8814236 |
| `luo2023crb` | **arXiv preprint 2301.09249** | — |

### Two corrections to the task's identifiers

* **CRB:** arXiv `2303.08040`, as given in the task, is a different paper ("Beyond
  Demographic Parity"). CRB is `2301.09249`.
* **CRB's venue:** no fetched record (OpenAlex, OpenReview) confirms ICLR 2023;
  OpenReview lists it only as "CoRR 2023". It is therefore cited as the arXiv
  preprint.
  * Its authors come from the OpenReview/DBLP record. OpenAlex's author list
    duplicates "Zi Helen Huang".
  * **Please check whether you want the ICLR version.**

### Count

The bibliography now has **35 entries**, all cited; it had 24.

### Text added to Section 2 (about 9 lines)

* Foggy Cityscapes, Dark Zurich, WoodScape (lens soiling), CODA and Fishyscapes are
  labelled sets to train and test on; we retrieve such scenes from an unlabelled
  corpus by language.
* Driving VQA (Talk2Car, nuScenes-QA, DriveLM, LingoQA) answers questions about a given
  scene; retrieval inverts it.
* Active learning selects scenes a 3D detector is uncertain about; mining selects by a
  described condition.

## 2. The five fixes

### #1 Venue

`\def\confYear{2027}`, so the header reads "CVPR 2027 Submission". The supplementary is
a plain article with no CVPR header; I only corrected its stale title comment.
`\paperID` is still `*****` for you to fill.

### #2 Page budget

**Final layout:**

* The main text ends on page 8. Its last column stops about **7 lines** above the
  bottom (measured at 60 dpi, about 9.3 px per line).
* A `\clearpage` keeps the references at the top of page 9.

**What made room:**

* the BDD100K summary table (its numbers are in Table 2 and the text) moved to the
  supplementary;
* Figure 2 (audits) narrowed to 0.82 `\textwidth`;
* about 25 lines of prose tightened, including related work, setup, human GT,
  premise, retrieval-to-breakage, Discussion and Limitations;
* the "False-negative rescue" and "Training" paragraphs merged.

**Floats:** the regimes table floats `[!htb]`. Page 7 is now Figure 3, then Tables 3
and 4 side by side, then text, with no stranded single line.

### #4 Numbers and counts

**Section 8** states the 0.735 / 0.737 distinction explicitly:

> SceneMiner leads all twelve alternatives (0.794; next the naive router 0.735 and the
> lookup 0.732; the strongest no-learning control, late fusion of SigLIP2 and the
> lookup, scores 0.737, Tab. 3)

The abstract now cites only 0.737, the best no-learning control.

**Counts:** "twelve alternatives" is used everywhere: twelve methods in thirteen Table 2
rows, because the CARIM-style baseline runs with two caption prompts. The intro adds
"and a per-query VLM reranker".

### #5 Abstract

The abstract is rewritten, 17 lines down from 23. It keeps:

* 4.68M parameters and 3.5 ms per query;
* +74% detector contamination;
* 0.794 / 0.829, and 0.798 against the second rater (κ = 0.80), against the 0.737
  control;
* the external tie on 3000 BDD100K images, with the per-query reranker better there at
  2300× the cost;
* −9.1 points of front-sector mAP and −19.7 points of VRU recall on val150.

The audit numbers 1.000 and 0.004 left the abstract; they remain in Sections 4 and 6.

## Verification

* **`verify_paper_numbers.py`: 84 claims, all pass.** New since M33: the reranker,
  backbone and seed-sd claims, the abstract's 0.737, and "3000", which is derived as
  BDD100K images plus holdout images.
* **Provenance scan:** zero unsourced numbers.
* **`check_anonymity.py`:** PASSED.
* **Compiles:** no undefined references and no overfull boxes in either document.
