# Milestone 39 — Submission rewrite, new main figure, Overleaf package

## Paper (`paper/main_cvpr.tex`)

### Structure

The paper was reorganised so it reads as problem → method → evidence:

| # | section |
|---|---|
| 1 | Introduction |
| 2 | Related work |
| 3 | Setup and three hidden failure modes (3.1 circular benchmark, 3.2 contaminated certifier, 3.3 silent modality failure) |
| 4 | SceneMiner |
| 5 | Experiments (protocol, main comparison, reliability of the human labels, where learned fusion helps, design progression) |
| 6 | Do the mined scenes break perception? |
| 7 | Conclusion, with a short scope paragraph |

### Text

* **Abstract and introduction:** rewritten for plain, direct prose.
* **Contributions:** reduced to three — the method, the diagnostic audit, and the
  human-grounded evaluation.
* **"What we do not claim" and the long limitations paragraph:** replaced by a short
  scope paragraph. The facts they carried are still stated in the results, with their
  statistics:
  * level with SigLIP2 externally;
  * the reranker is stronger outside degradation, at 2300× the cost;
  * the conjunction control is higher.
* **Wording corrected during the claim-by-claim pass:**
  * the +0.066 lead over the lookup on conjunctions now carries its p = 0.063;
  * the lead over the lookup under rater 2 is now "positive, but its interval includes
    zero";
  * the external-transfer claim in the abstract is limited to paradigms that run on
    single images.

### Figures

* **Figure 1** is the authors' new architecture figure (`paper/figures/fig1_main.jpg`).
  * The title and subtitle bands were cropped; the caption replaces them.
  * EXIF-free.
  * The caption was rewritten to match the figure.

### Tables

All tables are generated from the result files.

* **Table 1 (new), concept-gate decisions** (`scripts/gate_examples.py` →
  `milestone39_gate_examples.json`, `tab_gate.tex`).
  * Nine queries with γ_deg, the active context classes, and the dominant term of
    Eq. 1.
  * It includes the remaining false trigger, "snow covering the road" (0.726).
  * The gate was checked to be identical across the five seeds.
* **Table 2, the main comparison: now 8 columns, grouped** (`make_tab_master.py`).
  * **Columns:**
    * human-labelled degradation;
    * BDD100K and the holdout;
    * Stratum 5;
    * the unweighted mean of those four (`milestone39_main_table.json`);
    * ms per query.
  * **Mean of the four benchmarks:**

    | method | mean |
    |---|---|
    | SceneMiner | 0.529 (best) |
    | SceneMiner (portable) | 0.522 |
    | per-query VLM reranker | 0.516 |
    | SigLIP2 | 0.478 |

  * **Rows dropped from the main table:** SigLIP v1 and the listing-caption CARIM
    variant.
  * **Where the full table went:** the 10-column version is now in the supplementary
    (`tab_master_full.tex`, `supp_tables/master_full_table.tex` without citations).
* **Table 3, second rater: transposed.** Attributes are now the columns, so the table is
  readable at column width.
* **Table 5, design progression:** one number per cell. The calibrated value and the
  skew moved to the text and the supplementary.

### Layout

* The four column tables are scaled to column width only by 5–15%.
* The main text ends on page 8, about mid-column; the references start on page 9.

### Supplementary

* The full comparison table was added to S2.
* A paragraph on rejecting the object cap (recall 0.45 at a 1.3% false-flag rate) was
  added to S1; it had previously appeared only in the main paper's limitations.
* Manual references to main-paper table and section numbers were updated to the new
  numbering.

## Verification

* **`verify_paper_numbers.py`:** 89 registered claims pass, and every decimal in the
  main text and all tables has a source. The changes to the registry:
  * claims that moved out of the main text now point to the file that holds them;
  * the derived "0.008" gap was dropped (both values remain registered);
  * Milestone 39's means and gate values were added.
* **`check_anonymity.py`:** passes on the paper directory and on the Overleaf zip. Zip
  mode now parses PDF and image members the way file mode does: a raw-byte scan flagged
  "umd" inside compressed image streams (10 hits, all inside binary streams, none in
  text or metadata).
* **Compile:** main 10 pages, supplementary 22 pages. Both have 0 errors, 0 undefined
  references and 0 overfull boxes, in the repo and in a clean copy of the Overleaf
  project.

## Overleaf package (`scripts/build_overleaf.py --out <dir>`)

* **What it copies:** exactly the files the two documents read.
* **Renames:** `main_cvpr.tex` → `main.tex` and `supp_cvpr.tex` → `supp.tex`.
* **README:** included.
* **Checks:** it test-compiles the project in a clean copy and then zips it, about 5 MB.
* **Where it lives:** the zip and the compiled PDFs are built outside the repository.
