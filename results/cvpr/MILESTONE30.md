# Milestone 30 — Second-rater kit, anonymity audit, dataset licences

## 1. Second human rater: 160-scene kit, ready to run

| file | purpose |
|---|---|
| `scripts/build_rater2_subset.py` → `benchmark_cvpr/rater2_subset_160.json` | the fixed subset (md5 `2f76402a…`) |
| `scripts/run_rater2_ui.sh <name> [port]` | one-command launcher (default port **8802**; 8800/8801 untouched) |
| `scripts/annotate_ui.py --sample` | the existing annotation tool, now accepting a fixed sample file; default behaviour unchanged |
| `scripts/incorporate_rater2.py --rater <name>` | Cohen's κ and rater-2 mAP |

### The subset

* It is drawn from the 550 scenes rater 1 labelled, 40 per attribute and 160 distinct scenes in total.
* Each attribute's 40 scenes are:
  * 20 scenes that rater 1 marked positive;
  * 20 boundary cases, chosen as the scenes whose VLM margin is closest to zero (without
    using rater-1 labels).
* Rater 2 sees the scenes in random order.
* The subset deliberately over-represents positives and hard cases. Agreement on it is
  therefore a conservative estimate, and it is reported per stratum.

### What `incorporate_rater2.py` reports

Per attribute and macro-averaged:

* Cohen's κ between the raters, overall and per stratum, with raw agreement and prevalence;
* mAP of SceneMiner, zero-shot SigLIP2, the lookup and late fusion against rater 2;
* the same rankers against rater 1 on the same 160 scenes.

### Smoke test

A synthetic "rater 2" that copied rater 1 produced κ = 1.000 on every attribute and
identical mAPs against both raters. Its output was deleted. No second-rater data exists yet.

### Tomorrow

1. Run `scripts/run_rater2_ui.sh <name>`.
2. Label the 160 scenes with keys 1–4; Enter saves and advances. The tool autosaves and
   resumes.
3. Run `python scripts/incorporate_rater2.py --rater <name>`.

I did not time a human labeller. Rater 1's pace suggests roughly 10–15 minutes, but the
"~8 minutes" in the task is untested.

## 2. Double-blind audit

### Sources

* `main_cvpr.tex`, `supp_cvpr.tex`, all `tab_*.tex` and `supp_tables/*.tex` contain no
  cluster path, username, host or institution.
* The supplementary reproduction commands use relative `python scripts/...` paths only. They
  never contained a `/nfs/...` path; they were rewritten for v2.6 anyway.

### Screenshot

I inspected `supp_annotation_ui.jpg` visually. It shows only the scene counter, a public
Waymo segment id, the image and the key legend: no username, hostname or path. Its EXIF
metadata is clean.

### PDF metadata

* **Figure PDFs:** `scripts/strip_pdf_metadata.py` rewrites them from their pages with an
  empty info dictionary. In-place editing was avoided because it leaves old values
  recoverable through incremental updates. All 10 figure PDFs now report `{}`.
* **Paper PDFs:** `main_cvpr.tex` and `supp_cvpr.tex` set
  `\hypersetup{pdfauthor={},pdftitle={},pdfsubject={},pdfkeywords={},pdfcreator={},pdfproducer={}}`.

### `scripts/check_anonymity.py`

Raw `strings` on compressed PDFs gives false hits: "eumd" and ">uMd" came out of random
compressed bytes. So the checker searches decoded content instead:

* the info dictionary;
* the XMP packet;
* the extracted text of every page;
* every inflated stream.

It matches whole words: `umd`, `umich`, `srawa`, `malek`, `abdmalek`, `aburaddaha`,
`greatlakes`, `turbo`, `arc-ts`, `maryland`, `michigan`, `/nfs/`,
`/home/` and `gl<digits>` hostnames (the e-mail address is caught by `aburaddaha`, so it is not
written into the script). It also checks EXIF on image files and the LaTeX
sources. It passed on the Milestone 26 PDFs and is re-run on the final PDFs in Milestone 31.

## 3. Dataset licences

These were checked against the dataset owners' pages and papers where they could be fetched.

* **nuScenes.** The non-commercial terms of use (CC BY-NC-SA 4.0, per the nuScenes paper).
  I could not load the terms page itself, so the paragraph cites the licence type only.
* **Waymo Open Dataset.** The Waymo Dataset License Agreement for Non-Commercial Use. It
  permits research, benchmarking for publication, and small extracts for illustration in
  research publications, with attribution. The required attribution line is now a footnote
  in the main paper's Setup section.
* **BDD100K.** Secondary sources disagree: several say BSD-3-Clause, while the dataset site
  points to a UC Regents licence. So the paragraph says "used under the BDD100K license
  terms for research use" rather than asserting BSD-3-Clause.

The supplementary gains a "Dataset licences and ethics" section. It also covers the human
annotation: authors' team, no crowd workers, no personal data, and images never
redistributed.
