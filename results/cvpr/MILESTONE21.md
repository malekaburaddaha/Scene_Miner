# Milestone 21 — final visual inspection, supplementary sync, push

## Every page inspected

All 10 pages of `paper/main_cvpr.pdf` and all 13 pages of
`paper/supp_cvpr.pdf` were rasterised and inspected, two to three pages per
image. Defects found and fixed in this pass:

| defect | fix |
|---|---|
| "-12pt" printed in every page header | The 2026 kit computes `\textwidth-12pt` inside `\LenToUnit`, which this cluster's TeX Live 2018 `eso-pic` prints literally. The kit's own unmodified sample shows the same artefact. Fixed with a one-line `\dimexpr…\relax` edit in `cvpr.sty`, marked in the file; output is identical on current TeX and the edit can be reverted on Overleaf. |
| "viaSpatiotemporal" in the BEVFormer reference | Crossref titles carried non-breaking spaces; whitespace normalised in `build_bib.py` |
| hyphenated title words kept capitals ("Pre-Training") while the style lowercased the rest | only acronyms and mixed-case names are now brace-protected; proper names (Waymo Open Dataset) kept |
| 6 overflowing tables (13–25 pt) | `\resizebox{\columnwidth}` |
| result tables drifting past the bibliography | float sources moved next to their sections |
| supplementary Figure S1 and Table S3 floating 5 pages from Section S3 | pinned with `[H]` |
| supplementary still describing v2.3 (spec table, prompts, latency, strip caption, reproduction commands) | updated to the single model |

Final state: main paper with no LaTeX errors, undefined references, BibTeX
warnings or overfull boxes; main text ends at the bottom of page 8; references
on pages 9–10. Supplementary: 13 pages, clean.

## Figures (all regenerated for the single model)

| figure | content |
|---|---|
| Fig. 1 | architecture with the contrastive gate and the abstention equation |
| Fig. 2 | qualitative retrievals, **2 per corpus × 3 corpora × 4 queries** (8 nuScenes / 8 Waymo / 8 BDD100K) |
| Fig. 3 | circularity bar and contamination histogram |
| Fig. 4 | (a) val150 quartiles; (b) three human-graded regimes against no-learning fusion controls |
| Supp. strips | 2 per corpus × 3 corpora × 6 queries |
| Supp. quartiles | val150 retrieval quartiles, single model |

Per the request during Milestone 18, **no qualitative figure lets one dataset
dominate**. Every strip has the same number of images from nuScenes, Waymo and
BDD100K, each ranked within its own corpus. The exact images picked are logged
in `results/cvpr/figure_retrieval_picks.json`.

## Documentation

`README.md` §6–8 and `results/cvpr/MASTER_REPORT.md` (new Part V and a
rewritten closing section) carry the final results, including what does not
hold.
