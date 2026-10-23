# Milestone 17 — 8-page manuscript, supplementary, final repository check

## Manuscript (`paper/main_cvpr.tex` → `main_cvpr.pdf`)

**8 pages including references**: main text runs to ~60 % of page 8, then the
bibliography. Compiles with `pdflatex` with no errors, no undefined references
and no overfull boxes > 10 pt. All result floats precede the references (the
retrieval tables initially drifted past the bibliography and were moved).

Changes from the 6-page version:

* **SceneMiner v2.3** throughout: method section (PartNorm, concept-specificity
  gate, ResNorm, FN rescue, unchanged training recipe, implementation details),
  Audit Discovery 3 extended to two levels of incommensurability, and v2.3A
  presented explicitly as post-hoc.
* **Results**: human-GT table (v2.1/v2.2/v2.3/v2.3A, glare FN column), ablation
  table, BDD100K table with in-/out-of-vocabulary split, and a full-width
  per-query BDD table that shows the gate's behaviour query by query.
* **Stratum 5** section with two tables (overall and by category), including the
  contamination flags and the clean val150 column.
* **Premise** section extended with the retrieval → breakage loop (Table 6) and
  Figure 4.
* **Related work**, a **re-audit table** of the earlier version's diagnostic
  claims (one reproduces, seven do not), and a **19-entry bibliography**.
* Figures 1–3 updated from v2.2 to v2.3 (architecture boxes, qualitative
  retrievals, circularity bar).

**The paper does not claim** that v2.3 eliminates the in-domain vs external
trade-off. It closes the in-domain side (best human-GT numbers in the project),
but it stays below SigLIP2 on BDD100K, regresses domain skew, and does not beat
SigLIP2 on semantic queries. The Discussion's "what we do not claim" paragraph
lists each of these.

## Figure 4 (`paper/figures/fig4_front_sector_degradation_and_semantics.pdf`)

(a) val150 CAM_FRONT mAP and ghost-FP rate by retrieval quartile. These are two
stacked panels, not a dual axis. (b) mAP per query stratum S1–S5, with the
human-graded S5 rows shaded and labelled, because S1–S4 relevance is circular.

Colours come from the validated categorical palette (slots 1–4, run through the
palette validator: all checks pass; contrast warning on aqua/yellow relieved by
distinct markers and a legend). Every figure was rasterised and inspected; fixes
applied in this milestone: legend over data and undersized markers (fig4),
clipped data point at x = 0.05 (fig4), annotation colliding with legend and mean
lines (fig3), row spacing (supplementary strips).

## Supplementary (`paper/supp_cvpr.tex` → `supp_cvpr.pdf`, 13 pages)

| section | contents | source |
|---|---|---|
| S1 | architecture and hyperparameter table, concept prompts, OOV bank, label-free scale evidence | code defaults, verified by grep |
| S2 | why the 150-query suite is circular | — |
| S3 | degradation annotation protocol with a **real screenshot** of the UI; Stratum-5 flag definitions and prevalence | `annotate_ui.py`, nuScenes tables |
| S4 | all 30 front-sector tests with bootstrap CI, permutation p and BH q | Milestone 11 JSON |
| S5 | per-query mAP for all **190 evaluated queries** (150 suite + 40 Stratum 5) | recomputed; suite means reproduce Milestone 14 exactly (0.6313 / 0.5843 / 0.4446) |
| S6 | retrieval strips across nuScenes, Waymo and BDD100K; val150 quartile figure | `make_figures.py` |
| S7 | the **240 training queries**, listed but not scored | queries_390.json |
| S8 | reproduction commands | — |

"All 430 queries": 190 are scored. The 240 training queries are listed only,
because scoring them would measure memorisation of text the head trained on.

**Annotation-UI screenshot.** It was taken with headless Firefox against a
throwaway server instance on port 8812, with a scratch output directory, so the
real annotations and ports 8800/8801 were untouched. The page's async fetch
completes after the headless capture fires, so the final capture renders the
tool's own HTML and script with one real scene record inlined (droplets + glare,
as the rater labelled it). The caption says so.

## Final repository check

See the commit for this milestone. Checked: single author, no AI attribution in
any commit message or tracked file (the only vendor-name strings remain the RefProg
LLM-backend selector from Milestone 1), no checkpoints or datasets tracked, no
file over 5 MB (largest is the 1.0 MB supplementary PDF), clean working tree.

## For the authors before submission

* **Verify the bibliography.** The 19 entries were written from memory, and the
  arXiv numbers especially need checking.
* Drop the official `cvpr.sty` / `ieeenat_fullname.bst` into `paper/`. The
  `\IfFileExists` switch picks them up, and page count may shift slightly.
* Decide whether to lead with v2.3 (best in domain) or v2.3A (no semantic
  regression, low skew). The paper currently leads with v2.3 and presents v2.3A
  as post-hoc.
