# Milestone 32 — Figure legibility, typo audit, object×condition gate cap (not adopted)

**No reported number changes.** The model is still v2.6, unchanged.

## 1. Figures

### Figure 3 (`fig2_qualitative.pdf`) and the supplementary strips

**Script:** `scripts/make_qualitative_showcase.py`.

**Width and layout.**

* The figure now spans the full `\textwidth` (it was 0.86).
* The figure is laid out at 7 in, and the row height follows from 16:9 thumbnails.
* The label column shrinks from 17% to 13.5% of the width.
* Horizontal gaps go from 0.05 to 0.015; vertical gaps from 0.10 to 0.12, measured
  against taller rows.
* `bbox_inches="tight"` padding drops from 0.1 in to 0.02 in.
* On the page, each thumbnail is 0.950 in wide, against 0.757 in before (+25%),
  measured from the PDF bounding boxes.

**Query labels.**

* 8.5 pt bold, wrapped onto 3–4 lines.
* Set in DejaVu Sans Condensed Bold, so four lines fit within one row.
* No quotation marks and no italics.

**Annotation boxes.**

* Yellow boxes are drawn at lw 2.0 with a dark outer stroke (`patheffects.withStroke`,
  lw 3.6, `#111`).
* The inset box uses lw 1.8 with the same stroke.

**Inset.**

* Spans 45% of the thumbnail height and 42% of its width.
* Its zoom window has the inset's aspect ratio, so the inset is not distorted.
* Border: 2.0 pt yellow.

**Badges and titles.**

* Count badges: 7.5 pt bold with a dark edge.
* Frames whose dataset labels the class but which contain none show **"0 annotated"**
  in red. This replaces "no annotated pedestrians", which overflowed the cell.
* Corpus titles are 8.5 pt bold, panel headers 9 pt bold.

**Panel (b) header.** It now reads "Road users and objects (yellow: human annotations;
inset: smallest; red: none in frame)". The caption was changed to match.

**Selection is unchanged.**

* Figure 3: 42 cells, 42 distinct images, 13 of the 20 labelled cells show the class,
  maximum rank 18.
* Supplementary semantic strips: 60 / 60 / 28 of 48 / rank 8.
* The captions already state these numbers.

**`supp_semantic_strips.pdf`** uses the same cell style.

**`supp_failure_strip.pdf`** was re-laid out:

* Each query is a full-width 8.5 pt bold line above its row, with the selection reason
  right-aligned. The left label column is gone, so the thumbnails are wider.
* Method names (9 pt bold) and the legend sit in a header row.
* Spacer columns separate the three methods.
* ✓ and ✗ badges are 9 pt; the "constant score" note is 7 pt.

### Figure 1 (`fig1_architecture.pdf`, `supp_stage1.pdf`, `supp_stage2.pdf`)

**Script:** `scripts/make_fig1_architecture.py`, rewritten.

**Fonts.**

* Every label is set in mathtext: $\gamma_{\mathrm{deg}}$, $s_{\mathrm{head}}$,
  $s_{\mathrm{abst}}$, $m_{\mathrm{sel}}$, $\pi_{\mathrm{sel}}$, $\gamma_{\mathrm{macro}}$.
  No raw underscores remain.
* At print size the smallest font is 7.0 pt, enforced by an assert in the text helper.
* Box titles are 8–9 pt bold; the stage headers are 9 pt bold.

**Layout.**

* The query boxes were narrowed so the $q$ bus no longer crosses the SigLIP2 text box.
* The $s_{\mathrm{abst}}$ footnote that overflowed its tier was removed; the caption
  already defines $s_{\mathrm{abst}}$.

**Content.** The latency label is read from `milestone27_latency.json`: 3.5 ms.

**Supplementary crops.** `supp_stage1.pdf` and `supp_stage2.pdf` are crops of the same
drawing, so they have the same fonts.

### Checks

* Figure PDFs were re-stripped with `strip_pdf_metadata.py`.
* Pages 2 and 7 were rasterised at 200 dpi and inspected: every label is legible and
  nothing overlaps.

## 2. The three reported LaTeX typos are not in the source or the PDF

| reported | source | rendered (200 dpi crop) |
|---|---|---|
| γ_macro bullet, p. 4 | `\item $\gmac(q)$: the unprojected softmax mass` (`\gmac` = `\gamma_{\mathrm{macro}}`) | γ_macro(q) |
| "step 0", p. 5 | `base at step~$0$` | "step 0" |
| "without metadata in s_abst", Table 3 | `without metadata in $s_{\mathrm{abst}}$` (`tab_abl.tex`) | s_abst with a roman subscript |

All three render correctly. The underscores most likely come from copying text out of
the PDF, which flattens math subscripts. No edit was made.

## 3. Capping γ_deg on object×condition queries: tested on training text, not adopted

### Diagnosis (query side only, no evaluation score read)

| query | γ_deg | routing | Σγ_macro |
|---|---|---|---|
| "a large truck or bus on a wet road" | 1.000 | 0.25 cosine / 0.75 margin | 0.002 |
| "pedestrians in headlight glare" | 1.000 | 0.2 / 0.8 | – |
| "pedestrians crossing the road at night" | 0.083 | – | 1.000 |

**Hypothesis 1: the night metadata mass is too small.** False.

* Σγ_macro = 1.000 for "at night".
* $s_{\mathrm{abst}} = (\tilde p_{\cos} + \sum_m\gamma_m z_m)/\sqrt{1+(\sum_m\gamma_m)^2}$
  is already the unit-variance fusion $(p+z)/\sqrt2$.
* Normalising the metadata vector would therefore change nothing, so it was not
  implemented.

**Hypothesis 2: object×degradation queries are routed almost entirely to the
degradation margins.** Confirmed: γ_deg = 1, with 75–80% of the weight on the margin.

### Training-text rule

**Script:** `scripts/explore_object_cap_train.py` → `milestone32_object_cap_train.json`.

**The cap.** $\gamma_{\mathrm{deg}}' = \min(\gamma_{\mathrm{deg}}, 0.5)$, applied when a
detector says the query also names something outside the degradation and context
vocabulary. The value 0.5 is fixed by the task, not tuned.

**Positives.** No evaluation query exists in the training bank, so I built 120
synthetic conjunctions:

* each joins a training degradation phrasing with one of the bank's 12 generic
  out-of-vocabulary phrases, e.g. "dashcam footage showing a few water specks on the
  lens at a parking lot";
* the generic phrase is removed from the bank while scoring its own conjunctions.

**Detectors.**

* D1: best generic-class score minus best concept-class score.
* D2: 1 − squared norm of the query's projection onto the 4 concept and 7 context
  class means.

**Pre-declared rule, fixed before the run:**

* θ is the 99th percentile of each detector over the 156 training degradation
  phrasings, scored leave-group-out;
* pick the detector with the higher recall on the synthetic conjunctions;
* adopt the cap only if that recall is ≥ 0.80.

| detector | θ | false flags on degradation phrasings | recall on conjunctions | flag rate on the 240 training queries |
|---|---|---|---|---|
| D1 | −0.262 | 0.013 | 0.233 | 0.304 |
| **D2 (selected)** | 0.709 | 0.013 | **0.450** | 0.008 |

**Result: not adopted.**

* The best detector finds fewer than half the conjunctions at a 1.3% false-flag rate.
  The synthetic conjunctions overlap heavily with plain degradation phrasings: the D2
  median is 0.692 against a θ of 0.709.
* A cap triggered this unreliably cannot be selected on training text.
* Selecting it on evaluation queries would break the zero-leakage protocol, so I did
  not try.
* v2.6 is unchanged. Semantics×condition stays at 0.297 vs 0.327 for SigLIP2 + metadata
  (p = 0.016).
* The Figure 3(b) truck/bus row is unchanged: 3 of the 6 labelled cells show the class.

**In the paper.** The Limitations paragraph now states this in one sentence (recall
0.45 at a 1.3% false-flag rate). Both values are registered in
`verify_paper_numbers.py`, which now has 50 claims, all passing.

## 4. Layout and checks

**Main paper.**

* Figure 3 at full width plus the larger Figure 1 pushed about 20 lines past page 8.
* Figure 4 (front-sector quartiles and regimes) moved to the supplementary. Its numbers
  are in Tables 6 (regimes) and 8 (retrieval to breakage), and the text points to the plot.
* That freed more space than needed. The ablation table (Table 3) and the BDD100K
  summary table (Table 4), moved out in M31, are back in the main paper.
* Final layout: **main text ends at the bottom of page 8**; `\clearpage` puts the
  references at the top of page 9; 10 pages in total.
* Table 7's column spacing was tightened, removing a 3 pt overfull box.
* Table 6's source was moved after its paragraph. It now sits at the bottom of page 7's
  right column, instead of leaving one orphan line under Figure 3.
* No undefined references and no overfull boxes.

**Supplementary.** 20 pages; it now holds Figure 4. Its "Ablations and BDD100K summary"
section was removed, since both tables are back in the main paper.

**Checks.**

* `verify_paper_numbers.py`: REGISTRY PASSED (50 / 50).
* Provenance scan: zero unsourced numbers in `main_cvpr.tex` and every `tab_*.tex`.
* `check_anonymity.py`: ANONYMITY CHECK PASSED.
