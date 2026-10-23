# Milestone 22 — factored concept gate, paper reframing, final layout

## Task 1 — factored gate (SceneMiner v2.5, now the paper's model)

### What the data showed first

The diagnosis named two failures; both are real, and they are distinct:

* **6 of the 18 suite S3 queries are context × context** ("heavy rain at
  night", "dense traffic after dark"). For these γ ≈ 0 is *correct*, since no
  degradation is named. They failed because abstention discarded the recorded
  metadata (AP ≈ 0.03).
* **4 degradation × context compounds were wrongly gated off**: "slick pavement
  in the dark" (γ 0.001), "glare … after dark" (0.002), "glare … in rain"
  (0.000), "hazy … in rain" (0.090). This is the mutual-exclusion failure.

The Milestone-18 leave-one-out had missed the second failure because each
training compound's siblings (same base phrase, other contexts) stayed in the
bank. Gate selection now uses a stricter **leave-group-out**.

### The gate (`src/scene_mining/filter/sceneminer_v2_5.py`)

* **γ_deg**: the same contrastive posterior, computed after removing half of the
  context subspace (the span of the 7 context-class means) from query and bank.
* **γ_macro**: the unprojected posterior mass on each context class. It maps by
  class name to the recorded-metadata column it denotes (night, day, rain, dry,
  dense, sparse); "moderate" has no column.
* **s = γ_deg·s_head + (1 − γ_deg)·s_abst**, with
  s_abst = (p̃_cos + Σ γ_m z_m)/√(1 + (Σγ_m)²). Here z_m is the metadata column
  z-scored with training-scene statistics, and s_abst falls back to p̃_cos where
  no metadata exists (BDD100K).

### Selection on training text only (`scripts/explore_factored_gate_train.py`)

Positives: the 156 degradation phrasings, leave-group-out. Negatives: 84
context + 12 generic phrasings (leave-one-out), plus 60 synthetic
context × context conjunctions joined from two training phrasings. Rule, fixed
before evaluation: per variant, the smoothest τ with separation ≥ 0.99; then
maximise the share of positives with γ_deg > 0.9, subject to every negative
≤ 0.2.

| variant | τ | pos > 0.9 | compound p10 | neg max | conj max |
|---|---|---|---|---|---|
| single softmax | 0.01 | 0.98 | 1.000 | 0.009 | 0.061 |
| **half projection (chosen)** | **0.02** | **0.99** | 1.000 | 0.152 | 0.007 |
| full projection | 0.02 | 1.00 | 1.000 | **0.644** ✗ | 0.005 |

Honest caveat: training text separates these variants only weakly, because
training compounds use templates ("in daylight", "on an empty road") unlike the
evaluation phrasings that failed. The factorisation was motivated by evaluation
failures, and only its parameters were selected on training text. The paper's
Limitations paragraph says so.

On the 72 training queries, minimum γ_deg rose from 0.001 to 0.897. On
training text, γ_macro assigns every context phrasing to its own class, and the
generic phrases get no context mass.

### Results versus the targets

| target | v2.4 | **v2.5** | met |
|---|---|---|---|
| human GT 0.788 / 0.817 preserved | 0.7879 / 0.8165 | **0.7845 / 0.8263** | ≈ (−0.003 / +0.010) |
| BDD100K 0.6387 preserved, > SigLIP2 | 0.6387 | **0.6317** | > SigLIP2 ✓; −0.007 vs v2.4 |
| S5 atomic 0.337 / 0.383 kept | 0.337 / 0.383 | **0.332 / 0.392** | ≈ (corpus −0.005, p = 0.42; val150 +0.010) |
| S5 object × condition ≥ SigLIP2 + metadata (0.327) | 0.292 | **0.310** | ✗ (−0.017, p = 0.17; was −0.036, p = 0.0007) |
| suite S3 back toward 0.68–0.73 | 0.459 | **0.603** | partial |
| suite overall higher, skew low | 0.594 / +0.5 | **0.685 / +2.2** | ✓ |

* **BDD100K −0.007** comes from one query. Removing the context subspace raised
  "snow covering the road" from γ_deg 0.49 to 0.71 (0.566 → 0.484). It is reported
  in the paper's design-progression table and text.
* **Suite S2** (0.487 → 0.671) is partly circular: S2 relevance *is* recorded
  metadata, which s_abst now reads. The paper says so.
* **Conjunctions:** the gap to SigLIP2 + metadata narrowed from significant to
  non-significant (4/11 conjunctions won). It still trails. For night/rain the
  metadata is the same scene description that defines the ground truth, so that
  control is the correct bar.
* Rater robustness is unchanged: macro win vs the lookup in 92–100 % of trials,
  vs SigLIP2 in 100 %. In the val150 loop, the top-25 loses −0.057 mAP and
  −0.197 VRU recall.

## Task 2 — paper

* **Title:** "SceneMiner: Calibrated Multi-Modal Fusion and Diagnostic Auditing
  for Language-Guided Driving Scene Retrieval".
* **Framing:** a first-principles design and diagnostic study. Every reference
  to "an earlier version", "earlier milestones" and the re-audit table is gone,
  and **Table 12 is deleted**. The weaker designs appear as a **design
  progression** (Table 2: naive convex router → anti-collapse routing →
  ResNorm + single-prompt gate → contrastive gate + abstention → factored
  gate), each trained with the same recipe. That table also shows that no single
  design wins every column.
* **Section 6** now has three levels of silent modality failure: parts (28.1×),
  residual (std 6.31 vs 0.03–0.57), and the **width-1 LayerNorm**. The risk
  token takes one distinct value across 1508 scenes (std 0.0000) while 5-seed
  mAP is 0.311 ± 0.036 dead vs 0.307 ± 0.048 alive, on the predicate suite.
  Numbers verified against `MASTER_REPORT.md` / `MILESTONE1.md`.
* **One disclosure kept, in neutral wording, in Limitations:** text
  mean-centring, ResNorm and the gate factorisation were identified while
  diagnosing evaluation failures. None has a parameter fitted to evaluation
  data, and each is supported by training-only evidence. Removing this
  sentence would overstate how blind the design was.
* **Equation (1)** was slightly wider than the column, so its number fell to its
  own line. It is now split into two lines with arguments omitted; it renders in
  one piece with "(1)" on the line.
* **Greek letters:** checked at 150 dpi and by text extraction. γ, κ, τ and q
  render and extract correctly in Table 1, Table 8 and Table 10 of the previous
  PDF. The reported drop most likely came from copying text out of a viewer;
  the `mmap` package that would add Unicode maps for math fonts is not installed
  on this TeX.
* **Figure 1:** rebuilt for the factored gate (γ_deg, γ_macro, three-line
  formula box), with no text crossing box borders. It stays at the top of page 2.
* **Figure 2 (qualitative):** corpus headers enlarged (7.6 pt, bold) and the gap
  to Table 5 widened. Still exactly 2 images per corpus × 3 corpora.
* **Figure 3(a):** the "earlier version" bar label is now "naive convex router".
* **Page budget:** main text ends on page 8 (line 418); References begin at the
  top of page 9. No errors, undefined references, BibTeX warnings or overfull
  boxes.

## Task 3 — supplementary and inspection

Supplementary updated to the final model: title, specification (factored gate,
latency 7.93 ± 0.20 ms), full leave-group-out gate-selection grid, strip caption,
reproduction commands, corrected main-paper table references, and "Milestone"
removed from the BH-table caption. 13 pages, clean. Every page of both PDFs was
rasterised and inspected.
