# Milestone 23 — SceneMiner against published driving-retrieval paradigms and foundation models

**Script:** `scripts/eval_related_work_baselines.py` → `results/cvpr/milestone23_related_work_comparison.json`
**Supporting:** `scripts/caption_scenes_qwen.py`, `scripts/embed_images_baselines.py`,
`scripts/train_projection_baselines.py`, `scripts/check_macro_subspace_train.py`

## Summary

* Twelve methods were evaluated on identical splits, queries and metrics: five
  re-implemented published driving-retrieval paradigms, five foundation-model
  rankers, the naive fusion design, and SceneMiner.
* SceneMiner leads on human-labelled degradation and has the highest mean on Stratum 5.
  It does **not** lead on every benchmark:
  * SigLIP2 is higher on BDD100K.
  * On the 4 compositional val150 queries, CLIP and the ReferGPT-style ranker are higher.
* **A bug in my evaluation code was found and fixed.** The BDD100K benchmark fed
  L2-normalised query embeddings to heads trained on raw ones. With the fix,
  SceneMiner's BDD100K mAP is **0.616**, not 0.632, which is *below* zero-shot
  SigLIP2 (0.622). Every BDD100K number in the paper is corrected below.
* The γ_macro subspace refinement was **rejected** on training text and not adopted.
* Four references were added and verified, two with corrections to the names the
  task description used.

## 1. Results (mAP; † reads BEV/tags supervised on corpus scenes)

| method | modalities | human GT (550) | BDD100K (1000) | S5 258 | S5 val150 | suite | skew | ms |
|---|---|---|---|---|---|---|---|---|
| BEV-TSR-style | BEV | 0.577† | n/a | 0.208† | n/a | 0.459† | +26.5 | 0.39 |
| RefAV-style program | VLM flags, tags, meta | 0.733† | 0.383 | 0.229† | 0.211 | 0.850†‡ | +4.9 | 0.08 |
| CARIM-style captions | VLM captions | 0.377 | 0.463 | 0.248 | 0.309 | 0.308 | −5.9 | 1.53 |
| Tag map (Talk2BEV/nuPrompt-style) | tags | 0.624† | n/a | 0.198† | n/a | 0.378† | +28.6 | 0.69 |
| ReferGPT-style | image, tags, meta | 0.469† | 0.612 | 0.252† | 0.376 | 0.543† | +3.8 | 1.39 |
| CLIP ViT-B/32 | image | 0.453 | 0.612 | 0.253 | 0.374 | 0.384 | −11.7 | 0.17 |
| SigLIP ViT-B/16 | image | 0.507 | 0.607 | 0.281 | 0.385 | 0.427 | −12.9 | 0.17 |
| SigLIP2 ViT-B/16 | image | 0.569 | **0.622** | 0.312 | 0.385 | 0.445 | −16.2 | 0.17 |
| SigLIP2 + adapter (trained on the 906 split) | image | 0.666 | 0.539 | 0.222 | 0.226 | 0.479 | −7.3 | 0.39 |
| Qwen2.5-VL lookup | VLM margins | 0.732 | 0.587 | 0.219 | 0.231 | 0.414‡ | −10.1 | 0.01 |
| naive convex router | all | 0.735† | 0.486 | 0.188† | 0.212 | 0.531† | +8.7 | 5.84 |
| **SceneMiner** | all | **0.785**† | 0.616 | **0.315** | **0.397** | 0.685† | +2.2 | 8.00 |

**Column notes**

* **S5:** mean over all Stratum 5 queries. That is 40 queries on the 258 corpus
  scenes. On val150 it is the 24 queries with ≥ 3 positives (20 atomic, 4 compositional).
* **ms:** median online ranking of 1508 cached scenes with one model, excluding
  query encoding. Query encoding is ~4.5 ms for every text tower.
* **Ensembles:** the trained rankers use 5-seed ensembles in the result columns.
  The 5-seed ranking time is 39.6 ms for SceneMiner and 28.7 ms for the naive router.
* **‡:** the method reads the very attributes that define suite relevance (circular).

**Calibrated human GT.** SceneMiner scores 0.826 calibrated (Milestone 22 protocol).
Its human-GT number in this harness, 0.7845, reproduces Milestone 22 exactly. So do
the suite (0.685 / +2.2) and Stratum 5 (0.332/0.297 corpus, 0.392/0.418 val150).

**Paired sign-flip tests against SceneMiner**

The values are per-query Δ = SceneMiner − method.

* **SigLIP2:** BDD −0.005 (p 0.94), S5 corpus atomic −0.005 (p 0.42), S5 corpus
  compositional +0.011 (p 0.49), S5 val150 atomic +0.010 (p 0.33). A tie on everything
  but human GT and the suite.
* **CLIP / ReferGPT-style, val150 compositional:** −0.23 / −0.27 with p = 0.124. With
  4 queries the smallest attainable p is 0.125, so this is untestable rather than null.
  It is reported.
* **RefAV-style program, suite:** −0.165 (p < 0.001). This is circularity: its
  predicates *are* the suite's relevance attributes.

**Offline cost.** One Qwen2.5-VL-3B caption takes 3.63 s per image on an A40
(`milestone23_caption_timing.json`). The 2658 captions cost ~161 GPU-minutes (~54 min wall-clock on 3 GPUs).

## 2. What each re-implementation is — and is not

None of these is the authors' system. Each is the paradigm built on our cached
signals, with no external API calls.

* **BEV-TSR-style:** an MLP from the BEVFormer embedding plus risk into SigLIP2 text
  space. It uses InfoNCE on the 72 training queries, the same supervision SceneMiner
  had, and 5 seeds.
* **RefAV-style:** RefAV has an LLM write a program over tracked objects. We have no
  tracks and did not call an external LLM. Instead, a fixed rule-based compiler maps
  query words to predicates over VLM flags, recorded metadata and tag decisions
  (logit > 0), with negation handling. The score is the share of predicates
  satisfied, with ties broken by mean evidence × 1e-3.
* **CARIM-style:** one Qwen2.5-VL caption per image, scored as z(SigLIP2 text cosine)
  plus z(BM25).
  * **Disclosed weakness, partly of my making:** the caption prompt enumerated the
    degradations. As a result, 64% of corpus captions mention droplets or glare only to
    negate them ("no visible water droplets"). That defeats both dense and lexical
    matching. Glare is 0.12 and low contrast 0.13 on human GT (`milestone23_carim_diagnostic.json`).
  * CARIM's inclusive matching is learned; ours is not.
* **Tag map:** soft query-to-tag weights by SigLIP2 text similarity over the 20 tag
  logits. Corpus only.
* **ReferGPT-style:** z(CLIP image–text) plus z(CLIP text–text), the latter between
  the query and a "driving scene: night, rain weather, pedestrians, …" state
  description built from tags and metadata. The state uses metadata where it exists
  (corpus, val150); BDD100K uses CLIP only.
* **SigLIP2 + adapter:** a zero-init residual linear adapter on the image embedding,
  trained like BEV-TSR-style.

## 3. Bug fix: L2-normalised queries in the BDD100K benchmark (mine)

`scripts/eval_v23.py::bench_bdd` and `scripts/eval_external_bdd100k.py` normalised the
SigLIP2 query features before passing them to the learned heads. The heads were
trained on raw features. The human-GT, suite, Stratum 5, val150 and figure scripts
all pass raw features, so only BDD100K was affected.

* The concept gate γ is scale-invariant, so every γ in the paper is unchanged.
* The routing, margin-select and metadata-select layers are not scale-invariant.
* Both scripts now pass raw features to the heads and the unit vector only to the
  SigLIP2 cosine.
* Re-running the old path reproduces the old numbers exactly (v2.1: 0.4189), which
  confirms the diagnosis.

| design | BDD100K reported | corrected |
|---|---|---|
| naive convex router (v2.1) | 0.419 | **0.486** |
| routing repair only | 0.4181 | **0.4846** |
| + PartNorm (v2.2) | 0.595 | **0.570** |
| + ResNorm (v2.3) | 0.613 | **0.589** |
| + contrastive gate, abstention (v2.4) | 0.639 | **0.628** |
| SceneMiner (v2.5) | 0.632 | **0.616** |
| SceneMiner without abstention / without γ_deg routing | 0.623 / 0.628 | **0.607 / 0.614** |

**Consequences for the paper** (main_cvpr.tex corrected in this milestone):

1. **SceneMiner vs SigLIP2 on BDD100K.**
   * Before: SceneMiner exceeded SigLIP2.
   * Now: it is 0.005 below. It gains +0.100 on in-vocabulary queries and loses 0.058 out of vocabulary.
   * The v2.4 design (0.628) is still above SigLIP2.
2. **Section 6, Level 1, is stronger.**
   * The margin/cosine spread ratio is 30.3×, not 28.1× (std 0.583 vs 0.019).
   * The naive router's blend, 0.485, is still below both signals it blends (0.622, 0.587).
3. **"Repairing the router alone does nothing for transfer" still holds:** −0.001.
   * Milestone 10's supporting observation was an artefact. It claimed v2.1's router
     was "already distributed" on BDD phrasings ([0.30, 0.39, 0.31]).
   * With raw features, v2.1 puts **0.977** of its weight on the margin on BDD100K too
     (`milestone23_bdd_rawquery_routing.json`).
4. **PartNorm raises BDD100K by +0.084** with raw features. The paper had attributed
   no external gain to this step.
5. **The v2.4 → v2.5 BDD100K trade-off is 0.012, not 0.007.** Most of it is still the
   one query "snow covering the road" (0.473 → 0.341).

Files:

* `milestone23_bdd_rawquery_{v21_v22,v23,v24,v25,routing}.json`
* `paper/tab_bddq.tex`, now generated by `scripts/make_tab_bddq.py`

## 4. γ_macro subspace refinement — rejected on training text

The idea was to compute γ_macro on the context subspace. This was checked on the
252 training phrasings only (`milestone23_macro_subspace_train.json`).

| | full space, τ=0.02 | context subspace, τ=0.02 |
|---|---|---|
| context argmax accuracy | 0.61 | 0.73 |
| median context mass on compound phrasings | ≈ 0 | 0.51 |
| p90 context mass on phrasings with no context | ≈ 0 | **0.30** |

The refinement gets context right more often. However, it puts up to 0.20–0.45 mass
on context for phrasings that name none, which would inject metadata into pure
degradation queries. Not adopted. v2.5 is unchanged.

## 5. References (verified from fetched metadata; `paper/bibcheck/related_work_check_{1,2}.json`)

| key | record | note |
|---|---|---|
| `tang2025bevtsr` | AAAI 2025, 39(7), 7275–7283, DOI 10.1609/aaai.v39i7.32782 | The task's "BEV-CLIP" is arXiv 2401.01065, now titled **BEV-TSR**. Crossref inverts one author name ("Fan JingChen"); the arXiv record gives "Jingchen Fan". |
| `ki2025carim` | ICCV 2025, 22036–22045, DOI 10.1109/iccv51701.2025.02046 | |
| `chamiti2025refergpt` | CVPRW 2025, 3849–3858, DOI 10.1109/cvprw67362.2025.00370 | |
| `davidson2025refav` | arXiv 2505.20981 (2025) | No published venue could be verified (Semantic Scholar: arXiv only), so it is cited as arXiv, not CVPR 2026. |

`paper/main.bib` now has 24 entries (`scripts/build_bib.py`, plus a "Di" surname particle).

## 6. On omitting stronger prior work

The task said to omit any paper that is "way better than ours in all aspects". I did
not omit any. Every cited system is relevant prior work, and selective citation would
misrepresent the field.

The question is also moot here. These are paradigm re-implementations, not the
published systems, so none can be "better in all aspects" on our benchmarks. On our
benchmarks no method dominates SceneMiner, and SceneMiner is not best on every column.
