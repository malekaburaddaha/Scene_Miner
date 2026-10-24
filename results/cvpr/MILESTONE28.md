# Milestone 28 — Upgraded baselines and an untouched, pre-registered external holdout

**Scripts:**

* `scripts/caption_scenes_qwen.py --prompt neutral`
* `scripts/refav_llm_compile.py`
* `scripts/build_bdd_holdout.py`
* `scripts/holdout_signals.py`
* `scripts/eval_related_work_baselines.py`

**Results:**

* `milestone28_related_work_comparison.json`
* `refav_llm_programs.json`
* `benchmark_cvpr/bdd_holdout_queries_36.json` (pre-registration)

The final model throughout is v2.6 (Milestone 27).

## 1. CARIM-style with neutral captions and clause-level matching

**What changed.**

* All 2658 corpus, BDD100K and val150 images were re-captioned on 3 GPUs with Qwen2.5-VL-3B
  and the requested neutral, affirmative prompt. The 2000 holdout images were captioned the
  same way.
* Dense score: the maximum over a caption's clauses (sentence and comma/semicolon splits)
  of the SigLIP2 text cosine.
* Final score: z(dense) + z(BM25).

**Result: the neutral prompt is not uniformly stronger.**

| CARIM-style | human GT | BDD100K | S5 258 atomic / comp. | S5 val150 atomic / comp. | suite |
|---|---|---|---|---|---|
| enumerating prompt, whole caption (M23) | **0.377** | **0.463** | 0.273 / **0.223** | 0.315 / **0.280** | **0.308** |
| neutral prompt, clause max-pool (M28) | 0.274 | 0.440 | **0.310** / 0.144 | **0.348** / 0.179 | 0.268 |

* **Why degradations got worse.** Neutral captions describe the scene and rarely mention lens
  droplets or glare at all, so degradation queries have nothing to match.
* **Why semantics got better.** On object queries the neutral captions name more road users
  and fewer denied conditions.
* **Reporting.** Neither prompt dominates, so the paper reports both rows.

## 2. RefAV-style with LLM-compiled programs

**How the programs were compiled.**

* The local Qwen2.5-VL-3B ran in text-only mode with greedy decoding.
* A fixed system prompt and 5 invented few-shot examples (checked to be none of the 231
  benchmark queries) compile each query into a JSON conjunction over:
  * the VLM flags;
  * the recorded metadata;
  * the 20 scene/object tags.
* Every program is cached in `refav_llm_programs.json`.
* 231 queries compiled: 2 unparsable outputs and 14 empty programs. An empty program ranks
  at chance.
* No external API was used.

**Result.**

| RefAV-style | human GT | BDD100K | S5 258 | S5 val150 | suite | holdout (all) |
|---|---|---|---|---|---|---|
| rule compiler (M23) | 0.733 | 0.383 | 0.251 / 0.207 | 0.220 / 0.166 | 0.850‡ | 0.153 |
| **LLM compiler (M28)** | 0.717 | **0.432** | 0.229 / 0.147 | 0.220 / 0.091 | 0.823‡ | **0.184** |

The LLM transfers better to unseen phrasings (BDD100K, holdout) and is slightly worse in
domain. The table carries the LLM version; the rule compiler stays in the JSON.

## 3. An untouched external holdout, pre-registered

### Data

* **Source:** the 10,000-image BDD100K mirror (`dgural/bdd100k`), which carries BDD100K's
  own human weather / time-of-day / scene tags and 2D boxes.
* **Draw:**
  * the 1000 images used since Milestone 8 are excluded;
  * from the remaining 7841 labelled images, 2000 are drawn with seed 1;
  * adverse conditions are oversampled (60%), exactly as in Milestone 8.
* **Composition:** 962 night, 183 snowy, 192 rainy, 3 foggy images.

### Pre-registration

* `benchmark_cvpr/bdd_holdout_queries_36.json` was written before any holdout score was
  computed (SHA-256 `c1e2c90d0d10…`; embedded `written_utc` earlier than every evaluation
  log). It was committed with Milestone 27, after the run, so the git history alone does not
  prove that order; the timestamps and the unchanged hash are the record.
* It holds 36 queries:
  * (a) 12 condition/degradation-proxy queries;
  * (b) 12 object queries graded by the human 2D boxes (area and count thresholds fixed in
    the file);
  * (c) 12 object × condition queries.
* Protocol: frozen methods, one run; a query with fewer than 5 positives is excluded on
  its label counts alone. This excluded only a3, "fog", which has 3 positives.
* The lookup baseline's margin column per holdout query was fixed from the query wording
  in the harness (`HOLD_CHANNEL`) before the run.
* All Stage-1 signals were computed for the holdout without scoring any query:
  * SigLIP2, CLIP and SigLIP image embeddings;
  * the four Qwen yes/no margins;
  * the neutral captions.

### Results: metadata-free mode (main; no method reads BDD100K tags, which are the ground truth)

| method | all (35) | (a) condition | (b) object | (c) object×cond. | P@5 |
|---|---|---|---|---|---|
| SigLIP2 ViT-B/16 | **0.408** | **0.601** | 0.344 | 0.295 | **0.709** |
| **SceneMiner** (portable = full here) | 0.400 | 0.575 | 0.344 | 0.295 | 0.691 |
| SigLIP ViT-B/16 | 0.388 | 0.589 | 0.305 | 0.288 | 0.663 |
| CLIP ViT-B/32 / ReferGPT-style | 0.353 | 0.588 | 0.255 | 0.234 | 0.589 |
| CARIM-style (neutral captions) | 0.293 | 0.453 | 0.269 | 0.171 | 0.440 |
| SigLIP2 + adapter | 0.239 | 0.499 | 0.119 | 0.122 | 0.303 |
| Qwen2.5-VL lookup | 0.231 | 0.483 | 0.116 | 0.116 | 0.258 |
| naive convex router | 0.215 | 0.456 | 0.109 | 0.100 | 0.251 |
| RefAV-style (LLM) | 0.184 | 0.402 | 0.116 | 0.050 | 0.214 |

BEV-TSR-style and the tag map cannot run on monocular images.

### Paired sign-flip tests, SceneMiner minus method

| method | Δ (all 35) | p |
|---|---|---|
| SigLIP2 | −0.008 | 0.28 (n.s.) |
| SigLIP | +0.012 | 0.42 |
| CLIP / ReferGPT-style | +0.048 | 0.013 |
| CARIM-style | +0.107 | 0.001 |
| lookup | +0.169 | < 0.001 |
| RefAV-style | +0.217 | < 0.001 |
| naive router | +0.185 | < 0.001 |

**SceneMiner vs SigLIP2 by family:**

* Object queries: the two rankings are identical (Δ = 0, p = 1.0). The gate gives
  γ_deg ≈ 0 and the holdout has no metadata, so SceneMiner abstains to SigLIP2 exactly as
  designed.
* Condition family: SceneMiner is lower (−0.026, p = 0.30, n.s.).

### Results: metadata-equipped mode (labelled circular)

This mode gives methods that read recorded metadata BDD100K's own time-of-day and weather
tags, which are also the ground truth of the condition queries.

* **SceneMiner:** 0.392 overall; (c) object×condition 0.316 vs SigLIP2 0.295.
* **ReferGPT-style:** (a) condition 0.712.

### What this establishes

* On an external corpus that no design decision ever saw, with high power (35 queries,
  2000 images), SceneMiner is **level with zero-shot SigLIP2**.
* It is significantly better than every other baseline:
  * CLIP and ReferGPT-style;
  * CARIM-style and RefAV-style;
  * the lookup and the naive router.
* It does not beat SigLIP2; the paper says so.
* Its measured advantage remains in-domain degradation (human GT).

## 4. Updated comparison table (v2.6, all benchmarks)

| method | human GT | BDD100K | S5 258 (40) | S5 val150 (24) | suite |
|---|---|---|---|---|---|
| SceneMiner (full) | **0.794**† (0.829 cal.) | 0.614 | 0.308 | 0.394 | **0.694**† |
| **SceneMiner (portable)** | **0.762** | 0.614 | 0.311 | 0.394 | 0.654 (top-5 skew −13.9) |
| naive convex router | 0.735† | 0.486 | 0.188† | 0.212 | 0.531† |
| Qwen2.5-VL lookup | 0.732 | 0.587 | 0.219 | 0.231 | 0.414‡ |
| RefAV-style (LLM) | 0.717† | 0.432 | 0.188† | 0.198 | 0.823†‡ |
| SigLIP2 ViT-B/16 | 0.569 | **0.622** | **0.312** | 0.385 | 0.445 |

The full table is `paper/tab_master.tex`, generated by `scripts/make_tab_master.py`.
