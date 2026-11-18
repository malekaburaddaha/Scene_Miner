# Milestone 34 — Per-query VLM reranker, larger backbone, five-seed sds

**Two findings contradict the expectations in the task, and are reported as measured:**

1. **The per-query VLM reranker does not "match or lose to" SceneMiner everywhere.**
   * It is **better** on external and semantic queries.
   * SceneMiner is better on human-graded degradation and the suite.
   * It is about 2,300× cheaper online (not >2,500×).
2. **Zero-shot So400m falls to 0.543 on human GT**, below ViT-B/16 (0.569). The task
   expected 0.58–0.61.
   * The fusion's in-domain lead carries over to the larger backbone.
   * Externally SceneMiner again tracks its backbone. "Scales cleanly" would overstate
     it.

## 1. Per-query VLM reranker (`scripts/eval_per_query_vlm_reranker.py` → `milestone34_per_query_vlm.json`)

### Protocol

* **Stage 1:** SigLIP2 ViT-B/16 retrieves the top 50 scenes **within each evaluated
  pool**, the setting most favourable to the baseline. The pools are the 550 human-GT
  scenes, 1000 BDD100K images, 258 Stratum-5 scenes, val150, 2000 holdout images and
  602 suite scenes.
* **Stage 2:** Qwen2.5-VL-3B scores each candidate with the yes/no log-odds of "Does
  this driving scene show {query}? Answer Yes or No."
  * The image files, chat template and Yes/No token sets are the same as for the
    cached margins.
  * The 50 candidates are re-ordered above the rest of the pool.
* **Scale:** 13,151 VLM calls, 4 shards on 4 A40s, about 22 minutes.
* **Identity check:** the script's SigLIP2 per-query APs equal Table 2's on every
  benchmark (max |Δ| 7e-6).

### Results

"Δ" is SceneMiner minus the reranker. "Positives in top 50" is the share of relevant
scenes SigLIP2 places in its top 50, which bounds the reranker.

| benchmark | SigLIP2 | reranker | SceneMiner | Δ (p) | positives in top 50 |
|---|---|---|---|---|---|
| human GT (550 scenes, 4 queries) | 0.569 | 0.635 | **0.794** | **+0.159** (scene bootstrap, CI [0.114, 0.206], p = 0.001) | 44% |
| BDD100K | 0.622 | **0.640** | 0.614 | −0.026 (0.60) | 14% |
| holdout | 0.408 | **0.442** | 0.400 | **−0.042 (0.004)** | 11% |
| Stratum 5 atomic, corpus | 0.337 | **0.355** | 0.330 | **−0.025 (0.001)** | 33% |
| Stratum 5 conjunctions, corpus | 0.286 | 0.336 | 0.286 | −0.050 (0.15) | 77% |
| Stratum 5 atomic, val150 | 0.382 | 0.407 | 0.390 | −0.017 (0.33) | 51% |
| Stratum 5 conjunctions, val150 | 0.396 | 0.436 | 0.416 | −0.020 (0.88) | 100% |
| 150-query suite | 0.445 | 0.501 | **0.694** | +0.194 (< 0.001) | 24% |

**Per human-GT attribute,** SceneMiner wins droplets (0.943 vs 0.893), wet road (0.976
vs 0.866) and low contrast (0.665 vs **0.119**). The reranker wins glare (0.662 vs
0.592).

**Low contrast shows the reranker's limit:** SigLIP2's top 50 holds only 3 of the 66
positives, and a reranker cannot recover what the first stage missed.

### Latency

Measured on one A40, median of 5 queries:

| setting | per query |
|---|---|
| batches of 10 | 8,171 ms |
| one image at a time | 11,862 ms |
| SigLIP2 stage alone | 6.7 ms |

At 8,171 ms against SceneMiner's 3.51 ms, SceneMiner is about **2,300×** cheaper online.

### Paper

* **Table 2:** a new row, "SigLIP2 top-50 + VLM rerank".
* **Section 8:** the paragraph "A per-query VLM reranker".
* **Discussion:** "What we do not claim" now includes "or a per-query VLM reranker
  there (which is better, at 2300× the online cost)".
* **Supplementary:** Table `tab:vlmrr`.

## 2. Larger backbone (`scripts/eval_larger_siglip2.py` → `milestone34_larger_backbone.json`)

### Setup

* **Backbone:** `google/siglip2-so400m-patch14-384` (4.6 GB, in the local HF cache).
* **Image embeddings** (1152-d) were extracted from the same image files for all four
  corpora.
* **Retraining:** SceneMiner was retrained for 5 seeds with the unchanged v2.6 recipe,
  via the new `--siglip-repo` and `--siglip-emb` options of `train_sceneminer_v26.py`.
  The concept bank and gate are rebuilt in the So400m text space from the same training
  phrasings. Nothing is tuned.
* **Evaluation:** the Table 2 evaluator, through optional environment variables:
  * `SM_SIG2_REPO` and `SM_SIG2_EMB` select the backbone;
  * `SM_METHODS` restricts the rows.
  * Defaults are unchanged.

### Results

| | human GT | BDD100K | holdout | S5 atomic | S5 conjunctions | suite |
|---|---|---|---|---|---|---|
| SigLIP2 zero-shot, ViT-B/16 | 0.569 | 0.622 | 0.408 | 0.337 | 0.286 | 0.445 |
| SigLIP2 zero-shot, So400m/14 | 0.543 | 0.624 | **0.474** | 0.373 | **0.395** | 0.454 |
| SceneMiner, ViT-B/16 | **0.794** | 0.614 | 0.400 | 0.330 | 0.286 | 0.694 |
| SceneMiner, So400m/14 | 0.784 | **0.650** | 0.468 | 0.352 | 0.345 | **0.695** |

**SceneMiner on So400m against its own backbone:**

| benchmark | Δ | p |
|---|---|---|
| human GT | +0.241 | (vs +0.225 on ViT-B/16) |
| suite | +0.242 | < 0.001 |
| BDD100K | +0.027 | 0.73 |
| holdout | −0.007 | 0.60 |
| Stratum-5 conjunctions | **−0.050** | **0.021** |

The last row is a regression on the larger backbone.

**Paper:** a short paragraph in Section 8 and the supplementary Table `tab:backbone`.

## 3. Five-seed sds (`scripts/run_seed_sd.sh`, `scripts/summarize_seed_sd.py` → `milestone34_seed_sd.json`)

### Method

* Every trained row of Tables 1 and 2 was re-scored **one seed at a time** by the same
  scripts that produced the tables.
* `SM_SEED` was added to the Table 2 evaluator, and `SM_SEED_GLOB` to the v2.1/v2.2-era
  scripts. Defaults are unchanged.
* `bdd_routing_rawquery_check.py` gained an `SM_OUT` override.

### Checks

* **Human GT** is scored per seed in the paper. The five-seed means reproduce every
  reported human-GT value exactly (asserted).
* **Calibrated human GT** for v2.1 and v2.2 does *not* reproduce from per-seed runs:
  0.754 vs 0.792, and 0.739 vs 0.766. Its protocol is not a per-seed mean, so the paper
  shows no ± on calibrated values.

### Values

| row | human GT (mean ± sd) | BDD100K single-model sd |
|---|---|---|
| SceneMiner | 0.794 ± 0.008 | 0.002 |
| portable | 0.762 ± 0.004 | — |
| naive router | 0.735 ± 0.013 | — |
| adapter | 0.666 ± 0.005 | — |
| BEV-TSR-style | 0.577 ± 0.033 | — |
| v2.2 | 0.706 ± 0.022 | — |
| v2.3 | 0.790 ± 0.010 | — |
| v2.4 | 0.788 ± 0.010 | — |
| v2.5 | 0.785 ± 0.016 | — |

**Consequence, now stated in the paper:** the last four designs of Table 1 are within
about one seed sd of each other on human GT. External transfer and the suite separate
them.

### Paper

* **Table 1:** subscript sds on human GT (zero-leakage), BDD100K and suite. The caption
  says what each sd means.
* **Table 2:** subscript sds on the human-GT cells of the five trained rows.
* **Supplementary:** the full table (`tab:seedsd`).

## Verification

`verify_paper_numbers.py` registers the new reranker, backbone and sd claims. All pass;
see Milestone 35 for the final count.
