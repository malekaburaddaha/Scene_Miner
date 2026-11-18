# Milestone 41 — The [TODO] experiments of the v2 paper, and the end-to-end repository

Every number below was computed on CPU (`SM_DEVICE=cpu`) from the cached Stage-1 signals and
the saved checkpoints. The ready-to-paste tables are generated from the result files by
`scripts/make_m41_tables.py` into `MILESTONE41_tables.{tex,md}`.

## Experiments

### Table 2, late-fusion row (`eval_related_work_baselines.py`, method `late_fusion`)

**Definition.** z(SigLIP2 cosine) + z(a-priori VLM margin), with each part z-scored over the
ranked set.

**Reproduction of existing values:**
* the re-run reproduces every published SigLIP2, lookup and SceneMiner cell;
* late fusion reproduces 0.737 on the human labels.

**New values:**

| benchmark | late fusion | context |
|---|---|---|
| BDD100K-1000 | **0.650** | the best entry of that column; above SceneMiner (0.614, paired p = 0.43, 12 queries) and the reranker (0.640) |
| Holdout | 0.368 | SceneMiner (portable) +0.032; p = 0.050 unadjusted, Holm 0.150 over ten comparisons |

**Correction to the published semantic cell.** It is **0.306**, not 0.310. The published value
averaged the atomic mean (0.337) with the conjunction mean of a different query set (0.284, from
the 30-query synergy set).

**Latency.** ms/query: 0.32 ms (A40 component sum 0.18 ms + 0.14 ms host-side fusion overhead).

**Holm adjustment over the paper's nine holdout comparisons:**
* the published CLIP p = 0.052 is 0.051;
* with late fusion as a tenth comparison it becomes 0.064.

### E1, paired scene-bootstrap CIs (`e1_paired_bootstrap.py`, 2000 resamples)

Every point estimate reproduces the paper.

**Rater 1 (550 scenes):**

| comparison | macro Δ | 95% CI | p |
|---|---|---|---|
| SceneMiner (full) − late fusion | +0.057 | [+0.026, +0.085] | 0.001 |
| SceneMiner (full) − lookup | +0.061 | [+0.038, +0.083] | < 0.001 |
| SceneMiner (portable) − late fusion | +0.025 | [+0.003, +0.046] | — |
| SceneMiner (portable) − lookup | +0.030 | [+0.009, +0.047] | — |

**Rater 2 (160 scenes):**
* full − late fusion: +0.042 [+0.006, +0.079];
* full − lookup: +0.030 [−0.009, +0.077].

**Per attribute:**
* the gain is low contrast (+0.255 vs late fusion);
* late fusion is better on glare (−0.075 [−0.130, −0.024]).

### E2, training-free control (`sceneminer_tf.py`, `e2_build_training_free.py`)

**What the control keeps:**
* the same PartNorm parts, the concept gate, the fallback and the clamp;
* equal weights;
* prior entries selected by the concept bank;
* no learned parameter;
* PartNorm statistics recomputed label-free.

**Results:**

| benchmark | control | SceneMiner |
|---|---|---|
| Human labels, rater 1 | 0.752 | 0.794 |
| Human labels, rater 2 | 0.759 | 0.798 |
| BDD100K | 0.630 | 0.614 |
| Holdout | 0.404 | 0.400 |
| Semantic | 0.320 | 0.308 |
| Suite | 0.643 | 0.694 |

**Learned contribution:**
* +0.042 [+0.009, +0.074] (rater 1) and +0.039 [+0.002, +0.080] (rater 2);
* almost all of it is low contrast;
* the portable configuration is level with the control: +0.010 [−0.017, +0.037].

**Check.** A uniform (query-independent) prior gives 0.507 in full mode.

### Table 4 conjunction column (`table4_conjunctions.py`)

| query set | SceneMiner | SigLIP2 | late fusion | metadata fusion | SceneMiner − metadata fusion |
|---|---|---|---|---|---|
| Published (30 phrasings, 11 predicates) | 0.297 | 0.293 | 0.284 | 0.327 | p = 0.015 by predicate |
| 20 Stratum-5 conjunctions (Table 2 set) | 0.286 | 0.286 | 0.275 | 0.317 | −0.031, p = 0.097 by query (not significant) |

The SceneMiner value is the five-seed ensemble.

### E3, night/rain-controlled detection study (`e3_val150_stratified.py`, 5000 permutations)

**Table 6.** Reproduced for SceneMiner, the lookup, metadata and SigLIP2.

**Night/rain composition of the top 25:**
* 90% for SceneMiner;
* 87% for the metadata ranker;
* 26% for val150 as a whole.

**Under a stratum-preserving null:**
* no ranker's top 25 is significantly harder than random scenes with the same composition;
* SceneMiner VRU recall: −0.033, p = 0.108.

**Within day-dry scenes:** SceneMiner −0.016 mAP (p = 0.125).

**What survives.** SceneMiner − SigLIP2 on VRU recall: −0.099, p = 0.011 under the stratum null.

### E4, val150 degradation flags (`e4_val150_degradation_flags.py`)

**Rule:** any of the four zero-shot Qwen2.5-VL-3B CAM_FRONT yes/no log-odds > 0.

**Counts:**
* 42 degraded, 108 clean;
* 38 of the 42 are night or rain;
* only 4 of the 111 day-dry scenes are flagged.

**Means** reproduce to 4 decimals.

**Correction.** The main-paper mAOE q of 0.489 is wrong:
* BH over the published p-values gives 0.317, as in the supplementary BH table;
* with 10,000 permutations it is 0.338.

### E5, scale of s_head and s_abst (`e5_scale.py`, `sceneminer_poststd.py`)

**Scale on training queries × 906 training scenes:**

| term | mean | sd (pooled) | sd (within query) |
|---|---|---|---|
| s_head | −0.97 | 1.08 | 1.03 |
| s_abst | −0.46 | 1.11 | 0.71 |

**Decomposition check:** exact.

**Post-standardisation** per query on the training scenes moves no benchmark by more than 0.005
(BDD100K 0.614 → 0.619).

## Repository (Stage 1 → Stage 2)

### Added Stage-1 code

**Multi-camera path:**
* `scripts/train.py` and `train_wacv.py`, with configs v2 → v4 → v8, scrubbed and without the
  unused `refprog` base;
* feature export `export_filter_features_{scaled,overnight}.py`;
* targets `extract_nuprompt_tags.py`, `enrich_tags_from_nuscenes.py`, `compute_gt_risk_scores.py`;
* metadata `build_deterministic_gt.py`;
* Waymo dump `waymo_extract_perception.py`;
* `src/scene_mining/{data,datasets}`. These modules had been excluded by an unanchored
  `.gitignore` rule; the rule is now anchored to the repository root.

**Single-camera path:**
* `stage1_vlm_token.py`, extracted from `eval_cvpr_suites.py`;
* `stage1_build_prior.py`;
* `stage1_merge_shards.py`;
* `stage1_check_cache.py`.

### Checks against the existing cache

| signal | result |
|---|---|
| Prior | exact |
| Shard merges | exact |
| SigLIP2 | cosine 1.00000 |
| Margins | same sign, within bf16 precision (max 0.25 log-odds) |
| VLM token | cosine ≥ 0.9997 on CPU |

### Entry points, environments and benchmark files

**Entry points** (`pipeline/`, every script and flag checked):
* `stage1_multicam.sh`;
* `stage1_single_camera.sh`;
* `stage2_train.sh`;
* `eval_paper.sh`.

**Environments:**
* `requirements.txt` and `environment.yml` (exact versions);
* `environment_bevformer.yml`.

**Benchmark files** (`benchmark_cvpr/`):
* BDD100K-1000 query definitions and image list;
* holdout image list;
* Waymo frame identifiers.

### Device switch

`SM_DEVICE` in `scripts/paths.py`; 46 scripts now read it.

### Anonymisation

**Scrubbed:**
* cluster paths in configs and scripts;
* author name, e-mail and cluster paths in five earlier milestone notes.

**Checks:** `check_anonymity.py` passes on the paper and on the rebuilt code bundle (319 files).
