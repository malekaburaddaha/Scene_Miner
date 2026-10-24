# Milestone 27 — Cached scene index, conditional context gate, portable headline

Files:

* `src/scene_mining/filter/fast_index.py`
* `src/scene_mining/filter/sceneminer_v2_6.py`
* `scripts/explore_conditional_gate_train.py`
* `scripts/train_sceneminer_v26.py`
* `scripts/verify_fast_index.py`
* `scripts/time_sceneminer_latency.py`

Results:

* `milestone27_conditional_gate_train.json`
* `milestone27_v26.json`
* `milestone27_fast_index{,_fp64}.json`
* `milestone27_latency.json`
* `milestone27_semantic_edge_cases.json`
* `milestone27_compositional_synergy.json`
* `milestone27_rater_sensitivity.json`
* `milestone27_val150_retrieval_v26.json`
* `milestone27_regimes.json`

## Summary

| item | target (task) | result |
|---|---|---|
| online latency per model, 1508 scenes | < 0.50 ms | **3.51 ms** = 1.24 ms ranking (CUDA graph) + 2.27 ms query preparation; was 12.39 ms. **Target not met** |
| identity with the reference forward pass | max \|ΔAP\| < 1e-6 | **0.0 in float64 on every benchmark**; float32: 0.0 except the suite (≤ 5.5e-5, near-tie reordering) |
| BDD100K after the gate fix | ≈ 0.628 (above SigLIP2 0.622) | **0.614** — not reached |
| 150-query suite after the gate fix | ≈ 0.685 kept | **0.694** |
| portable (uncontaminated) human GT | 0.768 | **0.762**, still above every other method (next: 0.735) |

## 1. Cached scene index

`SceneMinerV25/26.forward` expands the query to one row per scene. As a result, every
query-only quantity is recomputed 1508 times:

* the concept-bank posteriors (several calls per forward);
* channel weights and routing;
* the query adapter, FiLM γ/β and the residual gate.

The six scene adapters also re-run on every query, although they do not depend on it.

`fast_index.py` restructures this work:

* **Index build, once (5 ms for 5 seeds):**
  * `precompute_scene_index` stores the six adapted scene tokens plus the type embedding (1508 × 6 × 256);
  * it also stores the scene-side inputs of the base parts and the standardised metadata.
* **`prepare_query`, once per query:** everything that depends only on the query.
* **`rank`, the per-scene work:**
  * base parts and PartNorm;
  * FiLM on the cached tokens;
  * the decoder, the scorer, ResNorm and the abstention mix.
* **Decoder folding:**
  * The decoder's attention is computed with the key and value projections moved to the query side:
    * q·(W_k z + b_k) = (W_kᵀq)·z + q·b_k
    * Σ_t a_t (W_v z_t + b_v) = W_v (Σ_t a_t z_t) + b_v
  * The 9048 scene tokens are therefore only dotted with 256-d vectors.
  * Layer 1's self-attention, whose target is a single query token, is computed once.
* **`GraphRanker`:** replays `rank` as a CUDA graph, which removes per-kernel launch overhead.

### Identity (`verify_fast_index.py`)

The check runs every benchmark query through both the reference `forward`/`base_score`
and the cached path, on all 5 seeds. That covers 4 human GT, 150 suite and 40 Stratum 5
(monocular) queries plus 12 BDD100K portable queries.

| benchmark | float64 max \|Δscore\| | float64 max \|ΔAP\| | float32 max \|Δscore\| | float32 max \|ΔAP\| |
|---|---|---|---|---|
| human GT (full) | 3.2e-15 | **0** | 1.4e-6 | 0 |
| 150-query suite (full) | 1.7e-7 | **0** | 1.4e-5 | 5.5e-5 |
| Stratum 5 (monocular) | 2.4e-7 | **0** | 5.9e-6 | 0 |
| BDD100K (portable) | 1.4e-14 | **0** | 3.3e-6 | 0 |

* **Float64:** the identity is exact.
* **Float32:** scores differ only by summation order. On the suite that reorders near-tied
  scenes, which moves some per-query APs by up to 5.5e-5.
* **Head skip:** skipping the learned head when γ_deg < 1e-6 leaves every AP unchanged
  (31/192 queries skipped). Larger thresholds change APs (1e-4: up to 2.5e-4), so they are
  not used.
* **Graph and eager paths:** `rank` equals the eager fast path exactly, and the CUDA-graph
  replay equals `rank` exactly.

### Latency (one A40, 1508 scenes, query embedding cached)

| | one model | 5-seed ensemble |
|---|---|---|
| reference `forward` | 12.39 ms | 62.1 ms |
| cached index, eager | 4.70 ms (prepare 2.27 + rank 2.41) | 23.4 ms |
| cached index, prepare + CUDA-graph `rank` | **3.51 ms** (2.27 + 1.24) | 16.1 ms (rank alone 5.8 ms) |
| portable path: reference `base_score` → cached | 7.02 ms → **2.47 ms** | |

**Why it is not < 0.5 ms**

* **What remains per scene:**
  * the two decoder feed-forward blocks (256 → 1024 → 256);
  * the scorer MLP (1024 → 256 → 1) over 1508 rows;
  * one per-scene self-attention projection in layer 2.
* **Lower precision:** TF32 or bf16 arithmetic would cut this further, but would break the
  exact identity, so I did not use it.
* **Query preparation (2.28 ms):**
  * It is independent of corpus size.
  * It is mostly the v2.6 concept gate: a host sync plus a small QR per query.
  * It is query-side work of the same kind as text encoding (4.5 ms for every text tower),
    which no row of Table 2 counts.
  * The paper reports both numbers separately.

**Offline Stage 1 per scene** (`milestone27_latency.json`): see §4.

## 2. Conditional context removal (v2.6)

**Problem.** v2.5 removes half of the 7-class context subspace from every query before computing γ_deg.

**Variants** (`explore_conditional_gate_train.py`). The protocol, the 60 synthetic
conjunctions and the selection rule are those of Milestone 22, fixed before running, on
the 252 training phrasings:

* `uncond`: the v2.5 gate, unchanged.
* `cond_all`: remove the whole subspace, but only if some context class beats the best
  generic (OOV) class score.
* `cond_sub`: remove only the span of the active context classes.

**Result:** `cond_sub`, λ = 0.5, τ = 0.02 is selected.

| variant (τ chosen by rule) | pos γ > 0.9 | compound p10 | max γ, negatives | max γ, synthetic conj. |
|---|---|---|---|---|
| uncond (v2.5), λ 0.5, τ 0.02 | 0.99 | 1.000 | 0.152 | 0.007 |
| cond_all, λ 0.5, τ 0.02 | 0.99 | 1.000 | 0.152 | 0.007 |
| **cond_sub, λ 0.5, τ 0.02** | **0.99** | 1.000 | **0.080** | 0.021 |

* `cond_all` equals `uncond`, because every training phrasing activates some context class.
* `cond_sub` halves the worst false positive among negatives.
* v2.6 was retrained for 5 seeds with this gate; everything else is v2.5.

### Evaluation, done once with the selected gate

| | v2.5 | **v2.6** |
|---|---|---|
| human GT zero-leakage / calibrated | 0.785 / 0.826 | **0.794 / 0.829** |
| BDD100K (raw queries) | 0.616 | 0.614 (in-vocab 0.698, OOV 0.572) |
| suite mAP / skew | 0.685 / +2.2 | **0.694** / +2.6 |
| Stratum 5 corpus atomic / compositional | 0.332 / 0.297 | 0.330 / 0.286 |
| Stratum 5 val150 atomic / compositional | 0.392 / 0.418 | 0.390 / 0.416 |

**The BDD100K target was not reached.**

* The motivating query, "snow covering the road", still has γ_deg = 0.73 (v2.5: 0.71).
* "Snow" activates a weather-context class, so its direction is still removed. The
  conditional rule does not treat it as context-free.
* I did not tune the rule further: the only evidence for doing so would be BDD100K itself.

v2.6 is the final model because the pre-declared training-text rule selected its gate,
not because of these evaluation numbers. Its in-domain gains (human GT +0.009, suite
+0.009) include a retraining effect; the external numbers are level.

## 3. Portable, uncontaminated SceneMiner

**Inputs.** `Rankers.sceneminer_portable` reads only frozen web-trained signals:

* the SigLIP2 image embedding;
* the Qwen2.5-VL yes/no margins;
* recorded metadata.

**Excluded.** It reads no BEVFormer embedding, no risk scalar and no tags:

* the 20 tag columns of the prior are zeroed;
* the meta channel is off;
* the residual branch is not used.

This is the same path as on BDD100K and val150.

| | portable | full (multi-camera) |
|---|---|---|
| human GT (550) | **0.762** (every other method ≤ 0.735) | 0.794 / 0.829 calibrated |
| BDD100K | 0.614 | 0.614 (identical path) |
| Stratum 5 corpus / val150 | 0.311 / 0.394 (all queries) | 0.308 / 0.394 |
| 150-query suite | 0.654 | 0.694 |

The portable model still leads every other method on human-labelled degradation without
reading any signal supervised on corpus scenes. The full model's BEV and tag inputs add
+0.032 on human GT and +0.040 on the suite.

## 4. Offline Stage-1 cost per scene (`time_sceneminer_latency.py`, A40, batch 1)

| step | time |
|---|---|
| multi-camera Stage-1 model, one 6-camera keyframe (SigLIP2 patches, BEVFormer, motion head, Q-Former, LoRA-Qwen, heads; bf16) | 114–117 ms |
| Qwen2.5-VL-3B, the four yes/no margins of one image (image re-encoded per question) | 0.63–0.93 s |
| SigLIP2 ViT-B/16 image embedding, one image incl. CPU preprocessing | 7–92 ms (CPU-bound, varies with node load) |
| Qwen2.5-VL-3B caption, one image (caption baselines only; Milestone 23) | 3.6 s |

All Stage-1 work happens once per scene, at indexing time. The VLM margins dominate it:
about 1 s per image.
