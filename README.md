# SceneMiner

A text-queryable index over driving logs for the **sensor-degradation and
macro-environmental conditions that degrade perception** — water on the lens,
headlight bloom on wet asphalt, loss of contrast at night — together with the
**collapse diagnostic protocol** that validates the fusion behind it.

All scene signals are cached at build time, so serving a query costs one text
encoding plus a single pass of a small head over the cached corpus.

---

## 1. System architecture

```
 STAGE 1 — build time, once per corpus
 ────────────────────────────────────────────────────────────────────────
  6 camera views ─► SigLIP2 (frozen) ─┐
                                       ├─► BLIP-2 Q-Former ─► Qwen2.5-VL ─► pooled
  ego pose ──────► BEVFormer 50×50 ───┤        (32 queries)      (3.77 B)      │
                   MotionHead ─────────┘                                       │
                                            ┌──────────────┬───────────┬───────┘
                                            ▼              ▼           ▼
                                     EmbeddingHead   TaggingHead   RiskHead
                                        e: 256-d       t: 20-d      r: 1-d

  thumbnail ─► Qwen2.5-VL + LoRA r=64 ──────────────────────────► v: 2048-d
  thumbnail ─► SigLIP2 image tower ─────────────────────────────► s: 768-d   [v2]
  thumbnail ─► Qwen2.5-VL binary Yes/No ────────────────────────► m: 4-d     [v2]
                 (droplets, glare, wet_road, low_contrast — log-prob margins)

 ╔══════════ CACHED SCENE BANK: 1,508 scenes × {e, t, r, v, s, m} ══════════╗

 STAGE 2 — per query, 8.65 ms over 1,508 scenes (1× A40)
 ────────────────────────────────────────────────────────────────────────
  query text ─► frozen text tower ─► q
                                      │
      ┌───────────────────────────────┴────────────────────────────┐
      ▼                                                            ▼
  s_base(q,x)  =  Σ_k w_k(q) · base_k(q,x)              s_res(q,x) = fusion head
     w(q) = softmax(W_route q̂ / τ_r)                     6 tokens → FiLM(q)
     base = [SigLIP2 cos, VLM margin, tag/meta prior]     → cross-attn → bil+MLP
      └───────────────────────────────┬────────────────────────────┘
                                      ▼
              s(q,x) = s_base + tanh(gate(q)) · s_res      ← gate init 0
                                      ▼
            z-score → abstain (z < 2.5) → MMR (λ=0.7) → top-k
```

**Identity-preserving growth (§3.4).** `W_route`, the channel selectors and the
gate head are all zero-initialised, so at step 0 routing is exactly uniform and
the residual is inert: `s ≡ s_base`, verified to 0.00e+00. Every departure from
the zero-shot prior is something training paid for.

**Scalar channels are standardised with fixed dataset statistics, never
`LayerNorm`.** A `LayerNorm` over a width-1 axis returns
`(x − x̄)/√(0+ε) ≡ 0` for every input and deletes the channel.
`--force-layernorm1d` restores that defect on purpose, to regenerate the
pre-repair baseline (`--legacy-scalar` does **not** — it only swaps (μ,σ)).

---

## 2. Collapse diagnostic protocol

For a mean-centred feature matrix `X` with singular values `σ₁ ≥ … ≥ σ_d` and
`pᵢ = σᵢ / Σⱼ σⱼ`, the spectral effective rank is

```
erank(X) = exp( − Σᵢ pᵢ log pᵢ )
```

This is uninterpretable alone, so it is read against the **label-entropy null**
`erank(L)` for the supervision the representation is trained to express, and the
ratio `erank(X)/erank(L)` reported. A ratio near or above 1 means the
representation uses as much of the space as its labels can distinguish.

> **Ceiling caveat.** A multi-hot matrix over *k* atomic binary attributes has
> rank ≤ *k*, so `erank(L) ≤ k` **by arithmetic**. Both constructions are
> reported: atomic (12 cols → 8.04) and query-predicate (233 cols → 22.39 at
> matrix rank 38). §4.3's 47.6 needs ≥48 *linearly independent* columns and is
> unreachable with this attribute vocabulary.

The protocol also reports:
- **routing mass** per modality, read from the decoder's real cross-attention
  (`need_weights=True`, averaged over heads and queries) — not a cosine proxy;
- **per-token distinct rows**, which catch a dead channel (a dead risk token
  shows 1 distinct row of 1,508 with per-dim std exactly 0.0000);
- **per-attribute probe gradients** `‖∂L/∂θ‖`, with an explicit bf16 floor.

---

## 3. Dataset, splits, and query suite

| | scenes | composition |
|---|---|---|
| corpus | 1,508 | 700 nuScenes + 808 Waymo 4-frame chunks (202 segments) |
| **train** | **906** | 422 nuScenes + 484 Waymo from 121 segments |
| **eval** | **602** | 278 nuScenes + 324 Waymo from the remaining 81 segments |

Waymo is split at **segment** granularity so no 20 s source segment contributes
chunks to both sides. Verified: 0 scene overlap, 0 segment overlap.

**Query suite** (`benchmark_cvpr/queries_390.json`) — 240 train / 150 eval:

| eval stratum | n | mean relevant | prevalence |
|---|---|---|---|
| sensor-degradation | 62 | 54.2 | 0.0900 |
| macro-environmental | 58 | 154.0 | 0.2558 |
| compositional (held out) | 18 | 24.3 | 0.0404 |
| negated | 12 | 60.6 | 0.1006 |

Train/eval text strings are 100 % disjoint, and **every compositional predicate
is reserved from the training pool** (verified 0 overlap) so the "never
supervised" claim holds.

**External corpus:** 1,000 stratified BDD100K images with native human
`weather`/`timeofday` labels (night 493, rainy 94, snowy 82).

---

## 4. Human annotation protocol

`scripts/annotate_ui.py` serves a zero-dependency local UI (stdlib
`http.server`). Both raters label the **same frozen 600-scene sample**, drawn
from the eval split and stratified over positives and borderline cases
(|VLM margin| < 0.5).

Keys: `1` droplets · `2` glare · `3` wet_road · `4` low_contrast ·
`Enter` save+next · `b` back · `u` unsure · `s` skip. **Autosaves after every
scene**, fully resumable.

Rater 1 completed 600 scenes: 530 `flag=None`, 20 `skip` (all-False),
50 `unsure`. Two scoring modes are reported — **Mode A** excludes
`skip`/`unsure` (530 scenes); **Mode B** keeps all-False `skip` rows as genuine
clear negatives (550). Results differ by < 0.006.

Against those labels the VLM's precision is **0.50–0.68** (κ 0.51–0.77), which
is what collapses the flat lookup from a tautological 1.000 under
predicate-relevance to 0.539 under human relevance.

---

## 5. Installation

```bash
# retrieval, diagnostics, v2/v2.1
conda activate /nfs/turbo/umd-srawa/conda_envs/envs/scene_mining   # torch 2.4.1, transformers 5.2
# BEVFormer 3D evaluation (needs the older mm-stack)
conda activate /nfs/turbo/umd-srawa/conda_envs/envs/bevformer      # torch 1.10, mmdet3d 0.17.1
export PYTHONPATH=src:$PYTHONPATH
```

> GPUs on this cluster are **`Exclusive_Process`: one job per GPU.** Launching
> more than 4 concurrent jobs kills the extras with "CUDA-capable device(s)
> is/are busy". Use `scripts/orchestrate_overnight.py`, which holds one job per
> GPU and retries once. Detach long jobs with `setsid`, not bare `nohup`.

---

## 6. Reproducing every table

```bash
# ── build-time assets ───────────────────────────────────────────────
python scripts/build_split_906_602.py                      # frozen 906/602 split
python scripts/build_query_suite_390.py                    # 240 train / 150 eval
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/extract_vlm_logits.py \
    --shard $i --nshards 4 --out work_dirs/unified_filter/logits_shard$i.json & done
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/extract_lowcontrast_strict.py \
    --shard $i --nshards 4 --out work_dirs/unified_filter/lc_shard$i.json & done
python scripts/cache_v2_signals.py                         # + SigLIP2 + margins + prior

# ── Table 1: collapse diagnostic, before vs after ───────────────────
python scripts/train_filter_cvpr.py --work-dir work_dirs/cvpr_ablation/Atrue_s0 \
    --supervision caption --legacy-scoring --force-layernorm1d --freeze-visual \
    --unconditional-negs --w-vicreg 0 --tau 1.0 --steps 3000 \
    --split-file benchmark_cvpr/split_906_602.json --seed 0
python scripts/run_diagnostic_protocol.py --checkpoint work_dirs/cvpr_ablation/Atrue_s0/final.pt \
    --legacy-scoring --force-layernorm1d --lora-r 1 --tau 1.0 --out results/cvpr/diag_before.json
python scripts/label_entropy_null.py                       # erank(L), both constructions
python scripts/blip2_diagnostic_transfer.py                # protocol on a second architecture

# ── Table 2: de-confounded ablation, 5 seeds ────────────────────────
python scripts/orchestrate_overnight.py                    # 41 runs, 4 GPU slots
python scripts/eval_cvpr_suites.py --mode ablation --out results/cvpr/table2.json
python scripts/eval_cvpr_suites.py --mode loao     --out results/cvpr/loao.json

# ── Table 3 + domain calibration on the 150-query suite ─────────────
python scripts/eval_attribute_index.py --output results/cvpr/attribute_index.json
python scripts/eval_domain_calibrated.py --out results/cvpr/milestone7_tables2_3.json

# ── SceneMiner v2 / v2.1 ────────────────────────────────────────────
for s in 0 1 2 3 4; do python scripts/train_sceneminer_v2.py  \
    --work-dir work_dirs/sceneminer_v2/s$s  --seed $s --steps 3000; done
for s in 0 1 2 3 4; do python scripts/train_sceneminer_v21.py \
    --work-dir work_dirs/sceneminer_v21/s$s --seed $s --steps 3000; done
python scripts/eval_v2_human_gt.py                         # v2 vs baselines, human GT
python scripts/eval_v21_human_gt.py                        # + 5-fold human calibration
python scripts/eval_human_gt_full.py                       # Modes A and B, all rankers

# ── 3D gate (bevformer env) ─────────────────────────────────────────
python tools/test.py projects/configs/cvpr_gate/bevformer_base_fullval.py \
    ckpts/bevformer_base.pth --launcher pytorch --out val_results.pkl
python scripts/per_scene_official_metrics.py \
    --results-json <results_nusc.json> --out results/cvpr/per_scene.json
python scripts/run_3d_gate.py --per-scene results/cvpr/per_scene.json \
    --out results/cvpr/milestone1_bevformer_3d.json
python scripts/val150_3d_study.py                          # uncontaminated val holdout

# ── SceneMiner v2.2 (Milestone 10) ──────────────────────────────────────
# 5 seeds, one GPU each (GPUs are Exclusive_Process: one job per GPU)
for s in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$s python scripts/train_sceneminer_v22.py \
      --work-dir work_dirs/sceneminer_v22/s$s --seed $s --steps 3000
done
python scripts/eval_v22_human_gt.py      # 550 human scenes, zero-leak + calibrated
python scripts/eval_v22_query_suite.py   # 150-query suite + mechanism ablations
python scripts/diag_bdd_blend.py         # the scale-incommensurability diagnostic

# ── Milestone 11: calibration, targeted 3D error, significance ──────────
python scripts/eval_domain_calibrated_v2.py   # (q,alpha) fit on train queries
python scripts/val150_camfront_errors.py      # CAM_FRONT degradation vs error

# ── Milestone 12: figures and paper ─────────────────────────────────────
python scripts/make_figures.py
cd paper && pdflatex main_cvpr.tex && pdflatex main_cvpr.tex

# ── external dataset ────────────────────────────────────────────────
python scripts/fetch_bdd100k.py                            # 1,000 stratified images
python scripts/eval_external_bdd100k.py

# ── human annotation (two raters, same frozen sample) ───────────────
python scripts/annotate_ui.py --rater <name> --port 8800
```

---

## 7. Headline results

**Against 550 human-labelled scenes** (macro-mAP over the four attributes):

| ranker | mAP |
|---|---|
| **SceneMiner v2.1 + 5-fold human calibration** | **0.7924** |
| SceneMiner v2.2 + 5-fold human calibration | 0.7656 |
| SceneMiner v2.1 (zero-leakage) | 0.7355 |
| logit VLM lookup | 0.7325 |
| SceneMiner v2.2 (monocular) | 0.7167 |
| SceneMiner v2.2 (zero-leakage) | 0.7065 |
| zero-shot SigLIP2 | 0.5688 |
| metadata prior | 0.5297 |

**External transfer, BDD100K** (1000 images, 12 queries, BDD's own human
weather/time-of-day labels, chance 0.2623):

| ranker | mAP | P@5 | NDCG@10 |
|---|---|---|---|
| zero-shot SigLIP2 | **0.6218** | **0.8500** | **0.8088** |
| **SceneMiner v2.2** (monocular) | 0.5952 | 0.7500 | 0.7892 |
| logit VLM lookup | 0.5871 | 0.7167 | 0.7054 |
| SceneMiner v2.1 (portable) | 0.4189 | 0.5500 | 0.5524 |

v2.2 lifts external transfer by **+0.1763 mAP** over v2.1 and beats both lookup
baselines, but **does not beat zero-shot SigLIP2** — stated plainly because it
is the number a reviewer will look for.

**The premise, tested correctly** (150 uncontaminated nuScenes val scenes,
CAM_FRONT degradation vs front-sector detection error):

| metric | degraded | clean | Δ | q (BH) |
|---|---|---|---|---|
| front-sector mAP | 0.3886 | 0.4792 | **−0.0906** | 0.015 |
| FP rate @ score>0.3 | 0.3503 | 0.2775 | **+0.0729** | 0.045 |
| mATE (m) | 0.8330 | 0.7553 | +0.0777 | 0.080 |
| recall > 30 m | 0.5312 | 0.6079 | −0.0767 | 0.080 |

Only 23.4 % of nuScenes GT lies in the CAM_FRONT FOV, which is why an earlier
360° scene average found nothing (< 0.07σ) and this targeted test finds
0.42–0.74σ.

**Known costs of v2.2**, not buried: −0.029 mAP in domain, and glare VLM
false-negative recovery falls from 7/20 to 1/20.

`results/cvpr/MASTER_REPORT.md` carries every table, including the claims that
**do not** reproduce and the open limitations.

---

## 8. Layout

```
src/scene_mining/      model, adapters, losses, diagnostic hooks
  filter/unified_filter.py     ModalityAdapter, FiLM, ScalarStandardize
  filter/lora_grounded_filter.py  v1 head, legacy/pre-repair switches
  filter/sceneminer_v2.py      v2 and v2.1 (query-conditioned routing)
  filter/sceneminer_v2_2.py    v2.2: PartNorm, entropy floor, modality masking
  adapters/qformer.py          BLIP-2 Q-Former adapter
scripts/               training, evaluation, baselines, diagnostics, annotation UI
benchmark_cvpr/        frozen split, 390-query suite, human annotations
results/cvpr/          MILESTONE1-13.md, MASTER_REPORT.md, all metric JSONs
paper/main_cvpr.tex    the CVPR manuscript (6 pp, compiles with pdflatex)
paper/figures/         fig1 architecture, fig2 qualitative, fig3 audit
```

Checkpoints, feature stores and datasets are untracked — see `.gitignore`.
