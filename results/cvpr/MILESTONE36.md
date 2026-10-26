# Milestone 36 — Code scrub and the anonymised supplementary code bundle

## 1. Scrub

### Hard-coded locations

New module: `scripts/paths.py`. Every path in it has an environment override.

| was | now |
|---|---|
| `/nfs/turbo/…/datasets/external_adverse[/…]` (17 uses) | `EXT` = `$SCENEMINER_DATA_ROOT/external_adverse` (default `data`) |
| `/nfs/turbo/…/hf_cache/Qwen2.5-VL-3B-Instruct` (17) | `QWEN_VL` (default `Qwen/Qwen2.5-VL-3B-Instruct`) |
| `/nfs/turbo/…/hf_cache/blip2-opt-2.7b` | `BLIP2` (default `Salesforce/blip2-opt-2.7b`) |
| the So400m snapshot path | `SIGLIP2_SO400M` (default `google/siglip2-so400m-patch14-384`) |
| `results/day1_gate/malek_repaired.json` (15 scripts) | `HUMAN_R1` = `benchmark_cvpr/human_rater1_600.json` |
| conda interpreter in `launch_seeds.sh`, `run_m27_evals.sh`, `run_rater2_ui.sh`, `run_seed_sd.sh` | `${PY:-python}` |
| interpreter in `orchestrate_overnight.py` | `$PY`, else `sys.executable` |

**Scope.** 44 tracked files changed. All compile, and all shell scripts parse
(`bash -n`).

**Grep result.** No tracked file under `scripts/` or `src/` now contains `/nfs/`,
`/home/`, `umd`, `srawa`, `umich`, `greatlakes`, `arc-ts`, `malek`, `abdmalek` or
`aburaddaha`. The one exception is `check_anonymity.py`, whose token list names them.

**Removed from the repo:** `benchmark_cvpr/human_annotations_rater1_600.json`. It was
tracked since Milestone 1 and carried `"rater": "malek"` in all 600 entries. Its labels
and flags are identical to the anonymised `benchmark_cvpr/human_rater1_600.json`
(checked).

* It remains in git history. The repository is private, and history was not rewritten.
* MASTER_REPORT now points to the anonymised file.

### Refactor check

Two evaluations were rerun through the refactored code (new data root and new rater-1
path):

* `incorporate_rater2.py`;
* `eval_per_query_vlm_reranker.py --part eval`.

Their JSON outputs are **identical** to the committed results, ignoring timing fields.

### Local note

On this cluster, `data/external_adverse` is a git-ignored symlink to the dataset
directory. Set `QWEN_VL` to the local checkpoint, or the default hub id will download
it.

## 2. `paper/supplementary_code.zip` (`scripts/build_supplementary_code.py`)

### What it contains

* **Size:** 254 files, **1.15 MB**, so it is committed (under 5 MB).
* **Included:** tracked `src/`, `scripts/`, `benchmark_cvpr/` (splits, queries,
  holdout definition, the 160-scene subset, both anonymised rater files) and
  `results/cvpr/*.json`.
* **README:** an `ANONYMOUS_README.md` with contents, environment variables, the
  data-licence note and reproduction commands.
* **Excluded:**
  * `check_anonymity.py`, because its token list names the affiliation;
  * the builder itself;
  * logs and non-JSON results.
* **Determinism:** member timestamps are fixed, so the archive depends only on the files.

### Scan

`check_anonymity.py` gained a zip mode. It runs automatically when
`paper/supplementary_code.zip` exists, or with `--zip <path>`. It scans:

* every member's name and contents;
* the archive comment;
* both paper PDFs, the figure PDFs and images, and the TeX sources, as before.

**Result:**

* **254 members: no leak.** ANONYMITY CHECK PASSED.
* **The detector works:** a planted test archive with `/nfs/turbo/x` and the surname is
  flagged.
* **Extra greps**, for identifiers outside the token list (another cluster user name,
  `gmail`, `github`): no hits.
* **One functional mention of an LLM vendor remains:** `src/scene_mining/refprog/synthesizer.py`
  is an older RefAV-style synthesizer with a selectable LLM backend. It is code, not
  authorship attribution. The paper's RefAV-style row uses the local Qwen compiler.
