# Milestone 13 — repository finalisation and verification

## Verification results

| check | result |
|---|---|
| commit authorship | **single author**: the repository owner (single author) |
| AI attribution in commit messages | **none** |
| AI attribution in tracked files | **none** |
| forbidden binaries (`.pth .pt .npz .ckpt .safetensors .bin .pkl .h5`) | **none tracked** |
| files over 5 MB | **none** (largest: `paper/main_cvpr.pdf`, 888 KB) |
| repository size | 31 MB total |
| working tree | clean |
| paper compiles | 6 pages, no errors, no overfull boxes |

### One deliberate exception, documented

`src/scene_mining/refprog/synthesizer.py` contains the strings `claude` and
`anthropic`. These are **not attribution**: the module is the RefProg
program-synthesis backend selector, where `backend="claude"` and
`backend="openai"` name which LLM API to call, following the RefAV paper's
recommended setup. The file predates this milestone series (first committed in
Milestone 1) and is functional code, so it is left intact rather than edited to
satisfy a text search.

## Documentation updated

* **`README.md`** — §6 gains the Milestone 10–12 reproduction commands (v2.2
  training across 5 GPUs one job each, the three evaluations, the blend
  diagnostic, calibration, CAM_FRONT analysis, figures, paper). §7 is rewritten
  with the human-GT table, the BDD100K transfer table, the CAM_FRONT premise
  table, and an explicit "known costs" line. §8 adds `sceneminer_v2_2.py`,
  `paper/main_cvpr.tex` and `paper/figures/`.
* **`results/cvpr/MASTER_REPORT.md`** — new **Part III** covering Milestones
  10–12, and a rewritten closing section listing the four defensible claims and
  what still does not survive.

## The four claims this repository now supports

1. **The premise holds.** Front-camera degradation costs the 3D detector 0.0906
   front-sector mAP and 7.3 points of false-positive rate on uncontaminated
   data, surviving Benjamini–Hochberg correction.
2. **Two structural defects.** A benchmark whose relevance is a predicate over
   cached model outputs is circular (lookup = 1.000 mAP); a detector trained on
   the scenes it certifies inflates per-scene mAP by +0.3349 (+74 %).
3. **A general mechanism result.** A convex router must standardise what it
   routes — demonstrated by a blend scoring below both inputs at a 28.1× scale
   gap.
4. **A measured trade-off.** De-shortcutting a VLM teacher costs 0.029 mAP in
   domain and buys 0.176 mAP externally.

## What is explicitly not claimed

* SceneMiner does **not** beat zero-shot SigLIP2 on BDD100K (0.5952 vs 0.6218).
* De-shortcutting is **not** free: glare VLM false-negative recovery falls from
  7/20 to 1/20.
* Domain-skew differences of a few points are **not** resolvable at n = 150
  queries (bootstrap CI widths 9.7–12.5 points).
* The text mean-centring design choice was informed by BDD100K measurements;
  this is disclosed in both the paper and the master report.

## Remaining action for the user

The repository has **14 commits** and is ready to push, but the push could not
be completed from the compute node: SSH ports 22 and 443 are firewalled and
HTTPS has no stored credentials. From a login node:

```bash
cd <repository root>
git push -u origin main
```

Confirm the GitHub repository is set to **private** before or immediately after
the first push.
