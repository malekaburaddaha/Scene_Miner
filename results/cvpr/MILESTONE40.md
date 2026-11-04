# Milestone 40 — Reference audit (2026-11-04)

**Scope.** All 35 entries of `paper/main.bib` were checked again from scratch against
live sources. The Milestone 20/35 caches were not reused.

**Tool.** `scripts/audit_references.py`.
* Raw responses: `paper/bibcheck/m40_raw.json`.
* Per-field comparison: `paper/bibcheck/m40_audit.json`.
* Venue pages for entries without a DOI: `paper/bibcheck/m40_venue_checks.json`.

## Sources

| source | entries |
|---|---|
| Crossref by DOI | 25 entries with a DOI (title, authors, container, year, volume, issue, pages) |
| arXiv API | every entry with an arXiv id (title, authors, comment / journal-ref) |
| PMLR | CLIP (v139, 8748–8763), BLIP-2 (v202, 19730–19742), BatchNorm (v37, 448–456) |
| CVF Open Access | RefAV (CVPR 2026), CARIM (ICCV 2025), ReferGPT (CVPRW 2025): metadata, and abstract or full text for the context check |
| ICLR pages | 2023 accepted list (CRB); 2018 poster 298 (Mu & Viswanath); 2017 conference poster list via the Internet Archive (Shazeer et al., poster C13) |
| Not usable | DBLP and OpenReview's v1 API answer automated clients with a bot challenge; Semantic Scholar rate-limited (429) |

## Result: every reference is real

All 35 resolve to the cited work.
* **Titles:** match the publisher record (similarity ≥ 0.993; the only differences are
  straight vs. curly apostrophes).
* **Authors:** every author list matches in full and in order, with two explained
  exceptions:
  * Mu et al.: arXiv adds Suma Bhat, but the ICLR 2018 version, which is the one cited,
    lists two authors.
  * BEV-TSR: Crossref swaps one name ("Fan JingChen"); arXiv gives "Jingchen Fan".
* **Years and pages:** every year and page range on a DOI entry matches Crossref.

## Corrections made

| entry | was | now | evidence |
|---|---|---|---|
| RefAV | arXiv preprint 2025 | CVPR 2026, pp. 21537–21548 (key `davidson2026refav`) | CVF Open Access page |
| CRB (Luo et al.) | arXiv preprint 2023 | ICLR 2023 | ICLR 2023 accepted list; arXiv comment "To appear in ICLR 2023" |
| CLIP, BLIP-2, BatchNorm | ICML, no pages | PMLR volume and pages added | PMLR pages |
| Geirhos, SigLIP 2, DriveLM, LingoQA | non-ASCII names were dropped from the typeset references ("Jrn-Henrik", "Hnaff", "Beiwenger", "Hnermann") | LaTeX accent commands (`texify` in `build_bib.py`) | rendered reference pages |

## Context check: does each cited work support its sentence?

All 35 citation contexts were read against the abstract, or the full text where the
abstract was not enough. Four sentences were reworded:

* **Active learning.** The text said both works select scenes a 3D detector is uncertain
  about. CRB argues that uncertainty- and diversity-based policies fail for 3D detection,
  and uses label, feature and geometric criteria. The text now cites Feng et al. for
  uncertainty and CRB for its own criteria.
* **Talk2Car.** It was listed as driving VQA, but it is an object-referral (command
  grounding) dataset. The text now says Talk2Car grounds commands while nuScenes-QA,
  DriveLM and LingoQA answer questions.
* **Anisotropy.** The text described "contrastive text embeddings" and credited mean
  removal to both papers. It now credits anisotropy of learned text representations to
  Ethayarajh, and mean removal to Mu & Viswanath (who study word, not contrastive,
  embeddings). SceneMiner's own observation on SigLIP2 is stated as ours.
* **BLIP-2.** The text said BLIP-2 and Qwen2.5-VL both provide the yes/no margins. In
  SceneMiner, BLIP-2's Q-Former provides the tags and Qwen2.5-VL the margins; the text
  now says so.

**Verified unchanged:**
* RefAV: RefProg has an LLM write programs over 3D tracks (paper body).
* ReferGPT: MLLM 3D-aware captions with CLIP-based matching (abstract).
* CARIM: VLM captions with inclusive text matching.
* nuPrompt: object-centric language prompts.
* Talk2BEV: language-enhanced BEV maps.
* ACDC: adverse images with normal-condition correspondences.
* WoodScape: soiling labels.
* Foggy Cityscapes, Dark Zurich, CODA, Fishyscapes: as described.
* Shazeer et al. (expert-collapse claim), Geirhos et al., batch normalisation: as cited.

**Still arXiv-only, cited correctly as preprints:** Qwen2.5-VL and SigLIP 2 (no journal
reference or venue comment on arXiv).

## Checks after the changes

* **Compile:** 0 undefined references and 0 overfull boxes; the main text still ends on
  page 8.
* **`verify_paper_numbers.py`:** passes.
* **`check_anonymity.py`:** passes on the paper and on the Overleaf zip.
* **Overleaf package:** rebuilt and test-compiled.
