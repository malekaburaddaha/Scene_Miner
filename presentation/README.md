# Defense slides

`build_deck.js` builds `SceneMiner_PhD_Defense.pptx` (35 slides, native charts and tables,
speaker notes) with pptxgenjs 3.12.

* **Inputs:**
  * `deck_data.json`, parsed from the generated paper tables (`paper/tab_*.tex`);
  * `assets/`, PNG renders of `paper/figures/*` plus `fig1_main.jpg`.
* **Run:** `NODE_PATH=<node_modules> SKILL_DIR=<pptx skill> node build_deck.js [out.pptx]`
* **Not committed:** the deck and the assets, because they contain dataset frames.
