// SceneMiner Ph.D. defense deck. Numbers come from deck_data.json, which is parsed from the
// paper's generated tables (paper/tab_*.tex <- results/cvpr/*.json).
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const md = require("react-icons/md");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require(process.env.SKILL_DIR + "/scripts/apply_theme.js");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const A = (f) => path.join(__dirname, "assets", f);
const OUT = process.argv[2] || path.join(__dirname, "SceneMiner_PhD_Defense.pptx");

const THEME = {
  name: "SceneMiner Asphalt",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1C2128", lt1: "FFFFFF", dk2: "3B4656", lt2: "EEF1F4",
    accent1: "E8A200", accent2: "168A7E", accent3: "C4553B",
    accent4: "8C96A5", accent5: "5B7083", accent6: "E7C35A",
    hlink: "168A7E", folHlink: "5B7083",
  },
};
const HEX = THEME.colors;
const W = 13.333, H = 7.5, M = 0.6;

async function icon(name, color = "FFFFFF", size = 256) {
  const el = React.createElement(md[name], { color: "#" + color, size: String(size) });
  const svg = ReactDOMServer.renderToStaticMarkup(el);
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
async function dims(f) { const m = await sharp(A(f)).metadata(); return [m.width, m.height]; }

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title = "SceneMiner - Ph.D. dissertation defense";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  // ------------------------------------------------------------------ layouts
  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 2.0, w: W - 2 * M, h: 1.9, fontSize: 40, bold: true,
          color: C.background1, valign: "bottom", align: "left" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: M, y: 4.15, w: W - 2 * M, h: 1.2, fontSize: 20,
          color: C.accent1, valign: "top", align: "left" }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "SECTION_DARK",
    background: { color: C.text1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 2.9, w: W - 2 * M, h: 1.2, fontSize: 40, bold: true,
          color: C.background1, valign: "middle", align: "left" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: M, y: 4.1, w: W - 2 * M, h: 0.8, fontSize: 18,
          color: C.accent1, valign: "top", align: "left" }, text: "" } },
    ],
    slideNumber: { x: W - 1.1, y: H - 0.5, w: 0.6, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: C.background1 },
    margin: [0.5, 0.6, 0.7, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.85, fontSize: 30, bold: true,
          color: C.text1, valign: "middle", align: "left" }, text: "" } },
      { text: { text: "SceneMiner  ·  Ph.D. dissertation defense", options: { x: M, y: H - 0.45, w: 6, h: 0.3, fontSize: 10,
          color: C.accent4, margin: 0 } } },
    ],
    slideNumber: { x: W - 1.1, y: H - 0.45, w: 0.6, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
  });

  const content = (section, title) => {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    return s;
  };
  const section = (name, num, sub) => {
    pres.addSection({ title: name });
    const s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: name });
    s.addText(name, { placeholder: "title" });
    s.addText(sub, { placeholder: "body" });
    s.addText(String(num).padStart(2, "0"), { x: M, y: 1.6, w: 2.5, h: 1.2, fontSize: 60, bold: true, color: C.accent1,
      fontFace: "Cambria", isTextBox: true, margin: 0, objectName: "section-number" });
    return s;
  };
  const txt = (s, t, o) => s.addText(t, Object.assign({ isTextBox: true, fontSize: 16, color: C.text1, valign: "top", margin: 0.05 }, o));
  const bullets = (s, items, o) => s.addText(items.map((t, i) => ({ text: t, options: { bullet: { indent: 18 },
    breakLine: i < items.length - 1, paraSpaceAfter: 8 } })), Object.assign({ isTextBox: true, fontSize: 16, color: C.text1,
    valign: "top", margin: 0.05 }, o));
  const card = (s, x, y, w, h, fill, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: fill }, line: { color: fill }, objectName: name });
  const badge = async (s, x, y, d, ico, fill, name) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: name + "-circle" });
    s.addImage({ data: await icon(ico), x: x + d * 0.2, y: y + d * 0.2, w: d * 0.6, h: d * 0.6, objectName: name + "-icon" });
  };
  const numBadge = (s, x, y, d, n, fill, name) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: name });
    s.addText(String(n), { x, y, w: d, h: d, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle",
      isTextBox: true, margin: 0, objectName: name + "-n" });
  };
  const stat = (s, x, y, w, big, small, color, name, size = 40) => {
    s.addText(big, { x, y, w, h: 0.9, fontSize: size, bold: true, color, fontFace: "Cambria", isTextBox: true, margin: 0,
      valign: "bottom", objectName: name + "-value" });
    s.addText(small, { x, y: y + 0.95, w, h: 0.75, fontSize: 14, color: C.text2, isTextBox: true, margin: 0, valign: "top",
      objectName: name + "-label" });
  };
  const fitImage = async (s, f, x, y, mw, mh, name, alignTop = false) => {
    const [iw, ih] = await dims(f); const r = Math.min(mw / iw, mh / ih);
    const w = iw * r, h = ih * r;
    s.addImage({ path: A(f), x: x + (mw - w) / 2, y: alignTop ? y : y + (mh - h) / 2, w, h, objectName: name,
      altText: name });
  };
  const chartText = () => ({ catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
    dataLabelFontSize: 11, legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.dk2, titleFontFace: "+mn-lt",
    titleFontSize: 14, titleColor: HEX.dk1, valGridLine: { color: HEX.lt2, size: 0.75 }, catGridLine: { style: "none" } });
  // horizontal bars, one highlighted: two stacked series, zeros hidden by the number format
  const hbar = (s, labels, vals, hi, o) => {
    const base = vals.map((v, i) => (hi.includes(i) ? 0 : v)), top = vals.map((v, i) => (hi.includes(i) ? v : 0));
    s.addChart(pres.charts.BAR, [{ name: "others", labels, values: base }, { name: "SceneMiner", labels, values: top }],
      Object.assign({ barDir: "bar", barGrouping: "stacked", chartColors: [HEX.accent4, HEX.accent2], showValue: true,
        dataLabelPosition: "inEnd", dataLabelColor: HEX.lt1, dataLabelFormatCode: "0.000;-0.000;;", showLegend: false,
        catAxisOrientation: "maxMin", valAxisHidden: true, barGapWidthPct: 45, showTitle: true }, chartText(), o));
  };
  const table = (s, rows, o, hl = [], boldCells = {}) => {
    const data = rows.map((r, ri) => r.map((c, ci) => ({ text: c, options: {
      bold: ri === 0 || hl.includes(ri) || (boldCells[ri] || []).includes(ci),
      color: ri === 0 ? HEX.lt1 : HEX.dk1,
      fill: { color: ri === 0 ? HEX.dk2 : hl.includes(ri) ? "D7EFEC" : ri % 2 ? HEX.lt1 : "F6F7F9" },
      align: ci === 0 ? "left" : "center", valign: "middle" } })));
    s.addTable(data, Object.assign({ fontSize: 13, fontFace: "Calibri", border: { type: "solid", pt: 0.5, color: "D5DAE1" },
      margin: [0.04, 0.08, 0.04, 0.08] }, o));
  };

  // ============================================================== 1. title
  pres.addSection({ title: "Opening" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
  s.addText("SceneMiner: Scale-Aware Multi-Modal Fusion and Diagnostic Auditing for Language-Guided Driving Scene Retrieval",
    { placeholder: "title" });
  s.addText("Ph.D. dissertation defense", { placeholder: "body" });
  await badge(s, M, 0.9, 0.8, "MdDirectionsCar", C.accent2, "title-badge");
  s.addNotes("Welcome. This defense presents SceneMiner, a system that retrieves the rare driving scenes behind perception " +
    "failures from a natural-language description, and the audit that shows why such systems must be evaluated against " +
    "human labels. Add presenter name, committee and date on this slide before presenting.");

  // ============================================================== 2. outline
  s = content("Opening", "Outline");
  const outline = [["Introduction", "Why scene mining, and the problem"], ["Related work", "What exists and the gap"],
    ["Contributions", "Three contributions"], ["Method", "Scale-aware fusion and the concept gate"],
    ["Experiments", "Data, benchmarks, twelve baselines"], ["Results", "Human labels, external data, perception"],
    ["Discussion", "What generalises, scope"], ["Conclusion", "Take-aways"]];
  for (let i = 0; i < outline.length; i++) {
    const col = i % 2, row = Math.floor(i / 2), x = M + col * 6.1, y = 1.55 + row * 1.25;
    card(s, x, y, 5.8, 1.0, HEX.lt2, `outline-card-${i}`);
    numBadge(s, x + 0.2, y + 0.18, 0.64, i + 1, C.accent1, `outline-n-${i}`);
    txt(s, outline[i][0], { x: x + 1.05, y: y + 0.12, w: 4.6, h: 0.4, fontSize: 18, bold: true, objectName: `outline-t-${i}` });
    txt(s, outline[i][1], { x: x + 1.05, y: y + 0.52, w: 4.6, h: 0.4, fontSize: 14, color: C.text2, objectName: `outline-s-${i}` });
  }
  s.addNotes("The talk follows the paper: motivation and problem, related work, three contributions, the method, the " +
    "experimental protocol, results on human labels, external data and a perception check, then discussion and conclusion.");

  // ============================================================== 3. motivation
  section("Introduction", 1, "Finding the scenes that break perception");
  s = content("Introduction", "Autonomy failures hide in a thin tail of the data");
  bullets(s, ["Rain beading on the lens, headlight bloom at night, a wet road that destroys texture",
    "These scenes are rare in large driving corpora, yet they drive detection errors",
    "Engineers describe what they need in words: “a low-contrast scene on a wet road”",
    "Scene mining: retrieve those scenes from an unlabelled corpus by language"],
    { x: M, y: 1.45, w: 5.2, h: 3.6 });
  await fitImage(s, "qual_a.png", 6.05, 1.4, 6.7, 4.6, "motivation-retrievals");
  txt(s, "SceneMiner's top retrievals for three degradation queries (nuScenes, Waymo, BDD100K)",
    { x: 6.05, y: 6.1, w: 6.7, h: 0.4, fontSize: 12, color: C.text2, italic: true, objectName: "motivation-caption" });
  stat(s, M, 5.0, 5.2, "−9.1 pts", "front-sector 3D detection mAP when the front camera is degraded (unseen nuScenes scenes)",
    C.accent3, "motivation-stat");
  s.addNotes("Failures concentrate in rare conditions. On uncontaminated nuScenes validation scenes, a degraded front camera " +
    "costs a 3D detector 9.1 points of front-sector mAP. Engineers need to find such scenes quickly from a description. " +
    "The images are SceneMiner retrievals from the paper's qualitative figure.");

  // ============================================================== 4. problem statement
  s = content("Introduction", "Problem statement");
  txt(s, "Given a text query q and a corpus of N driving scenes, rank the scenes so the ones matching q come first, " +
    "using signals computed once per scene.", { x: M, y: 1.35, w: W - 2 * M, h: 0.8, fontSize: 18 });
  const flow = [["MdTextFields", "Query", "“slick pavement in the dark”"], ["MdStorage", "Cached signals", "BEV · VLM answers · image–text · metadata"],
    ["MdTune", "Fusion ranker", "learned, per query"], ["MdFormatListNumbered", "Top-k scenes", "of 1,508 in milliseconds"]];
  for (let i = 0; i < flow.length; i++) {
    const x = M + i * 3.07;
    card(s, x, 2.4, 2.75, 1.75, HEX.lt2, `flow-card-${i}`);
    await badge(s, x + 0.2, 2.6, 0.6, flow[i][0], i === 2 ? C.accent2 : C.text2, `flow-${i}`);
    txt(s, flow[i][1], { x: x + 0.95, y: 2.62, w: 1.7, h: 0.5, fontSize: 17, bold: true, objectName: `flow-t-${i}` });
    txt(s, flow[i][2], { x: x + 0.2, y: 3.35, w: 2.4, h: 0.7, fontSize: 13, color: C.text2, objectName: `flow-s-${i}` });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.78, y: 3.1, w: 0.26, h: 0.35, fill: { color: HEX.accent1 },
      line: { color: HEX.accent1 }, objectName: `flow-arrow-${i}` });
  }
  txt(s, "Requirements", { x: M, y: 4.45, w: 4, h: 0.4, fontSize: 18, bold: true });
  const req = [["Combine signal families", "hard scenes are hard for different reasons"],
    ["Cheap online cost", "milliseconds per query over a cached index"],
    ["Transfer to single cameras", "external corpora have no 3D rig"],
    ["Measured against humans", "not against the system's own labels"]];
  for (let i = 0; i < 4; i++) {
    const x = M + i * 3.07;
    numBadge(s, x, 4.95, 0.45, i + 1, C.accent1, `req-n-${i}`);
    txt(s, req[i][0], { x: x + 0.55, y: 4.92, w: 2.45, h: 0.5, fontSize: 15, bold: true, objectName: `req-t-${i}` });
    txt(s, req[i][1], { x: x + 0.55, y: 5.42, w: 2.45, h: 0.7, fontSize: 13, color: C.text2, objectName: `req-s-${i}` });
  }
  s.addNotes("Formally: a query, a corpus of cached scenes, a ranking. Four requirements shape the design: fuse several " +
    "signal families, stay cheap online, run on single-camera data, and be validated against human judgement.");

  // ============================================================== 5. three failure modes
  s = content("Introduction", "Three failure modes that standard evaluation hides");
  const fails = [["1.000 mAP", "Circular benchmark", "Relevance labels are the VLM's own cached outputs, so a label lookup scores perfectly — yet the VLM's precision against humans is only 0.50–0.68"],
    ["+74%", "Contaminated certifier", "The public BEVFormer checkpoint scores 0.7864 mAP on corpus scenes from its training split vs. 0.4514 on unseen val scenes"],
    ["30×", "Silent modality failure", "The VLM margin has 30× the spread of the SigLIP2 cosine; a “50/50” blend is ~97% margin, and LayerNorm on a scalar deletes the risk token"]];
  for (let i = 0; i < 3; i++) {
    const x = M + i * 4.1;
    card(s, x, 1.5, 3.8, 4.6, HEX.lt2, `fail-card-${i}`);
    numBadge(s, x + 0.25, 1.75, 0.55, i + 1, C.accent3, `fail-n-${i}`);
    txt(s, fails[i][0], { x: x + 0.25, y: 2.45, w: 3.3, h: 0.85, fontSize: 40, bold: true, color: C.accent3, fontFace: "Cambria",
      objectName: `fail-stat-${i}` });
    txt(s, fails[i][1], { x: x + 0.25, y: 3.4, w: 3.3, h: 0.5, fontSize: 18, bold: true, objectName: `fail-t-${i}` });
    txt(s, fails[i][2], { x: x + 0.25, y: 3.95, w: 3.3, h: 2.0, fontSize: 14, color: C.text2, objectName: `fail-s-${i}` });
  }
  txt(s, "None of the three shows up in task mAP — they must be audited directly.", { x: M, y: 6.3, w: W - 2 * M, h: 0.45,
    fontSize: 16, bold: true, color: C.accent2 });
  s.addNotes("Three defects. First, the standard predicate benchmark is circular. Second, the detector used to certify that " +
    "mined scenes are hard has memorised the corpus. Third, the natural learned fusion silently collapses onto its teacher " +
    "and can lose a modality entirely, while its task mAP looks fine.");

  // ============================================================== 6. related work
  section("Related work", 2, "Where SceneMiner sits");
  s = content("Related work", "Related work: each line of work reads one signal family");
  const rw = [["MdRoute", "Driving scene retrieval", "BEV-TSR (text-to-BEV) · CARIM (captions) · RefAV (LLM programs over tracks) · ReferGPT (3D state + CLIP) · Talk2BEV / nuPrompt (tags)", "Single signal family; need tracks, HD maps or an LLM service"],
    ["MdImageSearch", "Vision–language models", "CLIP · SigLIP / SigLIP2 (zero-shot retrieval) · BLIP-2 · Qwen2.5-VL (attribute questions)", "Open vocabulary, but blind to subtle sensor degradation"],
    ["MdCloud", "Adverse-condition data and VQA", "ACDC · Foggy Cityscapes · Dark Zurich · WoodScape · CODA · DriveLM · LingoQA", "Labelled sets or QA on a given scene; retrieval inverts the task"],
    ["MdCallSplit", "Fusion pitfalls", "Shortcut learning · expert collapse in gated mixtures · batch normalisation · text anisotropy", "Known in isolation; not addressed jointly for retrieval"]];
  for (let i = 0; i < 4; i++) {
    const x = M + (i % 2) * 6.1, y = 1.45 + Math.floor(i / 2) * 2.45;
    card(s, x, y, 5.85, 2.2, HEX.lt2, `rw-card-${i}`);
    await badge(s, x + 0.22, y + 0.22, 0.6, rw[i][0], C.text2, `rw-${i}`);
    txt(s, rw[i][1], { x: x + 1.0, y: y + 0.25, w: 4.7, h: 0.5, fontSize: 18, bold: true, objectName: `rw-t-${i}` });
    txt(s, rw[i][2], { x: x + 1.0, y: y + 0.75, w: 4.7, h: 0.85, fontSize: 13, color: C.text2, objectName: `rw-b-${i}` });
    txt(s, rw[i][3], { x: x + 1.0, y: y + 1.62, w: 4.7, h: 0.5, fontSize: 13, italic: true, color: C.accent3, objectName: `rw-g-${i}` });
  }
  txt(s, "Gap: no retriever fuses all signal families, stays cheap, and is validated against human labels.",
    { x: M, y: 6.4, w: W - 2 * M, h: 0.45, fontSize: 16, bold: true, color: C.accent2 });
  s.addNotes("Driving retrievers each read one kind of signal. Foundation models give open-vocabulary retrieval but miss " +
    "subtle degradations. Adverse-condition datasets are labelled sets, not retrieval. And the fusion pitfalls we hit are " +
    "known in other settings, but not handled together for scene retrieval. We re-implement five driving paradigms on our signals.");

  // ============================================================== 7. contributions
  section("Contributions", 3, "Three contributions");
  s = content("Contributions", "Contributions");
  const contr = [["SceneMiner", "A scale-aware fusion of cached BEV, VLM, image–text and metadata signals with a concept gate built only from training text. 3.5 ms per query; ahead of all twelve alternatives on human-labelled degradation.", "MdTune", C.accent2],
    ["A diagnostic audit of scene mining", "Circular benchmarks, contaminated certifiers and silent modality failure — each with a cheap check that applies to any mining system.", "MdFactCheck", C.accent3],
    ["Human-grounded evaluation", "550 human-labelled scenes (160 double-labelled, κ = 0.80), two external BDD100K sets incl. a pre-registered holdout, 40 semantic queries with 3D ground truth, and a perception check on uncontaminated scenes.", "MdGroups", C.accent1]];
  for (let i = 0; i < 3; i++) {
    const y = 1.45 + i * 1.7;
    card(s, M, y, W - 2 * M, 1.45, HEX.lt2, `contr-card-${i}`);
    await badge(s, M + 0.3, y + 0.32, 0.8, contr[i][2], contr[i][3], `contr-${i}`);
    txt(s, `${i + 1}. ${contr[i][0]}`, { x: M + 1.4, y: y + 0.18, w: 10.4, h: 0.45, fontSize: 20, bold: true, objectName: `contr-t-${i}` });
    txt(s, contr[i][1], { x: M + 1.4, y: y + 0.65, w: 10.4, h: 0.75, fontSize: 15, color: C.text2, objectName: `contr-s-${i}` });
  }
  s.addNotes("Three contributions. One: the SceneMiner method. Two: the audit, which explains why each design element is " +
    "needed and gives four cheap checks for other systems. Three: a human-grounded evaluation across datasets, " +
    "including a test that the mined scenes actually break 3D detection.");

  // ============================================================== 8. method overview
  section("Method", 4, "Scale-aware fusion with a training-text concept gate");
  s = content("Method", "SceneMiner at a glance");
  await fitImage(s, "fig1_main.jpg", M, 1.25, W - 2 * M, 5.65, "architecture-figure");
  s.addNotes("Stage 1 is frozen and runs once per scene: a multi-camera path gives a BEV embedding, tags and a risk scalar; " +
    "a single-camera path gives SigLIP2 image features, four zero-shot VLM yes/no margins and a VLM token; recorded " +
    "metadata fills a prior. Stage 2 has 4.68M trainable parameters: a standardised, routed base score, a FiLM-conditioned " +
    "cross-attention correction bounded by ResNorm, and a concept gate that chooses between the head and a fallback to " +
    "SigLIP2 plus metadata. Each query costs 3.5 ms.");

  // ============================================================== 9. scale-aware fusion
  s = content("Method", "Standardise everything that is combined");
  txt(s, "Score for query q and scene i", { x: M, y: 1.4, w: 6.2, h: 0.4, fontSize: 16, bold: true });
  card(s, M, 1.85, 6.2, 1.3, HEX.lt2, "eq-card");
  txt(s, [{ text: "s_head = w(q)ᵀ p̃ + α · g(q) · ρ̃", options: { breakLine: true } },
    { text: "s = γ_deg(q) · s_head + (1 − γ_deg(q)) · s_abst" }],
    { x: M + 0.25, y: 1.95, w: 5.8, h: 1.1, fontSize: 18, fontFace: "Cambria", valign: "middle", objectName: "eq-text" });
  bullets(s, ["PartNorm: the three base parts (SigLIP2 cosine, VLM margin, prior) are standardised with frozen training statistics, so a routing weight is an influence weight",
    "ResNorm: the cross-attention correction is standardised and bounded (|g| ≤ 1, α = 0.5): at most half a part's standard deviation",
    "Scalars are standardised, never LayerNormed: no dead modality"],
    { x: M, y: 3.4, w: 6.2, h: 3.2, fontSize: 15 });
  s.addChart(pres.charts.BAR, [{ name: "training std", labels: ["SigLIP2 cosine", "VLM margin", "prior"], values: [0.031, 0.566, 0.159] }],
    Object.assign({ barDir: "col", chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelColor: HEX.dk1,
      dataLabelFormatCode: "0.000", showLegend: false, valAxisHidden: true, barGapWidthPct: 60, showTitle: true,
      title: "Spread of the raw parts before PartNorm (training std)", x: 7.2, y: 1.4, w: 5.5, h: 4.4 }, chartText()));
  txt(s, "Before standardisation, the naive router puts 0.967–0.992 of its weight on the VLM margin and scores 0.486 on BDD100K vs. 0.622 for SigLIP2.",
    { x: 7.2, y: 5.95, w: 5.5, h: 0.8, fontSize: 13, color: C.text2, italic: true });
  s.addNotes("The core equation. The learned head is a routed sum of standardised parts plus a bounded correction; the final " +
    "score mixes the head and an abstention target by the concept gate. The chart shows the label-free evidence: running " +
    "training standard deviations of the three parts differ by 18 times, so unstandardised routing weights are meaningless.");

  // ============================================================== 10. concept gate
  s = content("Method", "A concept gate built only from training text");
  bullets(s, ["Bank: 252 training phrasings in 23 classes — 4 degradations, 7 contexts (night, rain, dense …), 12 generic phrases",
    "γ_deg(q): is the query one of the four degradations the VLM teacher was asked about? Context directions the query activates are removed first",
    "γ_macro(q): which recorded context (night, rain …) does it name?",
    "γ_deg ≈ 1 → trust the learned head; γ_deg ≈ 0 → abstain to SigLIP2 + the named metadata",
    "Strength and τ chosen on training text only (leave-group-out)"],
    { x: M, y: 1.4, w: 5.3, h: 5.2, fontSize: 15 });
  const g = D.gate.map((r) => [r[0], r[1], r[2], r[3]]);
  g[0] = ["query", "context", "γ_deg", "ranked by"];
  table(s, g, { x: 6.2, y: 1.45, w: 6.5, colW: [3.25, 0.95, 0.8, 1.5], fontSize: 12, rowH: 0.42 }, [], {});
  txt(s, "“Snow covering the road” is the remaining false trigger: no training phrasing describes snow.",
    { x: 6.2, y: 5.85, w: 6.5, h: 0.6, fontSize: 13, italic: true, color: C.text2 });
  s.addNotes("The gate measures whether a query lies inside the teacher's vocabulary. A degradation named with a context, " +
    "like slick pavement in the dark, keeps gamma at 0.985. Context and object queries such as a vehicle at night get zero " +
    "and are ranked by SigLIP2 plus the recorded night flag. One honest failure: snow covering the road still triggers it.");

  // ============================================================== 11. training & efficiency
  s = content("Method", "Training and inference cost");
  const st = [["4.68 M", "trainable parameters (Stage 2)", C.accent2], ["3.5 ms", "per query over 1,508 cached scenes (2.3 ms preparation + 1.2 ms ranking)", C.accent2],
    ["< 1 min", "per training seed on one A40 (3,000 AdamW steps)", C.accent1], ["2300×", "cheaper online than a per-query VLM reranker (8.2 s)", C.accent1]];
  for (let i = 0; i < 4; i++) {
    const x = M + i * 3.07;
    card(s, x, 1.5, 2.85, 2.35, HEX.lt2, `cost-card-${i}`);
    stat(s, x + 0.2, 1.6, 2.5, st[i][0], st[i][1], st[i][2], `cost-${i}`);
  }
  txt(s, "Training recipe", { x: M, y: 4.2, w: 5, h: 0.4, fontSize: 18, bold: true });
  bullets(s, ["Teacher labels: Qwen2.5-VL pseudo-labels on 906 training scenes — no human label enters training",
    "Channel dropout (margin 50% of steps, 3D tokens 30%) and an entropy floor stop teacher shortcuts",
    "Pair weights from agreement between the two frozen teachers; listwise InfoNCE + conditional hinge",
    "Identity start: gate and selectors at zero, so the head equals the base at step 0"],
    { x: M, y: 4.7, w: W - 2 * M, h: 2.0, fontSize: 15 });
  s.addNotes("SceneMiner is small and fast: scene tokens are projected once at indexing, so a query costs 3.5 ms. Training " +
    "takes under a minute per seed. Human labels are used only for evaluation.");

  // ============================================================== 12. experimental setup
  section("Experiments", 5, "Data, benchmarks and baselines");
  s = content("Experiments", "Experimental setup");
  const ds = [["MdStorage", "Corpus", "1,508 nuScenes + Waymo scenes, segment-disjoint: 906 train / 602 evaluation"],
    ["MdPerson", "Human labels", "550 scenes × 4 degradations; 160 double-labelled by a second rater (κ = 0.80)"],
    ["MdPublic", "External BDD100K", "1,000 images (12 queries) + a pre-registered holdout of 2,000 untouched images (36 queries)"],
    ["MdViewInAr", "Semantic queries", "40 queries graded by official nuScenes 3D annotation over 258 scenes (+ val150)"],
    ["MdVerified", "Uncontaminated check", "150 official nuScenes val scenes (val150) for all detector claims"],
    ["MdCompare", "Twelve alternatives", "5 re-implemented driving paradigms, CLIP, SigLIP2, trained adapter, VLM lookup, naive router, VLM reranker"]];
  for (let i = 0; i < 6; i++) {
    const x = M + (i % 3) * 4.1, y = 1.45 + Math.floor(i / 3) * 2.55;
    card(s, x, y, 3.85, 2.3, HEX.lt2, `ds-card-${i}`);
    await badge(s, x + 0.22, y + 0.22, 0.6, ds[i][0], i === 5 ? C.accent1 : C.text2, `ds-${i}`);
    txt(s, ds[i][1], { x: x + 0.95, y: y + 0.28, w: 2.8, h: 0.5, fontSize: 18, bold: true, objectName: `ds-t-${i}` });
    txt(s, ds[i][2], { x: x + 0.22, y: y + 0.98, w: 3.45, h: 1.2, fontSize: 14, color: C.text2, objectName: `ds-s-${i}` });
  }
  txt(s, "Metrics: mAP; five training seeds (± sd), scene bootstrap CIs, paired permutation tests.",
    { x: M, y: 6.6, w: W - 2 * M, h: 0.35, fontSize: 14, italic: true, color: C.text2 });
  s.addNotes("Evidence comes from human labels, external data and human 3D annotation, never from the circular benchmark. " +
    "Every method runs on identical splits, queries and metrics.");

  // ============================================================== 13. main comparison table
  section("Results", 6, "Human labels, external data, perception");
  s = content("Results", "Main comparison: paradigms and foundation models");
  const mt = D.master.filter((r) => r[0] !== "method");
  const head = ["Method", "Degradation (human, 550)", "BDD100K 1000", "BDD100K holdout", "Semantic (3D GT)", "Mean", "ms / query"];
  const rows = [head].concat(mt.map((r) => [r[0], r[2], r[3], r[4], r[5], r[6], r[7]]));
  const hlr = rows.map((r, i) => (r[0].startsWith("SceneMiner") ? i : -1)).filter((i) => i > 0);
  const boldc = {};
  const colBest = [1, 2, 3, 4, 5].map((c) => Math.max(...rows.slice(1).map((r) => parseFloat(r[c]) || 0)));
  rows.forEach((r, ri) => { if (ri) [1, 2, 3, 4, 5].forEach((c, k) => { if (parseFloat(r[c]) === colBest[k]) (boldc[ri] = boldc[ri] || []).push(c); }); });
  table(s, rows, { x: M, y: 1.3, w: W - 2 * M, colW: [3.35, 2.05, 1.35, 1.45, 1.6, 1.1, 1.233], fontSize: 12, rowH: 0.34 }, hlr, boldc);
  txt(s, "Bold: best per column. † reads BEV or tags supervised on corpus scenes. –: needs BEV/tags, unavailable on monocular images. Mean: unweighted mean of the four benchmarks.",
    { x: M, y: 6.45, w: W - 2 * M, h: 0.5, fontSize: 12, color: C.text2, italic: true });
  s.addNotes("The main table of the paper. SceneMiner has the best human-labelled degradation score, 0.794, and the best mean " +
    "over the four human-graded or external benchmarks, 0.529; the portable mode is second at 0.522. The per-query VLM " +
    "reranker leads on the external and semantic columns, but at 8,171 ms per query instead of 3.5.");

  // ============================================================== 14. charts: degradation + mean
  s = content("Results", "Best on human-labelled degradation and best overall");
  const lab = mt.filter((r) => r[2] !== "–").map((r) => r[0]);
  const val = mt.filter((r) => r[2] !== "–").map((r) => parseFloat(r[2]));
  const ord = val.map((v, i) => i).sort((a, b) => val[b] - val[a]);
  hbar(s, ord.map((i) => lab[i]), ord.map((i) => val[i]), ord.map((i, k) => (lab[i].startsWith("SceneMiner") ? k : -1)).filter((k) => k >= 0),
    { x: M, y: 1.3, w: 6.1, h: 5.2, title: "Degradation mAP, 550 human-labelled scenes", valAxisMinVal: 0, valAxisMaxVal: 1 });
  const mm = mt.filter((r) => r[6] !== "–");
  const mo = mm.map((r, i) => i).sort((a, b) => parseFloat(mm[b][6]) - parseFloat(mm[a][6]));
  hbar(s, mo.map((i) => mm[i][0]), mo.map((i) => parseFloat(mm[i][6])), mo.map((i, k) => (mm[i][0].startsWith("SceneMiner") ? k : -1)).filter((k) => k >= 0),
    { x: 6.85, y: 1.3, w: 5.9, h: 5.2, title: "Mean over the four benchmarks", valAxisMinVal: 0, valAxisMaxVal: 0.6 });
  txt(s, "SceneMiner 0.794 vs. 0.737 for the best no-learning control (late fusion of SigLIP2 and the VLM lookup).",
    { x: M, y: 6.55, w: W - 2 * M, h: 0.4, fontSize: 13, italic: true, color: C.text2 });
  s.addNotes("Left: human-labelled degradation. SceneMiner leads every alternative; even the portable mode, which reads no " +
    "signal supervised on corpus scenes, is ahead. Right: averaged over all four benchmarks, SceneMiner is first, ahead of " +
    "the VLM reranker that costs 2300 times more.");

  // ============================================================== 15. external transfer
  s = content("Results", "Transfers to external data without loss");
  const ex = [["0.614 vs 0.622", "BDD100K (1,000 images): level with zero-shot SigLIP2, p = 0.94", C.accent2],
    ["0.400 vs 0.408", "Pre-registered holdout (2,000 untouched images): level with SigLIP2, p = 0.28", C.accent2],
    ["+0.048 … +0.217", "over CLIP, ReferGPT-, CARIM- and RefAV-style rankers, VLM lookup and naive router (all p ≤ 0.013)", C.accent1]];
  for (let i = 0; i < 3; i++) {
    const x = M + i * 4.1;
    card(s, x, 1.45, 3.85, 2.3, HEX.lt2, `ext-card-${i}`);
    stat(s, x + 0.25, 1.55, 3.4, ex[i][0], ex[i][1], ex[i][2], `ext-${i}`, 30);
  }
  bullets(s, ["In vocabulary, SceneMiner gains +0.098 over SigLIP2 on BDD100K; out of vocabulary it abstains",
    "Holdout fixed before any score was computed, every method run once, frozen",
    "Larger backbone (SigLIP2 So400m/14): SceneMiner keeps its in-domain lead (0.784 vs 0.543) and inherits the external gain (holdout 0.468, BDD100K 0.650)",
    "A per-query VLM reranker is higher externally (0.640 on BDD100K) at 8.2 s per query — and misses what SigLIP2 does not retrieve: SceneMiner leads it by +0.159 on human labels"],
    { x: M, y: 4.05, w: W - 2 * M, h: 2.8, fontSize: 15 });
  s.addNotes("Externally, learned fusion is level with its open-vocabulary backbone and clearly above every other " +
    "alternative of comparable cost. The pre-registered holdout guards against tuning on test data.");

  // ============================================================== 16. second rater
  s = content("Results", "The human labels are reliable — and the VLM teacher is not");
  const r2 = D.rater2;
  const cats = r2[0].slice(1);
  const kap = r2.slice(1, 4);
  s.addChart(pres.charts.BAR, kap.map((r) => ({ name: r[0], labels: cats, values: r.slice(1).map(parseFloat) })),
    Object.assign({ barDir: "col", barGrouping: "clustered", chartColors: [HEX.accent2, HEX.accent4, "C5CCD6"], showValue: true,
      dataLabelPosition: "outEnd", dataLabelColor: HEX.dk2, dataLabelFormatCode: "0.00", showLegend: true, legendPos: "b",
      valAxisMinVal: 0, valAxisMaxVal: 1, valAxisHidden: true, showTitle: true, title: "Cohen's κ on 160 double-labelled scenes",
      x: M, y: 1.3, w: 7.4, h: 5.2 }, chartText()));
  const sm2 = r2.find((r) => r[0] === "SceneMiner");
  stat(s, 8.4, 1.5, 4.3, sm2[5], "SceneMiner mAP graded by rater 2 alone — above every other method", C.accent2, "r2-sm");
  stat(s, 8.4, 3.35, 4.3, "0.801", "κ between the two raters (0.734–0.883 per attribute)", C.accent1, "r2-k");
  txt(s, "People agree with each other far more than with the VLM (κ 0.520 / 0.501): human labels, not VLM outputs, must grade a miner.",
    { x: 8.4, y: 5.25, w: 4.3, h: 1.2, fontSize: 14, color: C.text2 });
  s.addNotes("A second rater independently labelled 160 scenes, half of them where the VLM is least certain. Inter-rater " +
    "kappa is 0.80, against about 0.5 between the VLM and either rater. Graded by the second rater alone, SceneMiner scores " +
    "0.798, again first; its leads over SigLIP2 and late fusion have bootstrap intervals above zero.");

  // ============================================================== 17. regimes
  s = content("Results", "Where learned fusion helps");
  const rg = D.regimes.filter((r) => r[0] !== "");
  rg[0] = ["Ranker", "Degradation (550)", "Semantics (258 / val150)", "Semantics × condition (258)"];
  table(s, rg, { x: M, y: 1.4, w: 7.0, colW: [2.4, 1.4, 1.75, 1.45], fontSize: 13, rowH: 0.45 }, [1],
    { 5: [3] });
  const reg = [["Degradation", "+0.065 over the strongest no-learning control", C.accent2],
    ["Atomic semantics", "γ_deg ≈ 0 → matches SigLIP2 (p = 0.46); +0.113 over the lookup", C.accent1],
    ["Object × condition", "ties SigLIP2 (p = 0.92); metadata fusion is the right bar here", C.accent4]];
  for (let i = 0; i < 3; i++) {
    const y = 1.4 + i * 1.6;
    card(s, 7.95, y, 4.8, 1.4, HEX.lt2, `reg-card-${i}`);
    txt(s, reg[i][0], { x: 8.15, y: y + 0.12, w: 4.4, h: 0.45, fontSize: 17, bold: true, color: reg[i][2], objectName: `reg-t-${i}` });
    txt(s, reg[i][1], { x: 8.15, y: y + 0.58, w: 4.4, h: 0.75, fontSize: 14, color: C.text2, objectName: `reg-s-${i}` });
  }
  txt(s, "The gate uses learning where the teacher has vocabulary and abstains everywhere else.",
    { x: M, y: 4.6, w: 7.0, h: 0.8, fontSize: 16, bold: true, color: C.accent2 });
  s.addNotes("Three regimes graded by humans. On degradation, the regime it was trained for, SceneMiner leads clearly. On " +
    "object queries it abstains and matches SigLIP2. On object-times-condition conjunctions it ties SigLIP2; a no-learning " +
    "fusion of SigLIP2 with recorded metadata is somewhat higher there, because that metadata defines the ground truth.");

  // ============================================================== 18. design progression
  s = content("Results", "Every component earns its place");
  const dg = D.design.slice(1);
  const dl = ["naive router", "+ routing fix, PartNorm", "+ ResNorm", "+ contrastive gate", "+ factored gate", "+ cond. removal (ours)"];
  const num = (c) => parseFloat(c.split(" ")[0]);
  s.addChart(pres.charts.LINE, [{ name: "Degradation (human)", labels: dl, values: dg.map((r) => num(r[1])) },
    { name: "BDD100K", labels: dl, values: dg.map((r) => num(r[2])) }, { name: "150-query suite", labels: dl, values: dg.map((r) => num(r[3])) }],
    Object.assign({ chartColors: [HEX.accent2, HEX.accent1, HEX.accent4], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 9,
      showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0.000", dataLabelColor: HEX.dk2, showLegend: true, legendPos: "b",
      valAxisMinVal: 0.4, valAxisMaxVal: 0.85, valAxisLabelFormatCode: "0.00", showTitle: true,
      title: "mAP after each design step (same training recipe, mean of five seeds)", x: M, y: 1.3, w: 8.0, h: 5.4 }, chartText()));
  bullets(s, ["PartNorm: +0.084 on BDD100K", "ResNorm: largest in-domain gain (0.790)", "Contrastive gate + abstention: first external result above SigLIP2, domain skew removed",
    "Factored gate: compositional queries recover (0.459 → 0.671)", "Final: best human-label and suite scores"],
    { x: 8.9, y: 1.5, w: 3.85, h: 5.0, fontSize: 15 });
  s.addNotes("Each step is trained with the same recipe. Standardisation fixes external transfer, the bounded residual gives " +
    "the in-domain gain, the gate removes domain skew, and the factored gate recovers compositional queries.");

  // ============================================================== 19. audit figure
  s = content("Results", "The audit, measured");
  await fitImage(s, "fig3_audit.png", M, 1.3, W - 2 * M, 4.4, "audit-figure");
  txt(s, "(a) A lookup over the cached labels scores 1.000 on the standard suite by construction — we report it as a ceiling.   " +
    "(b) The certifying detector scores +0.3349 mAP higher on its own training scenes than on unseen val scenes, so all detector claims use val150.",
    { x: M, y: 5.85, w: W - 2 * M, h: 0.9, fontSize: 14, color: C.text2 });
  s.addNotes("Two of the three failure modes, measured. These motivate evaluating on human labels and running every " +
    "detector claim on the uncontaminated val150 split.");

  // ============================================================== 20. perception
  s = content("Results", "SceneMiner retrieves the scenes that break 3D detection");
  const lp = D.loop.slice(1);
  const vru = lp.map((r) => -parseFloat(r[4]));
  hbar(s, lp.map((r) => r[0]), vru, [0], { x: M, y: 1.3, w: 6.6, h: 4.6,
    title: "Drop in VRU recall: each ranker's top 25 vs. the val150 mean", valAxisMinVal: 0, valAxisMaxVal: 0.22 });
  stat(s, 7.6, 1.45, 5.1, "−0.197", "vulnerable-road-user recall in SceneMiner's top 25 (p < 0.001), twice SigLIP2's drop", C.accent3, "loop-vru");
  stat(s, 7.6, 3.25, 5.1, "−9.1 pts", "front-sector mAP when CAM_FRONT is degraded (42 vs 108 scenes, q = 0.015)", C.accent3, "loop-map");
  txt(s, "Measured on 150 unseen nuScenes val scenes, front sector only, with scene-permutation tests.",
    { x: 7.6, y: 5.15, w: 5.1, h: 0.8, fontSize: 14, color: C.text2 });
  s.addNotes("The premise holds once degradation and error are measured in the same sector: degraded front cameras cost 9.1 " +
    "points of front-sector mAP. SceneMiner's top 25 retrievals lose 0.197 vulnerable-road-user recall, significantly more " +
    "than SigLIP2's, and on par with the VLM lookup. The effect is concentrated in the top quartile, which is what a " +
    "mining pipeline consumes.");

  // ============================================================== 21-22. qualitative
  s = content("Results", "Qualitative retrievals: sensor and surface degradations");
  await fitImage(s, "qual_a.png", M, 1.3, W - 2 * M, 4.9, "qual-degradations");
  txt(s, "Top two per corpus: nuScenes and Waymo (evaluation split) and BDD100K (external, portable signals); every image from a different driving segment.",
    { x: M, y: 6.3, w: W - 2 * M, h: 0.5, fontSize: 13, color: C.text2, italic: true });
  s.addNotes("Retrievals for droplets on the lens, headlight bloom, and low contrast on a wet road, across three corpora.");
  s = content("Results", "Qualitative retrievals: road users and objects");
  await fitImage(s, "qual_b.png", M, 1.25, W - 2 * M, 5.0, "qual-objects");
  txt(s, "Yellow: human annotations of the queried class (nuScenes and Waymo 3D boxes projected; BDD100K 2D boxes). Red label: no object of that class in the frame's labels.",
    { x: M, y: 6.3, w: W - 2 * M, h: 0.5, fontSize: 13, color: C.text2, italic: true });
  s.addNotes("For object queries the gate abstains, so these rankings come from SigLIP2 plus metadata; the boxes are human " +
    "annotations used to grade Stratum 5.");

  // ============================================================== 23. discussion
  section("Discussion", 7, "What generalises beyond SceneMiner");
  s = content("Discussion", "Four cheap checks for any scene-mining system");
  const chk = [["MdRule", "Score a lookup over your cached labels", "If it scores near 1.000, the benchmark measures agreement with your teacher, not progress"],
    ["MdHistory", "Ask whether the certifier trained on your corpus", "Memorisation inflated detector mAP by 74% on our corpus scenes"],
    ["MdStraighten", "Inspect the scales a fusion combines", "Running stds expose a 30× gap without a single label"],
    ["MdCenterFocusStrong", "Measure an effect where its cause acts", "Whole-scene tests found nothing; the front sector shows −9.1 points"]];
  for (let i = 0; i < 4; i++) {
    const x = M + (i % 2) * 6.1, y = 1.45 + Math.floor(i / 2) * 2.5;
    card(s, x, y, 5.85, 2.2, HEX.lt2, `chk-card-${i}`);
    await badge(s, x + 0.25, y + 0.3, 0.75, chk[i][0], C.accent2, `chk-${i}`);
    txt(s, chk[i][1], { x: x + 1.25, y: y + 0.3, w: 4.45, h: 0.8, fontSize: 18, bold: true, objectName: `chk-t-${i}` });
    txt(s, chk[i][2], { x: x + 1.25, y: y + 1.15, w: 4.45, h: 0.9, fontSize: 14, color: C.text2, objectName: `chk-s-${i}` });
  }
  s.addNotes("The audit generalises as four checks that cost minutes and apply to any mining or retrieval pipeline.");

  // ============================================================== 24. limitations
  s = content("Discussion", "Scope and future work");
  card(s, M, 1.45, 5.85, 3.6, HEX.lt2, "scope-card");
  txt(s, "Scope", { x: M + 0.3, y: 1.65, w: 5.2, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  bullets(s, ["Learned gains lie within the teacher's vocabulary — four degradation attributes",
    "Outside it SceneMiner is level with SigLIP2; a per-query VLM reranker is stronger there at 2300× the cost",
    "One external dataset; a second rater on 160 of the 550 scenes"],
    { x: M + 0.3, y: 2.25, w: 5.25, h: 2.7, fontSize: 16 });
  card(s, 6.85, 1.45, 5.85, 3.6, "E3F3F1", "future-card");
  txt(s, "Next steps", { x: 7.15, y: 1.65, w: 5.2, h: 0.5, fontSize: 20, bold: true, color: C.accent2 });
  bullets(s, ["Grow the concept bank and the teacher's questions: each new attribute extends the regime where learning helps",
    "Distil the per-query reranker's semantic strength into the cached index",
    "Close the loop: mine, label, retrain the detector, and measure the gain on val150",
    "More external corpora and raters"],
    { x: 7.15, y: 2.25, w: 5.25, h: 2.7, fontSize: 16 });
  stat(s, M, 5.3, W - 2 * M, "Beyond four attributes", "each attribute added to the teacher and the concept bank widens the regime where SceneMiner leads", C.accent2, "scope-stat", 30);
  s.addNotes("SceneMiner improves retrieval where its teacher has vocabulary and is never significantly worse than its " +
    "backbone elsewhere. The direct path to wider gains is a larger concept bank and more teacher questions.");

  // ============================================================== 25. conclusion
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Conclusion" });
  s.addText("Conclusion", { x: M, y: 0.6, w: W - 2 * M, h: 0.9, fontSize: 36, bold: true, color: C.background1, fontFace: "Cambria",
    isTextBox: true, margin: 0 });
  const tk = [["0.794", "Best human-labelled degradation retrieval of all methods compared, at 3.5 ms per query"],
    ["= SigLIP2", "No loss on external data and semantic queries: the gate abstains outside the teacher's vocabulary"],
    ["−0.197", "VRU recall in its top retrievals on unseen scenes: the mined scenes do break perception"]];
  for (let i = 0; i < 3; i++) {
    const x = M + i * 4.1;
    s.addText(tk[i][0], { x, y: 2.0, w: 3.8, h: 1.0, fontSize: 44, bold: true, color: C.accent1, fontFace: "Cambria", isTextBox: true,
      margin: 0, objectName: `tk-v-${i}` });
    s.addText(tk[i][1], { x, y: 3.1, w: 3.7, h: 1.4, fontSize: 16, color: C.background1, isTextBox: true, margin: 0, valign: "top",
      objectName: `tk-s-${i}` });
  }
  s.addText("Standardise what you fuse, gate the teacher by its vocabulary, and grade against humans.",
    { x: M, y: 5.2, w: W - 2 * M, h: 0.6, fontSize: 20, italic: true, color: C.accent6, isTextBox: true, margin: 0 });
  s.addNotes("Three take-aways: the best human-graded degradation retrieval at millisecond cost; no loss outside the " +
    "teacher's vocabulary; and retrievals that measurably break 3D detection on uncontaminated data.");

  // ============================================================== 26. questions
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusion" });
  s.addText("Thank you — questions?", { placeholder: "title" });
  s.addText("Code, benchmark definitions and anonymised human labels will be released.", { placeholder: "body" });
  s.addNotes("Backup slides follow: the full design-progression table and the second-rater table.");

  // ============================================================== backup
  pres.addSection({ title: "Backup" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: design progression (mAP ± sd over five seeds)", { placeholder: "title" });
  const dt = [["Design step", "Degradation (human)", "BDD100K", "150-query suite"]].concat(dg.map((r, i) => [dl[i], r[1], r[2], r[3]]));
  table(s, dt, { x: M, y: 1.5, w: 9.5, colW: [3.5, 2.0, 2.0, 2.0], fontSize: 15, rowH: 0.5 }, [6], {});
  s.addNotes("Full numbers behind the design-progression chart.");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  s.addText("Backup: second rater, per attribute", { placeholder: "title" });
  const r2t = r2.map((r) => r.slice());
  r2t[0][0] = "";
  table(s, r2t, { x: M, y: 1.5, w: 10.5, colW: [3.0, 1.5, 1.5, 1.5, 1.5, 1.5], fontSize: 15, rowH: 0.48 }, [4], {});
  txt(s, "Rows 1–3: Cohen's κ. Rows 4–7: mAP graded by rater 2 alone (fusion = late fusion of SigLIP2 and the VLM lookup).",
    { x: M, y: 5.7, w: 11, h: 0.5, fontSize: 13, italic: true, color: C.text2 });
  s.addNotes("Per-attribute agreement and mAP under the second rater.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
