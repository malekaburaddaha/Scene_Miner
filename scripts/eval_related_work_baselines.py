#!/usr/bin/env python3
"""MILESTONE 23 - SceneMiner against published driving-retrieval paradigms and
vision-language foundation models, on identical splits, queries and metrics.

Every published system below is a RE-IMPLEMENTATION OF ITS PARADIGM on the
signals cached in this project, not the authors' released system, and is
labelled "-style":

  BEV-TSR-style   text-to-BEV contrastive retrieval [Tang et al., AAAI 2025]:
                  MLP from the BEVFormer embedding + risk into SigLIP2 text space,
                  trained on the 906 training scenes (scripts/train_projection_baselines.py).
  RefAV-style     programmatic scenario filter [Davidson et al., 2025]: the query is
                  compiled into an executable conjunction of predicates over
                  structured perception attributes (VLM flags, scene/object tags,
                  recorded metadata). RefAV synthesises the program with an LLM; we
                  use a fixed rule-based compiler (no external LLM calls).
  CARIM-style     caption retrieval with inclusive dense + lexical matching
                  [Ki et al., ICCV 2025]: Qwen2.5-VL caption per image
                  (scripts/caption_scenes_qwen.py), scored by SigLIP2 text cosine
                  plus BM25, z-scored and summed.
  Tag-map         Talk2BEV/nuPrompt-style retrieval over the 20 language-aligned
                  scene/object tags [Choudhary et al., ICRA 2024; Wu et al., AAAI 2025]:
                  query-to-tag weights by text similarity, score = weighted tag logits.
  ReferGPT-style  3D-state description + CLIP matching [Chamiti et al., CVPRW 2025]:
                  CLIP image-text score plus CLIP text-text score between the query
                  and a description of the scene's structured attributes.
  CLIP ViT-B/32, SigLIP ViT-B/16, SigLIP2 ViT-B/16 (zero-shot), SigLIP2 + fine-tuned
  adapter, Qwen2.5-VL attribute log-odds lookup, naive uncalibrated fusion (the
  convex-router design of Tab. 2), and SceneMiner.

Methods that cannot run on a corpus (BEV or tags on monocular BDD100K / val150) are
reported as None, never imputed. On the 258 nuScenes evaluation scenes, any method
reading the BEV embedding or the tags is flagged contaminated (those encoders were
supervised on these scenes); SceneMiner is reported there in monocular mode.
"""
from __future__ import annotations
import os
import glob, json, math, re, sys, time
from collections import Counter
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary, p_at, ndcg_at, load_attributes, predicate_mask
from eval_human_gt_full import load_human, ATTRS, QUERY
from eval_semantic_edge_cases import (text_embed, load_v23, chance_ap, TRAIN_ORDER, COND_COL, META_OF)
from eval_external_bdd100k import EVAL_Q, QUESTIONS, CHANNEL
from eval_domain_calibrated_v2 import mmr
from train_projection_baselines import BEV, SigFT

dev = torch.device("cuda")
W = Path("work_dirs/unified_filter")
TAG_NAMES = ["a dark scene", "a rainy scene", "a construction zone", "an intersection", "a parking lot",
             "a highway", "a school zone", "a residential street", "a pedestrian", "a bicycle",
             "a motorcycle", "a truck", "a bus", "a car", "an emergency vehicle", "an animal",
             "a jaywalking pedestrian", "a running pedestrian", "erratic driving", "stationary traffic"]
TAG_SHORT = ["dark", "rainy", "construction zone", "intersection", "parking lot", "highway", "school zone",
             "residential", "pedestrians", "bicycles", "motorcycles", "trucks", "buses", "cars",
             "emergency vehicles", "animals", "jaywalking", "running pedestrians", "erratic driving", "stopped traffic"]

def zs(x):
    x = np.asarray(x, float); sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


# ---------------------------------------------------------------- RefAV-style compiler
LEX = [  # (regex, predicate)
    (r"droplet|raindrop|spatter|spotted lens|wet lens|on the lens|lens", ("flag", "droplets")),
    (r"glare|bloom|flare|halo|headlight|low-sun|sun", ("flag", "glare")),
    (r"wet|reflective|puddle|slick|damp|standing water|glisten", ("flag", "wet_road")),
    (r"low contrast|murky|hazy|indistinct|barely|poor visibility|washed out|shapes", ("flag", "low_contrast")),
    (r"night|after dark|darkness|unlit|in the dark|lamplight", ("meta", ("lighting", "night"))),
    (r"daylight|daytime|day\b|afternoon|sunny|bright", ("meta", ("lighting", "day"))),
    (r"rain|downpour|drizzle|shower|rainy", ("meta", ("weather", "rain"))),
    (r"\bdry\b|clear weather|no precipitation", ("meta", ("weather", "dry"))),
    (r"dense|jam|congest|busy|bumper|heavy traffic|queued", ("meta", ("density", "dense"))),
    (r"empty|sparse|few vehicles|light traffic|quiet", ("meta", ("density", "sparse"))),
    (r"pedestrian|people|person|walking|crosswalk|on foot", ("tag", [8])),
    (r"cyclist|bicycl|bike|motorcycl|rider|vulnerable", ("tag", [9, 10])),
    (r"truck|bus\b|trailer|heavy vehicle|large vehicle", ("tag", [11, 12])),
    (r"construction|cone|barrier|work zone|road works", ("tag", [2])),
    (r"child|stroller|wheelchair|police|emergency|debris|unusual|rare|unexpected", ("tag", [14])),
]
NEG = re.compile(r"\b(no|without|not|non)\b")


def compile_program(q):
    ql = q.lower(); preds = []
    for rx, pr in LEX:
        m = re.search(rx, ql)
        if m:
            neg = bool(NEG.search(ql[max(0, m.start() - 20):m.start()]))
            preds.append((pr, neg))
    return preds


def run_program(preds, flags, marg, meta, tags, n):
    """flags: attr->bool array (VLM decisions); marg: attr->margin array; meta:
    field->value array or None; tags: (n,20) logits or None. A program's output is
    the share of its predicates a scene satisfies (tag predicates use the head's own
    decision threshold, logit > 0), with ties broken by the mean standardised
    evidence behind the predicates (weight 1e-3, too small to override a predicate). Predicates the corpus has no
    signal for are dropped; an empty program ranks every scene equally."""
    sat, ev, cnt = np.zeros(n), np.zeros(n), 0
    for (kind, arg), neg in preds:
        if kind == "flag" and flags is not None: v, e = flags[arg], zs(marg[arg])
        elif kind == "meta" and meta is not None and arg[0] in meta:
            v = meta[arg[0]] == arg[1]; e = v.astype(float)
        elif kind == "tag" and tags is not None:
            v = (tags[:, arg] > 0).any(1); e = zs(tags[:, arg].max(1))
        else: continue
        sat += (~v if neg else v); ev += (-e if neg else e); cnt += 1
    return (sat + 1e-3 * ev) / cnt if cnt else np.zeros(n)


# ---------------------------------------------------------------- BM25
TOK = re.compile(r"[a-z0-9]+")
STOP = set("a an the of on in at with and or to is are be from by for this that as it its into near".split())
def toks(s): return [w for w in TOK.findall(s.lower()) if w not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.d = [Counter(toks(x)) for x in docs]; self.L = np.array([sum(c.values()) for c in self.d], float)
        self.avg = self.L.mean(); self.k1, self.b = k1, b
        df = Counter(w for c in self.d for w in c); N = len(docs)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    def score(self, q):
        s = np.zeros(len(self.d))
        for w in set(toks(q)):
            if w not in self.idf: continue
            tf = np.array([c.get(w, 0) for c in self.d], float)
            s += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.L / self.avg))
        return s


# ---------------------------------------------------------------- encoders
class Enc:
    """CLIP ViT-B/32, SigLIP ViT-B/16 and SigLIP2 ViT-B/16 text towers, kept
    resident; text embeddings are cached so ranking latency excludes encoding
    (text-encoder latency is timed separately)."""
    def __init__(self):
        from transformers import AutoModel, CLIPModel, CLIPTokenizerFast, GemmaTokenizerFast, SiglipTokenizer
        self.m = {"clip": CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(dev).eval(),
                  "sig1": AutoModel.from_pretrained("google/siglip-base-patch16-224", use_safetensors=True).to(dev).eval(),
                  "sig2": AutoModel.from_pretrained("google/siglip2-base-patch16-224", use_safetensors=True).to(dev).eval()}
        self.tk = {"clip": (CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32"), dict(padding=True, truncation=True, max_length=77)),
                   "sig1": (SiglipTokenizer.from_pretrained("google/siglip-base-patch16-224"), dict(padding="max_length", truncation=True, max_length=64)),
                   "sig2": (GemmaTokenizerFast.from_pretrained("google/siglip2-base-patch16-224"), dict(padding="max_length", truncation=True, max_length=64))}
        self.cache = {k: {} for k in self.m}

    def encode(self, kind, texts, use_cache=True):
        miss = [t for t in dict.fromkeys(texts) if not use_cache or t not in self.cache[kind]]
        tk, kw = self.tk[kind]
        with torch.no_grad():
            for i in range(0, len(miss), 128):
                o = self.m[kind].get_text_features(**tk(miss[i:i+128], return_tensors="pt", **kw).to(dev))
                E = (o if torch.is_tensor(o) else o.pooler_output).float()
                for t, e in zip(miss[i:i+128], E): self.cache[kind][t] = e
        return torch.stack([self.cache[kind][t] for t in texts])           # raw (unnormalised) features

    def clip_t(self, x): return F.normalize(self.encode("clip", x), dim=-1)
    def sig1_t(self, x): return F.normalize(self.encode("sig1", x), dim=-1)
    def sig2_t(self, x): return F.normalize(self.encode("sig2", x), dim=-1)
    def sig2_raw(self, x): return self.encode("sig2", x)     # what the SceneMiner heads were trained on


# ---------------------------------------------------------------- data
def _t(x): return torch.from_numpy(np.asarray(x)).float().to(dev)


def load_corpus():
    z = np.load(W/"features_v2.npz", allow_pickle=False)
    toks_ = [str(t) for t in z["scene_tokens"]]
    vlm = json.load(open(W/"binary_flags_with_logits.json")); strict = json.load(open(W/"lowcontrast_strict.json"))
    meta = json.load(open(W/"deterministic_gt.json"))
    def src(a): return (strict, "low_contrast_strict") if a == "low_contrast" else (vlm, a)
    flags = {a: np.array([bool(src(a)[0].get(t, {}).get(src(a)[1], False)) for t in toks_]) for a in ATTRS}
    marg = {a: np.array([float(src(a)[0].get(t, {}).get(src(a)[1] + "_margin", -9.)) for t in toks_]) for a in ATTRS}
    md = {f: np.array([meta.get(t, {}).get(f) for t in toks_]) for f in ("lighting", "weather", "density")}
    rk = z["risk"].reshape(len(toks_), -1)
    return {"name": "corpus", "toks": toks_, "dom": z["domain"].astype(int), "n": len(toks_),
            "F": (_t(z["emb"]), _t(z["tag_logits"]), _t(z["risk"]),
                  torch.load(W/"V_frozen_cache.pt", map_location=dev).float(), _t(z["siglip_img"]),
                  _t(z["vlm_margin_z"]), _t(z["prior"])),
            "sig2": F.normalize(_t(z["siglip_img"]), dim=-1), "clip": _t(np.load("work_dirs/emb_clip.npz")["corpus"]),
            "sig1": _t(np.load("work_dirs/emb_siglip1.npz")["corpus"]), "flags": flags, "marg": marg, "meta": md,
            "tags": z["tag_logits"], "bev": np.concatenate([z["emb"], (rk - rk.mean()) / rk.std()], 1),
            "prior": _t(z["prior"]), "mz": _t(z["vlm_margin_z"])}


def _portable_dataset(name, toks_, S, M, PR, meta):
    MZ = (M - M.mean(0)) / (M.std(0) + 1e-6)
    E = {"clip": np.load("work_dirs/emb_clip.npz")[name], "sig1": np.load("work_dirs/emb_siglip1.npz")[name]}
    return {"name": name, "toks": toks_, "n": len(toks_), "sig2": F.normalize(_t(S), dim=-1),
            "clip": _t(E["clip"]), "sig1": _t(E["sig1"]),
            "flags": {a: M[:, i] > 0 for i, a in enumerate(TRAIN_ORDER)}, "marg": {a: M[:, i] for i, a in enumerate(TRAIN_ORDER)},
            "meta": meta, "tags": None, "bev": None, "prior": PR, "mz": _t(MZ)}


def load_bdd():
    root = Path("/nfs/turbo/umd-srawa/scene_mining/datasets/external_adverse/bdd100k")
    labels = json.load(open(root/"labels.json")); names = sorted(labels)
    z = np.load(root/"signals.npz", allow_pickle=False)
    ASK = [k for k, _ in QUESTIONS]; M = z["margin"][:, [ASK.index(k) for k in TRAIN_ORDER]]
    # BDD's own weather/time-of-day labels ARE the ground truth, so no method reads them
    D = _portable_dataset("bdd", names, z["siglip"], M, torch.zeros(len(names), 26, device=dev), None)
    D["labels"] = labels
    return D


def load_val150(G):
    vz = np.load("work_dirs/val150_siglip.npz", allow_pickle=False); sc = [str(t) for t in vz["scenes"]]
    mz = np.load("work_dirs/val150_camfront_signals.npz", allow_pickle=False); msc = [str(t) for t in mz["scenes"]]
    ASK = ["wet_road", "glare", "droplets", "low_contrast"]
    M = mz["margin"][[msc.index(t) for t in sc]][:, [ASK.index(k) for k in TRAIN_ORDER]]
    PR = torch.zeros(len(sc), 26, device=dev)          # recorded metadata from the scene description
    for i, t in enumerate(sc):
        PR[i, 20] = float(G[t]["night"]); PR[i, 21] = float(not G[t]["night"])
        PR[i, 22] = float(G[t]["rain"]); PR[i, 23] = float(not G[t]["rain"])
    md = {"lighting": np.array(["night" if G[t]["night"] else "day" for t in sc]),
          "weather": np.array(["rain" if G[t]["rain"] else "dry" for t in sc])}
    return _portable_dataset("val150", sc, vz["siglip"], M, PR, md)


def load_captions(kind="neutral"):
    """neutral (Milestone 28): affirmative prompt; enumerating (Milestone 23): the prompt
    that listed the degradations. Holdout captions are keyed 'holdout|<image>'."""
    pat = "work_dirs/qwen_captions_neutral*_shard*.json" if kind == "neutral" else "work_dirs/qwen_captions_shard*.json"
    cap = {}
    for f in sorted(glob.glob(pat)): cap.update(json.load(open(f)))
    return cap


HOLD = Path("/nfs/turbo/umd-srawa/scene_mining/datasets/external_adverse/bdd100k_holdout")


def load_holdout(with_meta=False):
    """Untouched BDD100K holdout (Milestone 28). with_meta gives the methods that read
    recorded metadata BDD100K's own timeofday/weather tags - which are also the
    ground truth of the condition queries, so that mode is labelled circular."""
    labels = json.load(open(HOLD/"labels.json")); names = sorted(labels)
    z = np.load(HOLD/"signals.npz", allow_pickle=False)
    assert [str(x) for x in z["names"]] == names
    ASK = [k for k, _ in QUESTIONS]; M = z["margin"][:, [ASK.index(k) for k in TRAIN_ORDER]]
    PR = torch.zeros(len(names), 26, device=dev); md = None
    if with_meta:
        tod = [labels[n]["timeofday"] for n in names]; wx = [labels[n]["weather"] for n in names]
        for i in range(len(names)):
            PR[i, 20] = float(tod[i] == "night"); PR[i, 21] = float(tod[i] == "daytime")
            PR[i, 22] = float(wx[i] == "rainy"); PR[i, 23] = float(wx[i] in ("clear", "partly cloudy", "overcast"))
        md = {"lighting": np.array(["night" if t == "night" else "day" if t == "daytime" else "dusk" for t in tod]),
              "weather": np.array(["rain" if w == "rainy" else "dry" if w in ("clear", "partly cloudy", "overcast") else w
                                   for w in wx])}
    MZ = (M - M.mean(0)) / (M.std(0) + 1e-6)
    D = {"name": "holdout", "toks": names, "n": len(names), "sig2": F.normalize(_t(z["siglip"]), dim=-1),
         "clip": _t(np.load("work_dirs/emb_clip_holdout.npz")["holdout"]),
         "sig1": _t(np.load("work_dirs/emb_siglip1_holdout.npz")["holdout"]),
         "flags": {a: M[:, i] > 0 for i, a in enumerate(TRAIN_ORDER)}, "marg": {a: M[:, i] for i, a in enumerate(TRAIN_ORDER)},
         "meta": md, "tags": None, "bev": None, "prior": PR, "mz": _t(MZ), "labels": labels}
    return D


def holds(lab, p):
    """pre-registered predicate (benchmark_cvpr/bdd_holdout_queries_36.json) on BDD100K labels"""
    if "all" in p: return all(holds(lab, x) for x in p["all"])
    if "any" in p: return any(holds(lab, x) for x in p["any"])
    if "box" in p:
        b = p["box"]
        n = sum(1 for x in lab["boxes"] if x["label"] in b["labels"] and x["bbox"][2] * x["bbox"][3] >= b["min_area"])
        return n >= b["min_count"] and ("max_count" not in b or n <= b["max_count"])
    return all(lab[k] in v for k, v in p.items())


# a-priori VLM-margin column for the lookup baseline on the holdout queries, fixed from
# the query wording before any holdout score was computed (same rules as CHANNEL)
HOLD_CHANNEL = {"a1": ("wet_road", 1), "a2": ("wet_road", 1), "a3": ("low_contrast", 1), "a4": ("low_contrast", 1),
                "a5": ("low_contrast", 1), "a6": None, "a7": ("low_contrast", 1), "a8": ("wet_road", 1),
                "a9": ("droplets", 1), "a10": ("glare", 1), "a11": ("low_contrast", 1), "a12": ("wet_road", 1),
                "c1": ("low_contrast", 1), "c2": ("wet_road", 1), "c3": ("low_contrast", 1), "c4": ("wet_road", 1),
                "c5": ("wet_road", 1), "c6": ("low_contrast", 1), "c7": ("low_contrast", 1), "c8": ("low_contrast", -1),
                "c9": ("low_contrast", 1), "c10": ("wet_road", 1), "c11": ("low_contrast", 1), "c12": ("low_contrast", 1)}


# ---------------------------------------------------------------- the rankers
def _load(cls, pat, **kw):
    M = []
    for c in sorted(glob.glob(pat)):
        d = torch.load(c, map_location=dev, weights_only=False)
        m = cls(**kw).to(dev); m.load_state_dict(d["state_dict"]); m.eval(); M.append(m)
    return M


class Rankers:
    """Each ranker maps (dataset, query) to scores of shape (n,), or (seeds, n)
    for trained methods, or None when the method cannot run on that dataset."""
    def __init__(self, enc, captions, rk, captions_enum=None, programs=None):
        from scene_mining.filter.sceneminer_v2 import SceneMinerV21
        self.enc, self.cap, self.cap_enum = enc, captions, captions_enum or {}
        self.programs = programs or {}
        self.M25 = load_v23(rk, which=os.environ.get("SM_FINAL", "v26"))
        for m in self.M25: m.use_abstain = True
        self.M21 = _load(SceneMinerV21, "work_dirs/sceneminer_v21/s*/final.pt",
                         d_qwen=2048, d_siglip=768, n_margins=4, n_meta=6, d_text=768)
        self.bev = _load(BEV, "work_dirs/baseline_bev/s*/final.pt", d_in=257)
        self.sft = _load(SigFT, "work_dirs/baseline_sigft/s*/final.pt")
        assert len(self.M25) == len(self.M21) == len(self.bev) == len(self.sft) == 5
        self.tagE = enc.sig2_t(TAG_NAMES)
        self.k = 5          # seeds used by the trained rankers (1 for single-model latency)
        self.cap_ix = {}

    def clip(self, D, q): return (D["clip"] @ self.enc.clip_t([q])[0]).cpu().numpy()
    def siglip1(self, D, q): return (D["sig1"] @ self.enc.sig1_t([q])[0]).cpu().numpy()
    def siglip2(self, D, q): return (D["sig2"] @ self.enc.sig2_t([q])[0]).cpu().numpy()

    def siglip2_ft(self, D, q):
        t = self.enc.sig2_t([q])[0]
        with torch.no_grad(): return np.stack([(m(D["sig2"]) @ t).cpu().numpy() for m in self.sft[:self.k]])

    def bev_tsr(self, D, q):
        if D["bev"] is None: return None
        if "bev_t" not in D: D["bev_t"] = _t(D["bev"])
        t = self.enc.sig2_t([q])[0]
        with torch.no_grad(): return np.stack([(m(D["bev_t"]) @ t).cpu().numpy() for m in self.bev[:self.k]])

    def tag_map(self, D, q):
        if D["tags"] is None: return None
        qe = self.enc.sig2_t([q])[0]; mu = self.tagE.mean(0)
        sim = F.normalize(self.tagE - mu, dim=-1) @ F.normalize(qe - mu, dim=-1)
        w = torch.softmax(sim / 0.05, 0).cpu().numpy()
        T = D["tags"]; return ((T - T.mean(0)) / (T.std(0) + 1e-6)) @ w

    def refav(self, D, q):
        return run_program(compile_program(q), D["flags"], D["marg"], D["meta"], D["tags"], D["n"])

    def carim(self, D, q):
        """neutral captions; dense score = max over the caption's clauses (sentence and
        comma/semicolon splits) of the SigLIP2 text cosine, plus BM25 on the caption."""
        key = ("neutral", D["name"])
        if key not in self.cap_ix:
            docs = [self.cap.get(f"{D['name']}|{t}") for t in D["toks"]]
            if any(d is None for d in docs): return None
            cl, owner = [], []
            for i, d in enumerate(docs):
                parts = [c.strip() for c in re.split(r"[.;,]\s+|\.$", d) if len(c.strip().split()) >= 2] or [d]
                cl += parts; owner += [i] * len(parts)
            E = torch.cat([self.enc.sig2_t(cl[i:i+512]) for i in range(0, len(cl), 512)])
            self.cap_ix[key] = (BM25(docs), E, torch.as_tensor(owner, device=dev), len(docs))
        bm, E, own, n = self.cap_ix[key]
        c = E @ self.enc.sig2_t([q])[0]
        dense = torch.full((n,), -2.0, device=dev).scatter_reduce(0, own, c, reduce="amax")
        return zs(dense.cpu().numpy()) + zs(bm.score(q))

    def carim_enum(self, D, q):
        """Milestone 23 variant: enumerating prompt, whole-caption embedding + BM25."""
        key = ("enum", D["name"])
        if key not in self.cap_ix:
            docs = [self.cap_enum.get(f"{D['name']}|{t}") for t in D["toks"]]
            if any(d is None for d in docs): return None
            self.cap_ix[key] = (BM25(docs), self.enc.sig2_t(docs))
        bm, E = self.cap_ix[key]
        return zs((E @ self.enc.sig2_t([q])[0]).cpu().numpy()) + zs(bm.score(q))

    def refav_llm(self, D, q):
        """RefAV-style with the program compiled by a local LLM (scripts/refav_llm_compile.py)."""
        prog = self.programs.get(q, {}).get("program", [])
        preds = []
        for p in prog:
            if "flag" in p: preds.append((("flag", p["flag"]), not p["value"]))
            elif "meta" in p: preds.append((("meta", (p["meta"], p["value"])), False))
            elif "tag" in p: preds.append((("tag", [TAG_SHORT.index(p["tag"])]), not p["value"]))
        return run_program(preds, D["flags"], D["marg"], D["meta"], D["tags"], D["n"])

    def sceneminer_portable(self, D, q):
        """Uses only frozen web-trained signals: SigLIP2, the Qwen2.5-VL margins and
        recorded metadata. On the corpus the 20 tag columns of the prior are zeroed and
        the meta channel is off, exactly as on BDD100K / val150."""
        if D["name"] != "corpus": return self.sceneminer(D, q)
        if "prior_meta" not in D:
            P = D["prior"].clone(); P[:, :20] = 0; D["prior_meta"] = P
        qv = self.enc.sig2_raw([q])[0].unsqueeze(0).expand(D["n"], -1); N = D["n"]
        av = torch.stack([torch.ones(N, device=dev), torch.ones(N, device=dev), torch.zeros(N, device=dev)], -1)
        with torch.no_grad():
            return np.stack([m.base_score(qv, D["sig2"], D["mz"], D["prior_meta"], av).cpu().numpy() for m in self.M25[:self.k]])

    def refergpt(self, D, q):
        img = zs(self.clip(D, q))
        if D["tags"] is None and D["meta"] is None: return img
        if "state_desc" not in D:
            desc = []
            for i in range(D["n"]):
                parts = []
                if D["meta"] is not None:
                    parts += [f"{D['meta']['lighting'][i]}time", f"{D['meta']['weather'][i]} weather"]
                    if "density" in D["meta"]: parts.append(f"{D['meta']['density'][i]} traffic")
                if D["tags"] is not None: parts += [TAG_SHORT[j] for j in range(20) if D["tags"][i, j] > 0]
                desc.append("a driving scene: " + ", ".join(parts))
            D["state_desc"] = desc
        st = (self.enc.clip_t(D["state_desc"]) @ self.enc.clip_t([q])[0]).cpu().numpy()
        return img + zs(st)

    def lookup(self, D, q, col=None, sign=1):
        return np.zeros(D["n"]) if col is None else sign * D["marg"][col]

    def naive(self, D, q):
        qv = self.enc.sig2_raw([q])[0].unsqueeze(0).expand(D["n"], -1)
        with torch.no_grad():
            if D["name"] == "corpus":
                return np.stack([m(qv, qv, *D["F"]).cpu().numpy() for m in self.M21[:self.k]])
            return np.stack([m.base_score(qv, qv, D["sig2"], D["mz"], D["prior"]).cpu().numpy() for m in self.M21[:self.k]])

    def sceneminer(self, D, q, mono=False):
        qv = self.enc.sig2_raw([q])[0].unsqueeze(0).expand(D["n"], -1)
        with torch.no_grad():
            if D["name"] == "corpus":
                h = (torch.zeros if mono else torch.ones)(D["n"], device=dev)
                return np.stack([m(qv, qv, *D["F"], has_3d=h).cpu().numpy() for m in self.M25[:self.k]])
            N = D["n"]; av = torch.stack([torch.ones(N, device=dev), torch.ones(N, device=dev), torch.zeros(N, device=dev)], -1)
            return np.stack([m.base_score(qv, D["sig2"], D["mz"], D["prior"], av).cpu().numpy() for m in self.M25[:self.k]])


METHODS = [("BEV-TSR-style text-to-BEV", "bev_tsr", ["BEV"]),
           ("RefAV-style programmatic filter", "refav_llm", ["VLM flags", "tags", "metadata"]),
           ("RefAV-style (rule compiler, M23)", "refav", ["VLM flags", "tags", "metadata"]),
           ("CARIM-style caption retrieval", "carim", ["VLM captions"]),
           ("CARIM-style (enumerating prompt, M23)", "carim_enum", ["VLM captions"]),
           ("Tag-map (Talk2BEV/nuPrompt-style)", "tag_map", ["tags"]),
           ("ReferGPT-style state + CLIP", "refergpt", ["image", "tags", "metadata"]),
           ("CLIP ViT-B/32", "clip", ["image"]), ("SigLIP ViT-B/16", "siglip1", ["image"]),
           ("SigLIP2 ViT-B/16", "siglip2", ["image"]), ("SigLIP2 + fine-tuned adapter", "siglip2_ft", ["image"]),
           ("Qwen2.5-VL lookup", "lookup", ["VLM margins"]),
           ("Naive uncalibrated fusion", "naive", ["image", "VLM", "metadata", "BEV", "tags"]),
           ("SceneMiner (portable)", "sceneminer_portable", ["image", "VLM", "metadata"]),
           ("SceneMiner", "sceneminer", ["image", "VLM", "metadata", "BEV", "tags"])]
CONTAM = {"bev_tsr", "tag_map", "refav", "refav_llm", "refergpt", "naive"}   # read the BEV embedding or tags on corpus scenes
OFFLINE = {"carim": "one Qwen2.5-VL caption per image (timed below)", "carim_enum": "one Qwen2.5-VL caption per image",
           "refav_llm": "VLM flags + tags + metadata (cached); LLM compile per query",
           "sceneminer_portable": "Qwen2.5-VL + SigLIP2 per image (cached)", "lookup": "4 Qwen2.5-VL yes/no passes per image",
           "sceneminer": "BEVFormer + Qwen2.5-VL + SigLIP2 per scene (Stage 1, cached)",
           "naive": "same Stage-1 cache as SceneMiner", "refav": "VLM flags + tags + metadata (cached)",
           "bev_tsr": "BEVFormer per scene", "tag_map": "BEVFormer + tag head per scene",
           "refergpt": "CLIP image embedding + tags", "clip": "CLIP image embedding", "siglip1": "SigLIP image embedding",
           "siglip2": "SigLIP2 image embedding", "siglip2_ft": "SigLIP2 image embedding"}


def ens(s):
    return None if s is None else (np.asarray(s, float).mean(0) if np.ndim(s) == 2 else np.asarray(s, float))


def ap_of(s, rel, per_seed=False):
    """AP; a constant ranking scores the permutation expectation. per_seed
    averages the AP of each seed (the human-GT protocol), else the seed ensemble."""
    if s is None: return None
    S = np.atleast_2d(np.asarray(s, float)) if per_seed else ens(s)[None]
    return float(np.mean([chance_ap(rel) if np.ptp(x) == 0 else ap_binary(np.argsort(-x), rel) for x in S]))


def sign_perm(d, n=10000, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1, 1], (n, len(d))) * d).mean(1))
    return float((1 + (null >= abs(d.mean())).sum()) / (n + 1))


def main():
    t_all = time.time()
    enc = Enc(); cap = load_captions("neutral"); cap_enum = load_captions("enumerating")
    prog_path = Path("results/cvpr/refav_llm_programs.json")
    programs = json.load(open(prog_path))["programs"] if prog_path.exists() else {}
    print(f"[rw] captions: {len(cap)} neutral, {len(cap_enum)} enumerating; {len(programs)} LLM programs", flush=True)
    z = np.load(W/"features_v2.npz", allow_pickle=False)
    R = Rankers(enc, cap, z["risk"].reshape(-1), cap_enum, programs)
    OUT = Path(os.environ.get("SM_OUT", "results/cvpr/milestone28_related_work_comparison.json"))
    G = json.load(open(W/"human_semantic_gt.json"))["scenes"]
    C, B, V = load_corpus(), load_bdd(), load_val150(G)
    cidx = {t: i for i, t in enumerate(C["toks"])}

    def call(meth, D, q, col=None, sign=1, mono=False):
        f = getattr(R, meth)
        if meth == "lookup": return f(D, q, col, sign)
        if meth == "sceneminer": return f(D, q, mono=mono)
        return f(D, q)

    hum = load_human("results/day1_gate/malek_repaired.json", "B")
    ht = [t for t in C["toks"] if t in hum]; hidx = np.array([cidx[t] for t in ht])
    ev = set(json.load(open("benchmark_cvpr/split_906_602.json"))["eval"])
    s5t = [t for t in C["toks"] if t in ev and t in G and t in hum]; s5i = np.array([cidx[t] for t in s5t])
    S5 = json.load(open("benchmark_cvpr/queries_semantic_edge_cases.json"))["queries"]
    truth = lambda t, k: G[t][k] if (k in G[t] and k not in ATTRS) else hum[t][k]
    ST = [t for t in C["toks"] if t in ev]; sidx = np.array([cidx[t] for t in ST]); dom = C["dom"][sidx]
    Vn = F.normalize(C["F"][4], dim=-1).cpu().numpy()[sidx]
    attrs = load_attributes(W)
    for f in ("binary_flags_with_logits.json", "lowcontrast_strict.json"):
        for t, r in json.load(open(W/f)).items():
            rec = {k: v for k, v in r.items() if not k.endswith(("_margin", "_logp_yes"))}
            if "low_contrast_strict" in rec: rec["low_contrast"] = rec.pop("low_contrast_strict")
            attrs.setdefault(t, {}).update(rec)
    SQ = json.load(open("benchmark_cvpr/queries_390.json"))["eval"]
    # warm every text cache once so ranking latency is comparable across methods
    enc.sig2_t([QUERY[a] for a in ATTRS] + [q for q, _ in EVAL_Q] + [q["text"] for q in S5] + [q["text"] for q in SQ])

    out = {"doc": __doc__, "n": {"human_gt_scenes": len(ht), "bdd_images": B["n"], "s5_corpus_scenes": len(s5t),
                                 "val150_scenes": V["n"], "suite_scenes": len(ST)},
           "methods": {}, "per_query": {}}
    for label, meth, mods in METHODS:
        t0 = time.time(); r = {"key": meth, "modalities": mods, "reads_bev_or_tags_on_corpus": meth in CONTAM,
                               "offline_per_scene": OFFLINE[meth]}; PQ = {}
        aps = [ap_of(None if (s := call(meth, C, QUERY[at], col=at)) is None else s[..., hidx],
                     np.array([hum[t][at] for t in ht]), per_seed=True) for at in ATTRS]
        r["human_gt"] = None if None in aps else float(np.mean(aps)); r["human_gt_per_attr"] = dict(zip(ATTRS, aps))
        bq = []
        for qt, (field, pos) in EVAL_Q:
            rel = np.array([B["labels"][n][field] in pos for n in B["toks"]])
            if rel.sum() in (0, len(rel)): continue
            att, sgn = CHANNEL[qt]; s = ens(call(meth, B, qt, col=att, sign=sgn))
            bq.append(ap_of(s, rel)); PQ.setdefault("bdd", []).append(bq[-1])
            if s is not None: PQ.setdefault("bdd_p5", []).append(p_at(np.argsort(-s), rel))
        r["bdd"] = None if None in bq else float(np.mean(bq))
        r["bdd_P@5"] = float(np.mean(PQ["bdd_p5"])) if "bdd_p5" in PQ else None
        for st in ["semantic_atomic", "semantic_compositional"]:
            tag = st.split("_")[1]; v, vv = [], []
            for q in [q for q in S5 if q["stratum"] == st]:
                rel = np.array([all(truth(t, k) == x for k, x in q["predicate"].items()) for t in s5t])
                s = call(meth, C, q["text"], col=COND_COL[q["condition"]] if q["condition"] else None, mono=True)
                v.append(ap_of(None if s is None else ens(s)[s5i], rel))
            for q in [q for q in S5 if q["stratum"] == st and q["condition"] in (None, "night", "rain")]:
                rel = np.array([all(G[t][k] == x for k, x in q["predicate"].items()) for t in V["toks"]])
                if rel.sum() < 3: continue
                vv.append(ap_of(call(meth, V, q["text"], col=COND_COL[q["condition"]] if q["condition"] else None), rel))
            r[f"s5_corpus_{tag}"] = None if None in v else float(np.mean(v))
            r[f"s5_val150_{tag}"] = None if None in vv else float(np.mean(vv))
            PQ[f"s5_corpus_{tag}"], PQ[f"s5_val150_{tag}"] = v, vv
        sm, sk = [], []
        for q in SQ:
            rel = predicate_mask(attrs, ST, q["predicate"])
            if rel.sum() in (0, len(rel)): continue
            s = ens(call(meth, C, q["text"], col=next((k for k in ATTRS if k in q["predicate"]), None)))
            if s is None: sm = None; break
            s = s[sidx]; sm.append(ap_of(s, rel))
            o = np.argsort(-s) if np.ptp(s) > 0 else np.random.default_rng(len(sm)).permutation(len(s))
            sk.append(float(np.mean(dom[mmr(o, Vn, ST, 5)] == 0)) - float(np.mean(dom[rel] == 0)))
        r["suite"] = None if sm is None else float(np.mean(sm))
        r["suite_skew_pts"] = None if sm is None else float(np.mean(sk) * 100)
        PQ["suite"] = sm
        q = "water droplets smearing the camera lens"; call(meth, C, q, col="droplets"); torch.cuda.synchronize(); ts = []
        for _ in range(20):
            a = time.perf_counter(); call(meth, C, q, col="droplets"); torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - a))
        r["rank_latency_ms_1508"] = float(np.median(ts))
        if meth in ("bev_tsr", "siglip2_ft", "naive", "sceneminer"):     # 5-seed ensemble above; one model here
            R.k = 1; ts = []
            for _ in range(20):
                a = time.perf_counter(); call(meth, C, q, col="droplets"); torch.cuda.synchronize()
                ts.append(1000 * (time.perf_counter() - a))
            R.k = 5; r["rank_latency_ms_1508_single_model"] = float(np.median(ts))
        out["methods"][label] = r; out["per_query"][label] = PQ
        f = lambda x: "  n/a " if x is None else f"{x:.3f}"
        print(f"[rw] {label:<36} hum {f(r['human_gt'])} bdd {f(r['bdd'])} s5c {f(r['s5_corpus_atomic'])}/{f(r['s5_corpus_compositional'])}"
              f" s5v {f(r['s5_val150_atomic'])}/{f(r['s5_val150_compositional'])} suite {f(r['suite'])}"
              f" skew {f(r['suite_skew_pts'])} {r['rank_latency_ms_1508']:.2f}ms ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- untouched BDD100K holdout (pre-registered, run once)
    HQ = json.load(open("benchmark_cvpr/bdd_holdout_queries_36.json"))["queries"]
    out["holdout"] = {"protocol": json.load(open("benchmark_cvpr/bdd_holdout_queries_36.json"))["doc"], "modes": {}}
    for mode in ("metadata_free", "metadata_equipped_circular"):
        H = load_holdout(with_meta=(mode != "metadata_free"))
        rels = {q["id"]: np.array([holds(H["labels"][n], q["predicate"]) for n in H["toks"]]) for q in HQ}
        keep = [q for q in HQ if rels[q["id"]].sum() >= 5]
        enc.sig2_t([q["text"] for q in keep])
        res = {"n_images": H["n"], "queries_kept": [q["id"] for q in keep],
               "queries_excluded_lt5_positives": [q["id"] for q in HQ if q not in keep],
               "n_positives": {q["id"]: int(rels[q["id"]].sum()) for q in HQ}, "methods": {}}
        for label, meth, mods in METHODS:
            per = {}
            for q in keep:
                ch = HOLD_CHANNEL.get(q["id"])
                s = ens(call(meth, H, q["text"], col=ch[0] if ch else None, sign=ch[1] if ch else 1))
                if s is None: per = None; break
                per[q["id"]] = {"AP": ap_of(s, rels[q["id"]]),
                                "P@5": float(rels[q["id"]][np.argsort(-s)[:5]].mean()) if np.ptp(s) > 0 else float(rels[q["id"]].mean())}
            if per is None: res["methods"][label] = None; continue
            fam = {f: [v["AP"] for k, v in per.items() if k[0] == f] for f in "abc"}
            res["methods"][label] = {"mAP_all": float(np.mean([v["AP"] for v in per.values()])),
                                     "P@5_all": float(np.mean([v["P@5"] for v in per.values()])),
                                     **{f"mAP_{f}": float(np.mean(v)) for f, v in fam.items() if v}, "per_query": per}
        ref = res["methods"].get("SceneMiner (portable)")
        res["paired_vs_sceneminer_portable"] = {}
        for label, v in res["methods"].items():
            if v is None or label == "SceneMiner (portable)" or ref is None: continue
            for scope in ("all", "a", "b", "c"):
                ks = [k for k in ref["per_query"] if scope == "all" or k[0] == scope]
                d = np.array([ref["per_query"][k]["AP"] - v["per_query"][k]["AP"] for k in ks])
                res["paired_vs_sceneminer_portable"].setdefault(label, {})[scope] = {
                    "delta": float(d.mean()), "perm_p": sign_perm(d), "n_queries": len(d)}
        out["holdout"]["modes"][mode] = res
        print(f"[holdout:{mode}] kept {len(keep)}/36 queries", flush=True)
        for label, v in res["methods"].items():
            if v: print(f"   {label:<38} all {v['mAP_all']:.3f}  a {v.get('mAP_a', float('nan')):.3f}  "
                        f"b {v.get('mAP_b', float('nan')):.3f}  c {v.get('mAP_c', float('nan')):.3f}  P@5 {v['P@5_all']:.3f}", flush=True)

    # paired tests: SceneMiner minus each method, per-query sign-flip permutation
    ref = out["per_query"]["SceneMiner"]; out["paired_vs_sceneminer"] = {}
    for label in out["methods"]:
        if label == "SceneMiner": continue
        out["paired_vs_sceneminer"][label] = {}
        for k in ["bdd", "s5_corpus_atomic", "s5_corpus_compositional", "s5_val150_atomic", "s5_val150_compositional", "suite"]:
            a_, b_ = ref.get(k), out["per_query"][label].get(k)
            if not a_ or not b_ or None in b_: continue
            d = np.array(a_) - np.array(b_)
            out["paired_vs_sceneminer"][label][k] = {"delta": float(d.mean()), "perm_p": sign_perm(d), "n_queries": len(d)}
    # text-encoder latency (one query, uncached)
    out["text_encoder_ms"] = {}
    for kind in ("clip", "sig1", "sig2"):
        enc.encode(kind, ["warm up"], use_cache=False); torch.cuda.synchronize(); ts = []
        for i in range(20):
            a = time.perf_counter(); enc.encode(kind, [f"a query {i}"], use_cache=False); torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - a))
        out["text_encoder_ms"][kind] = float(np.median(ts))
    cal = Path(os.environ.get("SM_CAL", "results/cvpr/milestone27_v26.json"))
    if cal.exists():
        mm = json.load(open(cal))["A_human_gt"]["macro_mAP"]
        out["methods"]["SceneMiner"]["human_gt_calibrated"] = mm.get("v2.3_human_calibrated_5fold")
        out["methods"]["SceneMiner"]["human_gt_eval_v23_full"] = mm.get("v2.3_full")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"[rw] wrote {OUT} ({time.time()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
