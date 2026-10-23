#!/usr/bin/env python3
"""MILESTONE 20 - verify every bibliography entry against live metadata.

For each cited work we query, over HTTPS:
  * the arXiv API by id            -> exact title, full author list, version date
  * Semantic Scholar (one batch)   -> published venue, year, DOI, DBLP key
  * Crossref by DOI / bibliographic query, for works without an arXiv id
and record the raw responses, so the final .bib is built from fetched metadata
rather than from memory. (DBLP answers automated clients with a bot challenge,
so it is not queried directly; Semantic Scholar returns DBLP keys instead.)
"""
import json, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path("paper/bibcheck"); OUT.mkdir(parents=True, exist_ok=True)
# key -> (arXiv id or None, DBLP query)
WORKS = {
 "bai2025qwen25vl": ("2502.13923", "Qwen2.5-VL Technical Report"),
 "benjamini1995fdr": (None, "Controlling the false discovery rate practical powerful approach multiple testing"),
 "caesar2020nuscenes": ("1903.11027", "nuScenes multimodal dataset autonomous driving Caesar"),
 "carbonell1998mmr": (None, "use of MMR diversity-based reranking reordering documents producing summaries"),
 "choudhary2024talk2bev": ("2310.02251", "Talk2BEV Language-enhanced Bird's-eye View Maps"),
 "ethayarajh2019anisotropy": ("1909.00512", "How Contextual are Contextualized Word Representations"),
 "geirhos2020shortcut": ("2004.07780", "Shortcut learning in deep neural networks"),
 "ioffe2015bn": ("1502.03167", "Batch Normalization Accelerating Deep Network Training Reducing Internal Covariate Shift"),
 "li2023blip2": ("2301.12597", "BLIP-2 Bootstrapping Language-Image Pre-training Frozen Image Encoders"),
 "li2022bevformer": ("2203.17270", "BEVFormer Learning Bird's-Eye-View Representation Multi-Camera Images"),
 "mu2018allbutthetop": ("1702.01417", "All-but-the-Top Simple and Effective Postprocessing for Word Representations"),
 "radford2021clip": ("2103.00020", "Learning Transferable Visual Models From Natural Language Supervision"),
 "roy2007erank": (None, "effective rank measure of effective dimensionality Roy Vetterli"),
 "shazeer2017moe": ("1701.06538", "Outrageously Large Neural Networks Sparsely-Gated Mixture-of-Experts"),
 "sun2020waymo": ("1912.04838", "Scalability in Perception for Autonomous Driving Waymo Open Dataset"),
 "tschannen2025siglip2": ("2502.14786", "SigLIP 2 Multilingual Vision-Language Encoders"),
 "wu2023nuprompt": ("2309.04379", "Language Prompt for Autonomous Driving"),
 "yu2020bdd": ("1805.04687", "BDD100K Diverse Driving Dataset Heterogeneous Multitask Learning"),
 "zhai2023siglip": ("2303.15343", "Sigmoid Loss for Language Image Pre-Training"),
 "sakaridis2021acdc": ("2104.13395", "ACDC Adverse Conditions Dataset with Correspondences"),
 "bardes2022vicreg": ("2105.04906", "VICReg Variance-Invariance-Covariance Regularization"),
}
UA = {"User-Agent": "sceneminer-bibcheck/1.0"}


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            err = e; time.sleep(3 * (k + 1))
    raise err


def arxiv(aid):
    x = get(f"https://export.arxiv.org/api/query?id_list={aid}")
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    e = ET.fromstring(x).find("a:entry", ns)
    j = e.find("x:journal_ref", ns)
    return {"title": " ".join(e.find("a:title", ns).text.split()),
            "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
            "published": e.find("a:published", ns).text[:10],
            "journal_ref": j.text if j is not None else None}


def s2_batch(ids):
    import json as _j
    body = _j.dumps({"ids": ids}).encode()
    for k in range(6):
        try:
            req = urllib.request.Request(
                "https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,venue,year,"
                "publicationVenue,externalIds,authors,journal", data=body,
                headers={**UA, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return _j.loads(r.read().decode())
        except Exception as e:
            err = e; time.sleep(15 * (k + 1))
    raise err


def crossref(q=None, doi=None):
    import json as _j
    if doi:
        m = _j.loads(get(f"https://api.crossref.org/works/{doi}"))["message"]
        items = [m]
    else:
        items = _j.loads(get("https://api.crossref.org/works?rows=3&query.bibliographic="
                             + urllib.parse.quote(q)))["message"]["items"]
    return [{"title": (i.get("title") or [""])[0], "container": (i.get("container-title") or [""])[0],
             "year": (i.get("issued", {}).get("date-parts") or [[None]])[0][0], "doi": i.get("DOI"),
             "volume": i.get("volume"), "issue": i.get("issue"), "page": i.get("page"),
             "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in i.get("author", [])]}
            for i in items]


DOI_ONLY = {"benjamini1995fdr": "10.1111/j.2517-6161.1995.tb02031.x",
            "carbonell1998mmr": "10.1145/290941.291025"}
res = {k: {"arxiv_id": aid} for k, (aid, q) in WORKS.items()}
for key, (aid, q) in WORKS.items():
    if aid:
        try: res[key]["arxiv"] = arxiv(aid)
        except Exception as e: res[key]["arxiv_error"] = repr(e)
        time.sleep(3.1)                                  # arXiv asks for >= 3 s between calls
ids = [f"arXiv:{aid}" for aid, _ in WORKS.values() if aid] + [f"DOI:{d}" for d in DOI_ONLY.values()]
keys = [k for k, (aid, _) in WORKS.items() if aid] + list(DOI_ONLY)
for k, r in zip(keys, s2_batch(ids)):
    res[k]["s2"] = r
for key, (aid, q) in WORKS.items():
    try:
        res[key]["crossref"] = crossref(doi=DOI_ONLY[key]) if key in DOI_ONLY else crossref(q=q)
    except Exception as e: res[key]["crossref_error"] = repr(e)
    time.sleep(0.5)
(OUT/"api_responses.json").write_text(json.dumps(res, indent=1))
for key, r in res.items():
    a = r.get("arxiv") or {}; s2 = r.get("s2") or {}
    pv = (s2.get("publicationVenue") or {}).get("name")
    print(f"== {key}")
    if a: print(f"   arXiv {r['arxiv_id']}: {a['title']}\n      {len(a['authors'])} authors: {', '.join(a['authors'][:4])}{' ...' if len(a['authors'])>4 else ''} | v1 {a['published']} | jref {a['journal_ref']}")
    if s2: print(f"   S2: venue={s2.get('venue')!r} pubVenue={pv!r} year={s2.get('year')} DOI={(s2.get('externalIds') or {}).get('DOI')} DBLP={(s2.get('externalIds') or {}).get('DBLP')}")
    for c in (r.get("crossref") or [])[:2]:
        print(f"   Crossref: {c['container']!r} {c['year']} DOI {c['doi']} | {c['title'][:80]}")
print(f"\nwrote {OUT/'api_responses.json'}")
