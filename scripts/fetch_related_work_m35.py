#!/usr/bin/env python3
"""MILESTONE 35 - fetch metadata for the related-work additions (never typed by hand).

Each candidate is identified by its exact title on OpenAlex (a normalised-title match;
anything else is reported and not used). Crossref gives the published record (authors,
venue, pages) for the DOI OpenAlex links; OpenReview gives the ICLR record of a paper
without a DOI. The arXiv API rate-limited this host (HTTP 429), so it is not used; the
arXiv ids first proposed were checked against these records (2303.08040 is not CRB).
Raw responses -> paper/bibcheck/related_work_check_3_raw.json; parsed records ->
paper/bibcheck/related_work_check_3.json
"""
import json, re, time, urllib.parse, urllib.request

CAND = {   # key -> exact title
    "yogamani2019woodscape": "WoodScape: A Multi-Task, Multi-Camera Fisheye Dataset for Autonomous Driving",
    "sakaridis2019darkzurich": "Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation",
    "sakaridis2018foggy": "Semantic Foggy Scene Understanding with Synthetic Data",
    "li2022coda": "CODA: A Real-World Road Corner Case Dataset for Object Detection in Autonomous Driving",
    "blum2021fishyscapes": "The Fishyscapes Benchmark: Measuring Blind Spots in Semantic Segmentation",
    "sima2024drivelm": "DriveLM: Driving with Graph Visual Question Answering",
    "qian2024nuscenesqa": "NuScenes-QA: A Multi-Modal Visual Question Answering Benchmark for Autonomous Driving Scenario",
    "deruyttere2019talk2car": "Talk2Car: Taking Control of Your Self-Driving Car",
    "marcu2024lingoqa": "LingoQA: Visual Question Answering for Autonomous Driving",
    "luo2023crb": "Exploring Active 3D Object Detection from a Generalization Perspective",
    "feng2019activelidar": "Deep Active Learning for Efficient Training of a LiDAR 3D Object Detector",
}
UA = {"User-Agent": "bibliography-check/1.0"}
CACHE = "paper/bibcheck/related_work_check_3_raw.json"      # raw responses by URL; reruns never refetch
try: RAW = json.load(open(CACHE))
except FileNotFoundError: RAW = {}
norm = lambda t: re.sub(r"[^a-z0-9]", "", (t or "").lower())


def get(url):
    if url in RAW: return RAW[url]
    req = urllib.request.Request(url, headers=UA)
    for k in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r: b = r.read().decode()
            RAW[url] = b; json.dump(RAW, open(CACHE, "w")); return b
        except Exception as e:
            if k == 5: raise
            time.sleep((20 if getattr(e, "code", None) == 429 else 3) * (k + 1))


def openalex(title):
    r = json.loads(get("https://api.openalex.org/works?per-page=10&search=" + urllib.parse.quote(title)))
    out = []
    for w in r["results"]:
        if norm(w.get("display_name")) != norm(title): continue
        out.append({"id": w["id"], "doi": (w.get("doi") or "").replace("https://doi.org/", ""), "year": w.get("publication_year"),
                    "authors": [a["author"]["display_name"] for a in w.get("authorships", [])],
                    "locations": [((l.get("source") or {}).get("display_name"), l.get("landing_page_url")) for l in w.get("locations", [])]})
    return out


def crossref(doi):
    m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    return {"doi": m.get("DOI"), "title": " ".join((m.get("title") or [""])[0].split()), "type": m.get("type"),
            "container": (m.get("container-title") or [None])[0], "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "volume": m.get("volume"), "issue": m.get("issue"), "page": m.get("page"), "publisher": m.get("publisher"),
            "authors": [" ".join(x for x in (a.get("given"), a.get("family")) if x) for a in m.get("author", [])]}


def openreview(title):
    r = json.loads(get("https://api2.openreview.net/notes/search?limit=10&term=" + urllib.parse.quote(title)))
    out = []
    for n in r.get("notes", []):
        c = n.get("content", {}); t = (c.get("title") or {}).get("value")
        if norm(t) != norm(title): continue
        out.append({"id": n.get("id"), "title": t, "venue": (c.get("venue") or {}).get("value"),
                    "authors": (c.get("authors") or {}).get("value")})
    return out


def main():
    out = {}
    for key, title in CAND.items():
        rec = {"title": title}
        try: oa = openalex(title)
        except Exception as e: oa = []; rec["openalex_error"] = repr(e)
        rec["openalex"] = oa
        dois = [o["doi"] for o in oa if o["doi"] and not o["doi"].lower().startswith("10.48550")]
        arx = sorted({u.rsplit("/", 1)[-1] for o in oa for _, u in o["locations"] if u and "arxiv.org/abs/" in u})
        rec["arxiv_ids_seen"] = arx
        if dois:
            try:
                rec["crossref"] = crossref(dois[0])
                rec["crossref_title_matches"] = norm(rec["crossref"]["title"]) == norm(title)
            except Exception as e: rec["crossref_error"] = repr(e)
        else:
            try: rec["openreview"] = openreview(title)
            except Exception as e: rec["openreview_error"] = repr(e)
        out[key] = rec
        cr = rec.get("crossref") or {}
        print(f"{key:<26} openalex={len(oa)} doi={dois[:1]} arxiv={arx} crossref={cr.get('container')!r} "
              f"{cr.get('year')} match={rec.get('crossref_title_matches')} openreview={[(x['venue']) for x in rec.get('openreview', [])]}",
              flush=True)
        time.sleep(1)
    json.dump(out, open("paper/bibcheck/related_work_check_3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
