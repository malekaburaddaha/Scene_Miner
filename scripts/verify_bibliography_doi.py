#!/usr/bin/env python3
"""MILESTONE 20 (part 2) - exact author lists, volumes and pages by DOI (Crossref),
plus a title search for the one work without an arXiv id or DOI hit."""
import json, time, urllib.parse, urllib.request
from pathlib import Path
UA = {"User-Agent": "sceneminer-bibcheck/1.0"}
def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60) as r: return json.loads(r.read())
DOIS = {"li2022bevformer": "10.1007/978-3-031-20077-9_1", "caesar2020nuscenes": "10.1109/cvpr42600.2020.01164",
        "sun2020waymo": "10.1109/cvpr42600.2020.00252", "yu2020bdd": "10.1109/cvpr42600.2020.00271",
        "zhai2023siglip": "10.1109/iccv51070.2023.01100", "sakaridis2021acdc": "10.1109/iccv48922.2021.01059",
        "choudhary2024talk2bev": "10.1109/icra57147.2024.10611485", "wu2023nuprompt": "10.1609/aaai.v39i8.32902",
        "geirhos2020shortcut": "10.1038/s42256-020-00257-z", "benjamini1995fdr": "10.1111/j.2517-6161.1995.tb02031.x",
        "carbonell1998mmr": "10.1145/290941.291025", "ethayarajh2019anisotropy": "10.18653/v1/d19-1006"}
out = {}
for k, d in DOIS.items():
    m = get(f"https://api.crossref.org/works/{d}")["message"]
    out[k] = {"doi": d, "title": m.get("title", [""])[0], "container": (m.get("container-title") or [""])[0],
              "year": m.get("issued", {}).get("date-parts", [[None]])[0][0], "volume": m.get("volume"),
              "issue": m.get("issue"), "page": m.get("page"), "publisher": m.get("publisher"),
              "authors": [f"{a.get('given','')} {a.get('family','')}".strip() for a in m.get("author", [])]}
    print(f"{k}: {out[k]['title'][:70]} | {out[k]['container'][:60]} {out[k]['year']} vol {out[k]['volume']} "
          f"no {out[k]['issue']} pp {out[k]['page']} | {len(out[k]['authors'])} authors: {', '.join(out[k]['authors'])}")
    time.sleep(0.5)
# Roy & Vetterli: Crossref bibliographic query restricted to 2007
q = "Roy Vetterli The effective rank: a measure of effective dimensionality 2007 European Signal Processing Conference"
r = get("https://api.crossref.org/works?rows=5&filter=from-pub-date:2007,until-pub-date:2008&query.bibliographic=" + urllib.parse.quote(q))
out["roy2007erank_search"] = [{"title": (i.get("title") or [""])[0], "container": (i.get("container-title") or [""])[0],
                               "doi": i.get("DOI"), "page": i.get("page"),
                               "authors": [f"{a.get('given','')} {a.get('family','')}".strip() for a in i.get("author", [])]}
                              for i in r["message"]["items"]]
for x in out["roy2007erank_search"]: print("  roy search:", x)
Path("paper/bibcheck/crossref_by_doi.json").write_text(json.dumps(out, indent=1))
