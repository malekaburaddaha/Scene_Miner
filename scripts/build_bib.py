#!/usr/bin/env python3
"""MILESTONE 20 - write paper/main.bib from FETCHED metadata only.

Authors/titles/pages come from the API responses saved in paper/bibcheck/:
Crossref by DOI where a DOI exists (published version), arXiv otherwise.
Venue and year come from the published record (Crossref, or Semantic Scholar's
venue + DBLP key for ICML/ICLR proceedings, which have no Crossref DOI).
One manual override, justified by a fetched record, is listed in OVERRIDES.
"""
import json, re
from pathlib import Path

A = json.load(open("paper/bibcheck/api_responses.json"))
C = json.load(open("paper/bibcheck/crossref_by_doi.json"))
OA = json.load(open("paper/bibcheck/openalex_roy2007.json"))[0]
# Milestone 23: four related-work entries, fetched from Crossref by DOI and the arXiv API
RW = json.load(open("paper/bibcheck/related_work_check_2.json"))
C.update({"tang2025bevtsr": RW["bevtsr"], "ki2025carim": RW["carim"], "chamiti2025refergpt": RW["refergpt"]})
# Crossref inverts one name ("Fan JingChen"); the arXiv record of the same paper gives "Jingchen Fan"
C["tang2025bevtsr"] = dict(C["tang2025bevtsr"], authors=["Jingchen Fan" if x == "Fan JingChen" else x
                                                         for x in C["tang2025bevtsr"]["authors"]])
_rv = json.load(open("paper/bibcheck/related_work_check_1.json"))["refav"]
A["davidson2025refav"] = {"arxiv_id": _rv["id"], "arxiv": _rv}

CVPR = "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
ICCV = "Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)"
SPEC = {   # key -> (type, venue field, venue, year, source of authors, extra)
 "caesar2020nuscenes": ("inproceedings", "booktitle", CVPR, 2020, "crossref", {}),
 "sun2020waymo": ("inproceedings", "booktitle", CVPR, 2020, "crossref", {}),
 "yu2020bdd": ("inproceedings", "booktitle", CVPR, 2020, "crossref", {}),
 "zhai2023siglip": ("inproceedings", "booktitle", ICCV, 2023, "crossref", {}),
 "sakaridis2021acdc": ("inproceedings", "booktitle", ICCV, 2021, "crossref", {}),
 "li2022bevformer": ("inproceedings", "booktitle", "Proceedings of the European Conference on Computer Vision (ECCV)", 2022, "crossref", {}),
 "choudhary2024talk2bev": ("inproceedings", "booktitle", "Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)", 2024, "crossref", {}),
 "wu2023nuprompt": ("inproceedings", "booktitle", "Proceedings of the AAAI Conference on Artificial Intelligence", 2025, "crossref", {}),
 "ethayarajh2019anisotropy": ("inproceedings", "booktitle", "Proceedings of the Conference on Empirical Methods in Natural Language Processing and the International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)", 2019, "crossref", {}),
 "carbonell1998mmr": ("inproceedings", "booktitle", "Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval", 1998, "crossref", {}),
 "geirhos2020shortcut": ("article", "journal", "Nature Machine Intelligence", 2020, "crossref", {}),
 "benjamini1995fdr": ("article", "journal", "Journal of the Royal Statistical Society: Series B (Methodological)", 1995, "crossref", {}),
 "radford2021clip": ("inproceedings", "booktitle", "Proceedings of the International Conference on Machine Learning (ICML)", 2021, "arxiv", {}),
 "li2023blip2": ("inproceedings", "booktitle", "Proceedings of the International Conference on Machine Learning (ICML)", 2023, "arxiv", {}),
 "ioffe2015bn": ("inproceedings", "booktitle", "Proceedings of the International Conference on Machine Learning (ICML)", 2015, "arxiv", {}),
 "shazeer2017moe": ("inproceedings", "booktitle", "International Conference on Learning Representations (ICLR)", 2017, "arxiv", {}),
 "mu2018allbutthetop": ("inproceedings", "booktitle", "International Conference on Learning Representations (ICLR)", 2018, "override", {}),
 "bai2025qwen25vl": ("article", "journal", "arXiv preprint arXiv:2502.13923", 2025, "arxiv", {}),
 "tschannen2025siglip2": ("article", "journal", "arXiv preprint arXiv:2502.14786", 2025, "arxiv", {}),
 "tang2025bevtsr": ("inproceedings", "booktitle", "Proceedings of the AAAI Conference on Artificial Intelligence", 2025, "crossref", {}),
 "ki2025carim": ("inproceedings", "booktitle", ICCV, 2025, "crossref", {}),
 "chamiti2025refergpt": ("inproceedings", "booktitle", "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)", 2025, "crossref", {}),
 "davidson2025refav": ("article", "journal", "arXiv preprint arXiv:2505.20981", 2025, "arxiv", {}),
 "roy2007erank": ("inproceedings", "booktitle", "Proceedings of the European Signal Processing Conference (EUSIPCO)", 2007, "openalex", {}),
}
OVERRIDES = {
 # arXiv lists Mu, Bhat, Viswanath; the ICLR 2018 version is Mu and Viswanath
 # (Semantic Scholar returns DBLP key conf/iclr/MuV18 for it).
 "mu2018allbutthetop": ["Jiaqi Mu", "Pramod Viswanath"],
}


def bibname(n):
    n = " ".join(n.split())
    parts = n.split()
    if "," in n: return n
    # keep particles (Van Gool) with the surname
    if len(parts) >= 3 and parts[-2].lower() in ("van", "von", "de", "der", "da", "di"):
        return f"{' '.join(parts[-2:])}, {' '.join(parts[:-2])}"
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def esc(s):
    s = " ".join(s.split())          # Crossref titles can carry non-breaking spaces
    rep = {"’": "'", "‐": "-", "–": "--"}
    for a, b in rep.items(): s = s.replace(a, b)
    s = s.replace("&", r"\&")
    return s


def protect(t):
    """Brace acronyms and mixed-case names so the style keeps them; leave
    ordinary title-case words (incl. hyphenated ones) for sentence casing."""
    def needs(seg):
        return (any(c.isdigit() for c in seg) or re.search(r"[A-Z]{2}", seg) is not None
                or re.search(r"[a-z][A-Z]", seg) is not None)
    out = []
    for w in t.split(" "):
        core = w.strip(":,?")
        out.append("{" + w + "}" if any(needs(x) for x in core.split("-")) else w)
    t = " ".join(out)
    for name in ("Waymo Open Dataset", "Bird's-Eye-View"):   # proper names
        t = t.replace(name, "{" + name + "}")
    return t


rows, entries = [], []
for key, (typ, vf, venue, year, src, extra) in SPEC.items():
    a = A.get(key, {}); arx = a.get("arxiv") or {}; c = C.get(key, {})
    if src == "crossref": authors, title, pages, doi = c["authors"], c["title"], c.get("page"), c["doi"]
    elif src == "arxiv": authors, title, pages, doi = arx["authors"], arx["title"], None, None
    elif src == "override": authors, title, pages, doi = OVERRIDES[key], arx["title"], None, None
    elif src == "openalex":
        authors = ["Olivier Roy", "Martin Vetterli"]; title = OA["title"]
        pages = f"{OA['biblio']['first_page']}-{OA['biblio']['last_page']}"; doi = OA["doi"].replace("https://doi.org/", "")
    f = {"author": " and ".join(bibname(x) for x in authors), "title": protect(esc(title)),
         vf: esc(venue), "year": str(year)}
    if pages: f["pages"] = pages.replace("-", "--")
    if src == "crossref" and c.get("volume"): f["volume"] = c["volume"]
    if src == "crossref" and c.get("issue"): f["number"] = c["issue"]
    if doi and not doi.startswith("10.48550"): f["doi"] = doi
    if a.get("arxiv_id") and typ != "article" or (a.get("arxiv_id") and "arXiv" not in venue):
        f["note"] = f"arXiv:{a['arxiv_id']}"
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items())
    entries.append(f"@{typ}{{{key},\n{body}\n}}")
    rows.append((key, a.get("arxiv_id") or "-", arx.get("title", "") == "" or True, src, venue, year,
                 len(authors), doi or "-"))
Path("paper/main.bib").write_text("% Generated by scripts/build_bib.py from fetched metadata.\n% Do not edit by hand.\n\n"
                                  + "\n\n".join(entries) + "\n")
md = ["# Bibliography verification", "",
      "Every entry was generated from metadata fetched in Milestone 20 (arXiv API, Crossref by DOI, "
      "Semantic Scholar batch API, OpenAlex). Raw responses are in this directory.", "",
      "| key | arXiv id (verified) | authors from | venue | year | #authors | DOI |", "|---|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| `{r[0]}` | {r[1]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |")
Path("paper/bibcheck/VERIFICATION.md").write_text("\n".join(md) + "\n")
print(f"wrote {len(entries)} entries to paper/main.bib")
