#!/usr/bin/env python3
"""MILESTONE 40 - independent audit of every entry in paper/main.bib against live sources.

For each entry, fields are compared with what the publisher or index returns now:
  DOI      -> Crossref (title, authors, container, year, volume, issue, pages)
  arXiv id -> arXiv API (title, authors)
  all      -> Semantic Scholar batch (venue, year,
              external ids, abstract for the context check)
Raw responses go to paper/bibcheck/m40_raw.json; the per-field comparison to
paper/bibcheck/m40_audit.json. Nothing in main.bib is changed here.
"""
import json, re, time, unicodedata, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path

BIB = Path("paper/main.bib"); OUT = Path("paper/bibcheck")
UA = {"User-Agent": "sceneminer-reference-audit/1.0 (research use)"}


def get(url, data=None, headers=None, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={**UA, **(headers or {})})
            with urllib.request.urlopen(req, timeout=40) as r: return r.read().decode("utf-8", "replace")
        except Exception as e:
            err = e; time.sleep(3 * (k + 1))
    return {"error": repr(err)}


def parse_bib(t):
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", t, re.S):
        typ, key, body = m.groups(); f = {"type": typ}
        for fm in re.finditer(r"(\w+)\s*=\s*\{(.*?)\},?\s*\n", body + "\n", re.S): f[fm.group(1)] = fm.group(2)
        out[key] = f
    return out


def norm(s):
    s = re.sub(r"\\[\"'`^~]\{?(\w)\}?", r"\1", s or "").replace(r"{\ss}", "ss")   # LaTeX accents -> base letter
    s = unicodedata.normalize("NFKD", s.replace("ß", "ss")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", re.sub(r"[{}\\]", "", s).lower()).strip()


def sim(a, b): return SequenceMatcher(None, norm(a), norm(b)).ratio()


def surnames(authors):  # "Last, First and Last, First" or list of "First Last"
    if isinstance(authors, str): return [norm(a.split(",")[0]).split()[-1] for a in authors.split(" and ")]
    return [norm(a).split()[-1] if norm(a) else "" for a in authors]


def main():
    bib = parse_bib(BIB.read_text()); raw = {}; rep = {}
    arx = {k: re.search(r"arXiv[: ]*(?:preprint arXiv:)?(\d{4}\.\d{4,5})", f.get("note", "") + " " + f.get("journal", ""))
           for k, f in bib.items()}
    arx = {k: m.group(1) for k, m in arx.items() if m}
    # Semantic Scholar batch: one request for all entries (DOI or arXiv id, else title search later)
    ids = {k: (f"DOI:{f['doi']}" if "doi" in f and not f["doi"].startswith("10.5281") else f"ARXIV:{arx[k]}" if k in arx else None)
           for k, f in bib.items()}
    q = [v for v in ids.values() if v]
    s2 = get("https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,venue,year,authors,externalIds,publicationVenue,abstract",
             data=json.dumps({"ids": q}).encode(), headers={"Content-Type": "application/json"})
    s2 = dict(zip(q, json.loads(s2))) if isinstance(s2, str) else {}
    for key, f in bib.items():
        r = raw[key] = {}; c = rep[key] = {"bib": {k: v for k, v in f.items()}}
        if "doi" in f:
            j = get("https://api.crossref.org/works/" + urllib.parse.quote(f["doi"]))
            if isinstance(j, str) and j.startswith("{"):
                m = json.loads(j)["message"]; r["crossref"] = m
                ct = (m.get("title") or [""])[0]
                c["crossref"] = {"title": ct, "title_sim": round(sim(ct, f["title"]), 3),
                                 "authors": [a.get("family", "") for a in m.get("author", [])],
                                 "container": (m.get("container-title") or [""])[0],
                                 "year": (m.get("published-print") or m.get("published") or m.get("issued", {})).get("date-parts", [[None]])[0][0],
                                 "volume": m.get("volume"), "issue": m.get("issue"), "pages": m.get("page")}
            else:
                # DataCite for non-Crossref DOIs (e.g. Zenodo)
                j = get("https://api.datacite.org/dois/" + urllib.parse.quote(f["doi"]))
                if isinstance(j, str) and j.startswith("{"):
                    a = json.loads(j)["data"]["attributes"]; r["datacite"] = a
                    c["datacite"] = {"title": a["titles"][0]["title"], "title_sim": round(sim(a["titles"][0]["title"], f["title"]), 3),
                                     "authors": [x.get("familyName", x.get("name", "")) for x in a.get("creators", [])],
                                     "year": a.get("publicationYear")}
                else: c["doi_error"] = j
        if key in arx:
            x = get(f"https://export.arxiv.org/api/query?id_list={arx[key]}")
            if isinstance(x, str):
                ns = {"a": "http://www.w3.org/2005/Atom"}; e = ET.fromstring(x).find("a:entry", ns)
                if e is not None and e.find("a:title", ns) is not None:
                    t = " ".join(e.find("a:title", ns).text.split())
                    au = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
                    r["arxiv"] = {"title": t, "authors": au, "published": e.find("a:published", ns).text}
                    c["arxiv"] = {"id": arx[key], "title": t, "title_sim": round(sim(t, f["title"]), 3), "authors": surnames(au),
                                  "published": e.find("a:published", ns).text[:10]}
            time.sleep(3)
        # DBLP is not queried: it answers automated clients with a bot challenge (HTTP 429)
        sid = ids.get(key)
        if sid and s2.get(sid):
            p = s2[sid]; r["s2"] = p
            c["s2"] = {"title": p.get("title"), "venue": p.get("venue"), "year": p.get("year"),
                       "authors": surnames([a["name"] for a in p.get("authors", [])]), "externalIds": p.get("externalIds"),
                       "abstract": p.get("abstract")}
        # author comparison against the best available source
        ba = surnames(f["author"])
        src = c.get("crossref", {}).get("authors") or c.get("arxiv", {}).get("authors") or c.get("datacite", {}).get("authors") \
            or c.get("s2", {}).get("authors")
        if src:
            sa = [norm(x).split()[-1] if norm(x) else "" for x in src]
            c["author_check"] = {"n_bib": len(ba), "n_src": len(sa), "order_match": ba == sa,
                                 "missing_in_bib": [x for x in sa if x not in ba], "extra_in_bib": [x for x in ba if x not in sa]}
        print(f"{key:28s} crossref={c.get('crossref', {}).get('title_sim', c.get('datacite', {}).get('title_sim', '-'))} "
              f"arxiv={c.get('arxiv', {}).get('title_sim', '-')} s2={'s2' in c} "
              f"authors={c.get('author_check', {}).get('order_match', '-')}", flush=True)
    (OUT/"m40_raw.json").write_text(json.dumps(raw, indent=1, ensure_ascii=False))
    (OUT/"m40_audit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
