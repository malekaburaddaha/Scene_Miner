#!/usr/bin/env python3
"""MILESTONE 30 - double-blind leak check for the submission PDFs and sources.

For every PDF: the document-info dictionary, the XMP packet, the extracted text of
every page, and every decompressed content/object stream are searched for
identifying tokens as whole words (raw `strings` on compressed PDFs yields random
byte matches such as "eumd"). JPEG/PNG figure files are checked with their EXIF/XMP
metadata. LaTeX sources and generated tables are searched as text.
Exit status 1 if anything matches.
"""
import re, subprocess, sys, zlib
from pathlib import Path
from pypdf import PdfReader

TOKENS = ["umd", "umich", "srawa", "malek", "abdmalek", "aburaddaha", "greatlakes", "turbo", "arc-ts",
          "maryland", "michigan"]
PAT = re.compile(r"(?<![A-Za-z])(" + "|".join(map(re.escape, TOKENS)) + r")(?![A-Za-z])", re.I)
PATHS = re.compile(r"/nfs/|/home/|/scratch/|\bgl\d{4}\b", re.I)


def hits(text):
    return sorted({m.group(0) for m in PAT.finditer(text)} | {m.group(0) for m in PATHS.finditer(text)})


def check_pdf(p):
    r = PdfReader(str(p)); found = {}
    meta = {k: str(v) for k, v in (r.metadata or {}).items()}
    if h := hits(" ".join(meta.values())): found["info"] = h
    xmp = r.trailer["/Root"].get("/Metadata")
    if xmp is not None and (h := hits(xmp.get_object().get_data().decode("latin-1"))): found["xmp"] = h
    for i, pg in enumerate(r.pages):
        if h := hits(pg.extract_text() or ""): found[f"text p{i+1}"] = h
    raw = p.read_bytes()
    for m in re.finditer(rb"stream\r?\n", raw):
        start = m.end(); end = raw.find(b"endstream", start)
        try: data = zlib.decompress(raw[start:end])
        except Exception: continue
        if h := hits(data.decode("latin-1")): found.setdefault("streams", set()).update(h)
    return meta, found


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "paper")
    bad = False
    for p in [root/"main_cvpr.pdf", root/"supp_cvpr.pdf", *sorted((root/"figures").glob("*.pdf"))]:
        meta, f = check_pdf(p)
        print(f"{'LEAK ' if f else 'ok   '} {p}  info={ {k: v for k, v in meta.items() if v} }  {f if f else ''}")
        bad |= bool(f)
    for p in sorted((root/"figures").glob("*.jp*g")) + sorted((root/"figures").glob("*.png")):
        out = subprocess.run(["exiftool", str(p)], capture_output=True, text=True).stdout
        h = hits(out); print(f"{'LEAK ' if h else 'ok   '} {p} exif {h if h else ''}"); bad |= bool(h)
    for p in sorted(root.glob("*.tex")) + sorted((root/"supp_tables").glob("*.tex")) + [root/"main.bib"]:
        h = hits(p.read_text(errors="ignore")); print(f"{'LEAK ' if h else 'ok   '} {p} {h if h else ''}"); bad |= bool(h)
    print("ANONYMITY CHECK", "FAILED" if bad else "PASSED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
