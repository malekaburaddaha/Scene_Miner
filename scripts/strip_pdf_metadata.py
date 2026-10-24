#!/usr/bin/env python3
"""MILESTONE 30 - rewrite figure PDFs without document metadata.

Matplotlib stamps /Creator, /Producer and /CreationDate into every figure PDF.
Editing them in place (e.g. exiftool) appends an incremental update, so the old
values stay recoverable; this script instead rewrites each file from its pages
with an empty /Info dictionary and no XMP packet. The paper PDFs themselves are
blanked at compile time by \\hypersetup in main_cvpr.tex / supp_cvpr.tex.
"""
import sys
from pathlib import Path
from pypdf import PdfReader, PdfWriter

for p in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "paper/figures").glob("*.pdf")):
    r = PdfReader(str(p)); w = PdfWriter()
    for pg in r.pages: w.add_page(pg)
    w.add_metadata({})
    info = w._info.get_object() if w._info is not None else {}   # pypdf stamps /Producer by default
    for k in list(info.keys()): del info[k]
    tmp = p.with_suffix(".tmp.pdf")
    with open(tmp, "wb") as f: w.write(f)
    meta = PdfReader(str(tmp)).metadata
    tmp.replace(p)
    print(f"{p.name}: metadata now {dict(meta) if meta else {}}")
