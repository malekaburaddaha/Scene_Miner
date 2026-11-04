#!/usr/bin/env python3
"""MILESTONE 39 - an Overleaf-ready project for the paper and the supplementary.

Run from the repository root. Copies exactly the files the two documents read (style,
bibliography, generated tables, figures) into <out>/SceneMiner_CVPR2027_overleaf/,
renames main_cvpr.tex -> main.tex (Overleaf's default main document) and
supp_cvpr.tex -> supp.tex, writes a README, test-compiles both documents in a clean
copy and zips the project.
  python scripts/build_overleaf.py --out <dir>
"""
import argparse, re, shutil, subprocess, tempfile, zipfile
from pathlib import Path

P = Path("paper")
STATIC = ["cvpr.sty", "ieeenat_fullname.bst", "silence.sty", "cleveref.sty", "main.bib"]
README = """SceneMiner - CVPR 2027 submission (Overleaf project)

main.tex   the paper (8 pages + references), CVPR author kit, review mode
supp.tex   the supplementary material (separate PDF)

Overleaf: upload this zip with New Project > Upload Project. main.tex is the main
document; to build the supplementary, open the project menu, set Main document to
supp.tex and recompile (or download supp.pdf from the original build). Compiler:
pdfLaTeX. Only main.tex uses BibTeX; supp.tex has no references.

Local build:
  pdflatex main && bibtex main && pdflatex main && pdflatex main
  pdflatex supp && pdflatex supp

Before submission: replace \\def\\paperID{*****} in main.tex with the paper ID, and
switch to the official CVPR 2027 author kit when it is released (cvpr.sty,
ieeenat_fullname.bst), then re-check that the main text ends on page 8.

Tables tab_*.tex and supp_tables/*.tex are generated from the result files by the
repository scripts (make_main_tables.py, make_tab_master.py, gate_examples.py); edit
the scripts, not the tables, if numbers change. figures/fig1_main.jpg is the main
architecture figure.
"""


def inputs(tex):
    s = re.sub(r"(?m)%.*$", "", tex.read_text())
    out = set(re.findall(r"\\input\{([^}]+)\}", s))
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s):
        if Path(P/g).suffix: out.add(g)
        else: out.add(next(str(Path(g).with_suffix(e)) for e in (".pdf", ".png", ".jpg") if (P/g).with_suffix(e).exists()))
    return {f if f.endswith((".tex", ".pdf", ".png", ".jpg")) else f + ".tex" for f in out}


def build(proj):
    if proj.exists(): shutil.rmtree(proj)
    proj.mkdir(parents=True)
    files = set(STATIC) | inputs(P/"main_cvpr.tex") | inputs(P/"supp_cvpr.tex")
    for f in sorted(files):
        (proj/f).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(P/f, proj/f)
    shutil.copy2(P/"main_cvpr.tex", proj/"main.tex"); shutil.copy2(P/"supp_cvpr.tex", proj/"supp.tex")
    (proj/"README.txt").write_text(README)
    return sorted(files) + ["main.tex", "supp.tex", "README.txt"]


def test_compile(proj):
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)/"p"; shutil.copytree(proj, t)
        run = lambda *c: subprocess.run(c, cwd=t, capture_output=True, text=True)
        for c in (["pdflatex", "-interaction=nonstopmode", "main"], ["bibtex", "main"],
                  ["pdflatex", "-interaction=nonstopmode", "main"], ["pdflatex", "-interaction=nonstopmode", "main"],
                  ["pdflatex", "-interaction=nonstopmode", "supp"], ["pdflatex", "-interaction=nonstopmode", "supp"]):
            run(*c)
        rep = {}
        for doc in ("main", "supp"):
            log = (t/f"{doc}.log").read_text(errors="ignore")
            rep[doc] = {"pdf": (t/f"{doc}.pdf").exists(), "undefined": len(re.findall(r"undefined", log)),
                        "overfull": log.count("Overfull"), "errors": log.count("\n! ")}
            if (t/f"{doc}.pdf").exists(): shutil.copy2(t/f"{doc}.pdf", proj.parent/f"{doc}.pdf")
        return rep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); proj = out/"SceneMiner_CVPR2027_overleaf"
    files = build(proj)
    print(f"{len(files)} files ->", proj)
    print("test compile:", test_compile(proj))
    z = out/"SceneMiner_CVPR2027_overleaf.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(proj.rglob("*")):
            if f.is_file(): zf.write(f, f.relative_to(proj))
    print(f"{z}: {z.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
