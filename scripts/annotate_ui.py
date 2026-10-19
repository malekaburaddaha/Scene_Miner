#!/usr/bin/env python3
"""TASK 5 - zero-dependency human annotation UI (stdlib http.server only)."""
from __future__ import annotations
import argparse, base64, json, random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

ATTRS = ["droplets", "glare", "wet_road", "low_contrast"]

def find_thumb(thumbs_dir: Path, token: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ""):
        p = thumbs_dir / f"{token}{ext}"
        if p.is_file():
            return p
    # Fallback search if token has slashes/subfolders or prefix mismatch
    matches = list(thumbs_dir.glob(f"*{token}*"))
    return matches[0] if matches else None

def build_sample(n, split_file, flags_file, thumbs, seed=7):
    ev = set(json.load(open(split_file))["eval"])
    rng = random.Random(seed)
    recs = json.load(open(flags_file)) if Path(flags_file).exists() else {}
    thumbs_p = Path(thumbs)
    pool = [t for t in ev if find_thumb(thumbs_p, t) is not None]
    if not recs:
        rng.shuffle(pool); return pool[:n]
    per = max(1, n // (len(ATTRS) * 3))
    chosen, seen = [], set()
    for a in ATTRS:
        have = [t for t in pool if a in recs.get(t, {})]
        pos = [t for t in have if recs[t].get(a) is True]
        neg = [t for t in have if recs[t].get(a) is False]
        bord = sorted(have, key=lambda t: abs(recs[t].get(a + "_margin", 9e9)))
        for grp in (pos, neg, bord):
            if grp is not bord: rng.shuffle(grp)
            for t in grp[:per]:
                if t not in seen: seen.add(t); chosen.append(t)
    rest = [t for t in pool if t not in seen]; rng.shuffle(rest)
    chosen += rest[:max(0, n - len(chosen))]
    return chosen[:n]

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>SceneMiner annotation</title>
<style>
 :root{--bg:#11151b;--fg:#e6e9ee;--card:#1a2029;--on:#d98324;--off:#39424e;--ok:#3f9b6d}
 body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:18px}
 .wrap{max-width:1100px;margin:0 auto}
 img{max-width:100%;min-height:240px;border-radius:6px;background:#000;display:block}
 .row{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}
 .btn{flex:1;min-width:150px;padding:14px;border-radius:6px;background:var(--off);
      text-align:center;cursor:pointer;user-select:none;font-weight:600;border:2px solid transparent}
 .btn.on{background:var(--on);color:#11151b;border-color:#fff}
 .bar{height:6px;background:var(--off);border-radius:3px;overflow:hidden;margin:10px 0}
 .bar>div{height:100%;background:var(--ok)}
 .meta{font:12px ui-monospace,monospace;color:#96a1af}
 kbd{background:#2a323d;border-radius:3px;padding:1px 6px;font:12px ui-monospace,monospace}
</style></head><body><div class="wrap">
<h2 style="margin:0">Scene <span id=i></span> / <span id=n></span> &nbsp;<span class=meta id=tok></span></h2>
<div class=bar><div id=pb></div></div>
<img id=im src="" alt="Loading scene...">
<div class=row id=btns></div>
<div class=meta>
 <kbd>1</kbd> droplets &nbsp; <kbd>2</kbd> glare &nbsp; <kbd>3</kbd> wet_road &nbsp; <kbd>4</kbd> low_contrast
 &nbsp;|&nbsp; <kbd>Enter</kbd> save+next &nbsp; <kbd>b</kbd> back &nbsp; <kbd>u</kbd> unsure &nbsp; <kbd>s</kbd> skip
 &nbsp;|&nbsp; autosaved every scene
</div><div class=meta id=st></div></div>
<script>
const A=%ATTRS%; let cur={}, idx=0, N=0, tok="", busy=false;
function draw(){
 document.getElementById('btns').innerHTML=A.map((a,k)=>
  `<div class="btn ${cur[a]?'on':''}" onclick="tog('${a}')">${k+1}. ${a}${cur[a]?' ✓':''}</div>`).join('');
 document.getElementById('i').textContent=idx+1; document.getElementById('n').textContent=N;
 document.getElementById('tok').textContent=tok;
 document.getElementById('pb').style.width=(100*idx/Math.max(1,N))+'%';
}
function tog(a){cur[a]=!cur[a];draw();}
async function load(d){
 const r=await fetch('/api/get?delta='+d+'&t='+Date.now());
 const j=await r.json();
 idx=j.idx; N=j.n; tok=j.token; cur=j.labels||{};
 document.getElementById('im').src = j.img_b64 || ('/api/img?token='+encodeURIComponent(j.token));
 draw();
 document.getElementById('st').textContent=`labelled ${j.done} of ${j.n}` + (j.img_err ? ` [WARN: ${j.img_err}]` : '');
}
async function save(flag){
 if(busy) return; busy=true;
 try {
  await fetch('/api/put',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({token:tok,labels:cur,flag:flag||null})});
  await load(1);
 } finally { busy=false; }
}
document.addEventListener('keydown',e=>{
 if(e.key>='1'&&e.key<='4'){tog(A[+e.key-1]);e.preventDefault();}
 else if(e.key==='Enter'||e.key===' '){save();e.preventDefault();}
 else if(e.key==='b'){load(-1);} else if(e.key==='u'){save('unsure');}
 else if(e.key==='s'){save('skip');}});
load(0);
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a): pass
    def _j(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query, keep_blank_values=True); S = self.server.state
        if u.path == "/":
            b = HTML.replace("%ATTRS%", json.dumps(ATTRS)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif u.path == "/api/get":
            S["idx"] = max(0, min(len(S["order"]) - 1, S["idx"] + int(q.get("delta", ["0"])[0])))
            t = S["order"][S["idx"]]
            p = find_thumb(Path(S["thumbs"]), t)
            img_b64, img_err = None, None
            if p and p.is_file():
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                img_b64 = f"data:image/jpeg;base64,{b64}"
            else:
                img_err = f"Missing thumbnail for token '{t}' in {S['thumbs']}"
            self._j({"idx": S["idx"], "n": len(S["order"]), "token": t,
                     "labels": S["ann"].get(t, {}).get("labels", {}),
                     "done": len(S["ann"]), "img_b64": img_b64, "img_err": img_err})
        elif u.path == "/api/img":
            raw_tok = unquote(q.get("token", [""])[0])
            p = find_thumb(Path(S["thumbs"]), raw_tok)
            if p and p.is_file():
                b = p.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            else:
                self._j({"error": f"thumbnail not found: {raw_tok}"}, 404)
        else:
            self._j({"error": "not found"}, 404)
    def do_POST(self):
        S = self.server.state
        n = int(self.headers.get("Content-Length", 0))
        d = json.loads(self.rfile.read(n) or b"{}")
        tk = d.get("token")
        if tk:
            S["ann"][tk] = {"labels": {a: bool(d.get("labels", {}).get(a)) for a in ATTRS},
                            "flag": d.get("flag"), "rater": S["rater"]}
            Path(S["out"]).write_text(json.dumps(S["ann"], indent=1))
        self._j({"ok": True, "done": len(S["ann"])})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--split", default="benchmark_cvpr/split_906_602.json")
    ap.add_argument("--flags", default="work_dirs/unified_filter/binary_flags_with_logits.json")
    ap.add_argument("--thumbs", default="work_dirs/unified_filter/scene_thumbs")
    ap.add_argument("--outdir", default="results/overnight_oct18/annotations")
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    out = Path(a.outdir) / f"{a.rater}.json"
    sample_file = Path(a.outdir) / "sample_600.json"
    if sample_file.exists():
        order = json.load(open(sample_file))["tokens"]
    else:
        order = build_sample(a.n, a.split, a.flags, a.thumbs)
        sample_file.write_text(json.dumps({"n": len(order), "tokens": order}, indent=1))
    ann = json.loads(out.read_text()) if out.exists() else {}
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    srv.state = {"order": order, "idx": min(len(ann), max(0, len(order) - 1)), "ann": ann,
                 "out": str(out), "thumbs": a.thumbs, "rater": a.rater}
    import socket
    print(f"[annotate] rater={a.rater}  {len(order)} scenes  (resuming at {len(ann)})")
    print(f"[annotate] open:  http://{socket.gethostname()}:{a.port}/   (or http://localhost:{a.port}/)")
    print(f"[annotate] saving to {out}")
    srv.serve_forever()

if __name__ == "__main__":
    main()
