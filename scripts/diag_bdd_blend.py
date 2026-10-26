"""Why does a convex blend of two signals score BELOW both of them?"""
from paths import EXT
import glob, json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_attribute_index import ap_binary
from eval_external_bdd100k import EVAL_Q, QUESTIONS, CHANNEL

root = EXT + "/bdd100k"
labels = json.load(open(f"{root}/labels.json")); names = sorted(labels)
z = np.load(f"{root}/signals.npz", allow_pickle=False)
SIG, MARG = z["siglip"], z["margin"]
MZ = (MARG - MARG.mean(0, keepdims=True)) / (MARG.std(0, keepdims=True) + 1e-6)
ATT = [k for k, _ in QUESTIONS]
dev = torch.device("cuda")

from transformers import AutoModel, GemmaTokenizerFast
P = "google/siglip2-base-patch16-224"
sm = AutoModel.from_pretrained(P, use_safetensors=True).to(dev).eval()
tk = GemmaTokenizerFast.from_pretrained(P)
ft = lambda o: (o if torch.is_tensor(o) else o.pooler_output).float()
with torch.no_grad():
    ti = tk([q for q,_ in EVAL_Q], padding="max_length", truncation=True,
            max_length=64, return_tensors="pt").to(dev)
    QT = F.normalize(ft(sm.get_text_features(**ti)), dim=-1)
del sm; torch.cuda.empty_cache()

from scene_mining.filter.sceneminer_v2_2 import SceneMinerV22
R = []
for c in sorted(glob.glob("work_dirs/sceneminer_v22/s*/final.pt")):
    d = torch.load(c, map_location=dev, weights_only=False)
    m = SceneMinerV22(d_text=QT.shape[1], n_margins=4, n_meta=6).to(dev)
    m.load_state_dict(d["state_dict"]); m.eval(); R.append(m)

S = torch.from_numpy(SIG).float().to(dev); M = torch.from_numpy(MZ).float().to(dev)
PR = torch.zeros(len(names), 26, device=dev)
N = len(names)
av2 = torch.stack([torch.ones(N,device=dev), torch.ones(N,device=dev),
                   torch.zeros(N,device=dev)], -1)

acc = {}
scale_rows = []
sel_rows = []
for qi, (qt, (field, pos)) in enumerate(EVAL_Q):
    rel = np.array([labels[n][field] in pos for n in names])
    if rel.sum() in (0, N): continue
    qv = QT[qi].unsqueeze(0).expand(N, -1)
    with torch.no_grad():
        parts = torch.stack([m.base_parts(qv, qv, S, M, PR) for m in R]).mean(0)  # (N,3)
        w = torch.stack([m.route(qv, av2).mean(0) for m in R]).mean(0)            # (3,)
        wm = torch.stack([torch.softmax(m.margin_select(qv), -1).mean(0) for m in R]).mean(0)
    p = parts.cpu().numpy(); ww = w.cpu().numpy()
    ap = lambda s: ap_binary(np.argsort(-s), rel)
    # 1. each part alone
    acc.setdefault("part0_siglip_cos", []).append(ap(p[:,0]))
    acc.setdefault("part1_s_marg(learned select)", []).append(ap(p[:,1]))
    # 2. the blend as shipped
    acc.setdefault("blend_as_shipped", []).append(ap((ww*p*av2.cpu().numpy()).sum(-1)))
    # 3. the same blend after per-query z-scoring of each part (scale-matched)
    pz = (p - p.mean(0)) / (p.std(0) + 1e-9)
    acc.setdefault("blend_z_scored", []).append(ap((ww*pz*av2.cpu().numpy()).sum(-1)))
    # 4. a priori channel instead of the learned select
    att, sgn = CHANNEL[qt]; j = ATT.index(att)
    acc.setdefault("apriori_margin_alone", []).append(ap(sgn*MZ[:,j]))
    pz_ap = pz.copy(); pz_ap[:,1] = (sgn*MZ[:,j] - (sgn*MZ[:,j]).mean())/(sgn*MZ[:,j]).std()
    acc.setdefault("blend_z_scored_apriori_channel", []).append(ap((ww*pz_ap*av2.cpu().numpy()).sum(-1)))
    scale_rows.append([p[:,0].std(), p[:,1].std(), abs(p[:,0]).mean(), abs(p[:,1]).mean()])
    sel_rows.append((qt, att, ATT[int(np.argmax(wm.cpu().numpy()))],
                     float(wm.max()), ww.tolist()))

print(f"\n{'variant':<38}{'mAP':>8}")
for k, v in sorted(acc.items(), key=lambda kv: -np.mean(kv[1])):
    print(f"{k:<38}{np.mean(v):>8.4f}")

A = np.array(scale_rows).mean(0)
print(f"\nSCALE MISMATCH (mean over queries)")
print(f"  std(siglip cos part) = {A[0]:.4f}   mean|.| = {A[2]:.4f}")
print(f"  std(s_marg part)     = {A[1]:.4f}   mean|.| = {A[3]:.4f}")
print(f"  -> s_marg carries {A[1]/max(A[0],1e-9):.1f}x the spread of the cosine part,")
print(f"     so a 50/50 ROUTING weight is not a 50/50 influence.")

print(f"\nLEARNED margin_select vs the a-priori channel")
print(f"  {'query':<52}{'a-priori':<14}{'learned argmax':<14}{'p_max':>7}")
nmis = 0
for qt, att, lrn, pmax, ww in sel_rows:
    hit = "" if att == lrn else "  MISMATCH"
    nmis += att != lrn
    print(f"  {qt[:50]:<52}{att:<14}{lrn:<14}{pmax:>7.3f}{hit}")
print(f"  -> learned channel disagrees with the a-priori one on {nmis}/{len(sel_rows)} queries")
