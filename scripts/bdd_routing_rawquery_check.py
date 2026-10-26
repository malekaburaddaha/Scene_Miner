#!/usr/bin/env python3
"""MILESTONE 23 - BDD100K check of the v2.1, routing-only and v2.2 heads with raw
query features (their training input) and with the L2-normalised features the
Milestone 10 evaluation used. -> results/cvpr/milestone23_bdd_rawquery_routing.json"""
from paths import EXT
import sys, glob, json, numpy as np, torch, torch.nn.functional as F
import os
SEED_GLOB = os.environ.get("SM_SEED_GLOB", "s*")      # Milestone 34: one seed at a time for +/- sd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_external_bdd100k import EVAL_Q, QUESTIONS
from eval_attribute_index import ap_binary
from eval_semantic_edge_cases import text_embed
from scene_mining.filter.sceneminer_v2_2 import SceneMinerV22
from scene_mining.filter.sceneminer_v2 import SceneMinerV21

dev = torch.device("cuda"); root = EXT + "/bdd100k"
labels = json.load(open(root + "/labels.json")); names = sorted(labels); z = np.load(root + "/signals.npz")
TO = ["droplets", "glare", "wet_road", "low_contrast"]; ASK = [k for k, _ in QUESTIONS]
M = z["margin"][:, [ASK.index(k) for k in TO]]; MZ = (M - M.mean(0)) / (M.std(0) + 1e-6)
S = torch.from_numpy(z["siglip"]).float().to(dev); Mt = torch.from_numpy(MZ).float().to(dev)
N = len(names); PR = torch.zeros(N, 26, device=dev)
av2 = torch.stack([torch.ones(N, device=dev), torch.ones(N, device=dev), torch.zeros(N, device=dev)], -1)
QR = text_embed([q for q, _ in EVAL_Q]); QN = F.normalize(QR, dim=-1)


def load(cls, pat):
    out = []
    for c in sorted(glob.glob(pat)):
        d = torch.load(c, map_location=dev, weights_only=False)
        m = cls(d_text=768, n_margins=4, n_meta=6).to(dev); m.load_state_dict(d["state_dict"]); m.eval(); out.append(m)
    return out


# the routing-only build has the v2.1 architecture; it is scored with availability masking
R = {"v21": load(SceneMinerV21, "work_dirs/sceneminer_v21/" + SEED_GLOB + "/final.pt"),
     "v22_routingonly": load(SceneMinerV21, "work_dirs/sceneminer_v22_routingonly/" + SEED_GLOB + "/final.pt"),
     "v22": load(SceneMinerV22, "work_dirs/sceneminer_v22/" + SEED_GLOB + "/final.pt")}


def score(k, m, qv):
    if k == "v22":
        w = m.route(qv, av2); return (w * m.base_parts(qv, qv, S, Mt, PR) * av2).sum(-1), w[0]
    w = torch.softmax(m.W_route(qv) / m.tau_route, -1); p = m.base_parts(qv, qv, S, Mt, PR)
    if k == "v22_routingonly":
        w = w * av2; w = w / w.sum(-1, keepdim=True)
    return (w * p).sum(-1), w[0]


res = {}
for qname, Q in [("raw", QR), ("normalised", QN)]:
    for k, Ms in R.items():
        aps, ws = [], []
        for qi, (qt, (f, pos)) in enumerate(EVAL_Q):
            rel = np.array([labels[n][f] in pos for n in names])
            if rel.sum() in (0, N): continue
            qv = Q[qi].unsqueeze(0).expand(N, -1)
            with torch.no_grad(): out = [score(k, m, qv) for m in Ms]
            s = np.mean([o[0].cpu().numpy() for o in out], 0); ws.append(np.mean([o[1].cpu().numpy() for o in out], 0))
            aps.append(ap_binary(np.argsort(-s), rel))
        res[f"{k}|{qname}"] = {"mAP": float(np.mean(aps)), "mean_routing_siglip_margin_meta": np.mean(ws, 0).round(3).tolist(),
                               "n_seeds": len(Ms)}
        print(qname, k, len(Ms), round(np.mean(aps), 4), np.mean(ws, 0).round(3))
json.dump({"doc": "BDD100K routing-only check, Milestone 23: raw query features (the heads' training input) vs the "
                  "L2-normalised features the Milestone 10 evaluation used", "rows": res},
          open(os.environ.get("SM_OUT", "results/cvpr/milestone23_bdd_rawquery_routing.json"), "w"), indent=1)
