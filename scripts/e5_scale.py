#!/usr/bin/env python3
"""MILESTONE 41 - E5: scale of s_head and s_abst before the Eq. (1) mixture.

1. Decomposition check: g * s_head + (1 - g) * s_abst reproduces the v2.6 forward
   pass to float precision (src/scene_mining/filter/sceneminer_poststd.py).
2. Empirical mean and sd of s_head and s_abst over the 906 training scenes, for
     (a) the 72 Stage-2 training queries x 906 training scenes (pooled, and the mean
         within-query sd), full mode, five seeds;
     (b) the 4 human-GT degradation queries and the 150 suite queries x 906 scenes.
   gamma_deg per query is reported with them.
3. Writes the post-standardised family work_dirs/sceneminer_v26ps/ (the trained v2.6
   checkpoints flagged post_std); evaluate it with
     SM_FINAL=v26ps SM_METHODS=sceneminer_portable,sceneminer scripts/eval_related_work_baselines.py
-> results/cvpr/milestone41_e5_scale.json
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from paths import DEVICE
from eval_semantic_edge_cases import text_embed, load_v23, training_reference
from eval_human_gt_full import ATTRS, QUERY
from e2_build_training_free import training_queries
from scene_mining.filter.sceneminer_poststd import SceneMinerPostStd

dev = torch.device(DEVICE)


def main():
    texts, _, z = training_queries()
    suite = [q["text"] for q in json.load(open("benchmark_cvpr/queries_390.json"))["eval"]]
    human = [QUERY[a] for a in ATTRS]
    M = load_v23(z["risk"].reshape(-1), which="v26"); [setattr(m, "use_abstain", True) for m in M]
    full, _ = training_reference(); N = full[0].shape[0]; on = torch.ones(N, device=dev)
    out = {"doc": __doc__, "n_training_scenes": N, "sets": {}}
    # 1. decomposition check on the training scenes, seed 0, 8 queries
    m0 = M[0]; ps = SceneMinerPostStd.__new__(SceneMinerPostStd); ps.__dict__ = m0.__dict__
    QT = text_embed(texts[:8]); err = 0.0
    with torch.no_grad():
        for q in QT:
            qv = q.unsqueeze(0).expand(N, -1)
            h, a = SceneMinerPostStd._head_abst_full(ps, qv, *full, has_3d=on); _, g, _ = m0.concept(qv)
            err = max(err, float((g * h + (1 - g) * a - m0(qv, qv, *full, has_3d=on)).abs().max()))
    out["decomposition_max_abs_error"] = err
    print(f"[E5] decomposition max |error| {err:.2e}", flush=True)
    for name, Q in [("stage2_training_queries", texts), ("human_gt_queries", human), ("suite_eval_queries", suite)]:
        QT = text_embed(Q); H, A, G = [], [], []
        with torch.no_grad():
            for m in M:
                ps.__dict__ = m.__dict__
                for q in QT:
                    qv = q.unsqueeze(0).expand(N, -1)
                    h, a = SceneMinerPostStd._head_abst_full(ps, qv, *full, has_3d=on)
                    H.append(h.cpu().numpy()); A.append(a.cpu().numpy()); G.append(float(m.concept(qv[:1])[1]))
        H, A, G = np.array(H), np.array(A), np.array(G)
        r = {"n_queries": len(Q), "n_pairs_per_seed": len(Q) * N,
             "s_head": {"mean": float(H.mean()), "sd_pooled": float(H.std()), "sd_within_query_mean": float(H.std(1).mean()),
                        "sd_within_query_range": [float(H.std(1).min()), float(H.std(1).max())]},
             "s_abst": {"mean": float(A.mean()), "sd_pooled": float(A.std()), "sd_within_query_mean": float(A.std(1).mean()),
                        "sd_within_query_range": [float(A.std(1).min()), float(A.std(1).max())]},
             "gamma_deg": {"mean": float(G.mean()), "share_gt_0.9": float((G > 0.9).mean()), "share_lt_0.1": float((G < 0.1).mean())}}
        out["sets"][name] = r
        print(f"[E5] {name}: s_head mean {r['s_head']['mean']:+.3f} sd {r['s_head']['sd_pooled']:.3f} (within-query {r['s_head']['sd_within_query_mean']:.3f}); "
              f"s_abst mean {r['s_abst']['mean']:+.3f} sd {r['s_abst']['sd_pooled']:.3f} (within-query {r['s_abst']['sd_within_query_mean']:.3f}); "
              f"gamma>0.9 {r['gamma_deg']['share_gt_0.9']:.2f} <0.1 {r['gamma_deg']['share_lt_0.1']:.2f}", flush=True)
    for c in sorted(Path("work_dirs/sceneminer_v26").glob("s*/final.pt")):
        d = torch.load(c, map_location="cpu", weights_only=False); d["post_std"] = "per_query_training_scenes"
        o = Path("work_dirs/sceneminer_v26ps")/c.parent.name; o.mkdir(parents=True, exist_ok=True); torch.save(d, o/"final.pt")
    Path("results/cvpr/milestone41_e5_scale.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
