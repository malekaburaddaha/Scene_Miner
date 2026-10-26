#!/usr/bin/env python3
"""MILESTONE 11 - targeted CAM_FRONT degradation vs 3D detection error, val150.

MILESTONE 8 asked whether any ranker retrieves scenes the detector handles worse
and found nothing: every delta was under 0.07 sigma. That test averaged scene
mAP over all six cameras and the full 360 degrees, which dilutes a front-camera
effect by roughly a factor of six. This milestone makes the test TARGETED in
both directions at once:

  * degradation is read from the CAM_FRONT keyframe only (the Qwen2.5-VL margins
    on the cached val150 thumbnails, which are CAM_FRONT by construction);
  * error is measured only over objects inside the CAM_FRONT field of view,
    computed from that camera's own intrinsics, in the ego frame.

Metrics, all restricted to the front sector:
  * FP rate at score > 0.3 - the operating point a mining pipeline would use;
  * mATE / mAOE - official nuScenes TP metrics at the 2 m threshold;
  * recall beyond 30 m - where camera degradation should bite hardest.

AP and the TP metrics use the official `accumulate` / `calc_ap` / `calc_tp`.
FP rate and far recall need a fixed score operating point, which the official
metrics integrate over, so they use explicit greedy centre-distance matching at
the same 2 m threshold and in the same descending-confidence order the devkit
uses. Uncertainty is 1000-resample bootstrap over SCENES, with paired
permutation tests between the degraded and clean partitions.

These 150 scenes are nuScenes val: bevformer_base never trained on them
(per-scene mAP 0.4514 here versus 0.7864 on the memorised train scenes), so
nothing in this analysis is contaminated.
"""
from __future__ import annotations
from paths import QWEN_VL
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")

QUESTIONS = [
 ("wet_road", "Is the road surface visibly wet or reflective from water? Answer only yes or no."),
 ("glare", "Is there strong light glare, bloom, or lens flare from headlights, streetlights, or the sun? Answer only yes or no."),
 ("droplets", "Are there visible water droplets or rain smear on the camera lens? Answer only yes or no."),
 ("low_contrast", "Are object boundaries severely obscured by darkness, fog, or bloom such that vehicles within 30 m are barely distinguishable from the background? Answer only yes or no."),
]
ATTRS = [k for k, _ in QUESTIONS]


def quat_to_yaw_inv_rot(q):
    """Inverse rotation of an ego pose quaternion (w,x,y,z), as a 2x2 yaw matrix."""
    w, x, y, z = q
    yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([[c, -s], [s, c]])


def extract_margins(nusc, thumbs, qwen_path, cache, dev="cuda"):
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        print(f"[cf] loaded cached CAM_FRONT margins {cache}", flush=True)
        return [str(t) for t in z["scenes"]], z["margin"]
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    scenes = sorted(p.stem for p in Path(thumbs).glob("*.jpg"))
    proc = AutoProcessor.from_pretrained(qwen_path)
    qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_path, torch_dtype=torch.bfloat16).to(dev).eval()
    tok = proc.tokenizer
    YES = sorted({tok.encode(s, add_special_tokens=False)[0] for s in
                  ["Yes","yes"," Yes"," yes"] if len(tok.encode(s, add_special_tokens=False))==1})
    NO = sorted({tok.encode(s, add_special_tokens=False)[0] for s in
                 ["No","no"," No"," no"] if len(tok.encode(s, add_special_tokens=False))==1})
    M = np.zeros((len(scenes), len(QUESTIONS)), dtype=np.float32)
    for i, sc in enumerate(scenes):
        img = Image.open(Path(thumbs)/f"{sc}.jpg").convert("RGB")
        for j, (_, q) in enumerate(QUESTIONS):
            msgs = [{"role":"user","content":[{"type":"image","image":img},
                                              {"type":"text","text":q}]}]
            t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[t], images=[img], return_tensors="pt").to(dev)
            with torch.no_grad():
                g = qm.generate(**inp, max_new_tokens=1, do_sample=False,
                                output_scores=True, return_dict_in_generate=True)
            lp = torch.log_softmax(g.scores[0][0].float(), -1)
            M[i, j] = float(torch.logsumexp(lp[YES],0) - torch.logsumexp(lp[NO],0))
        if (i+1) % 25 == 0: print(f"  [cf] qwen {i+1}/{len(scenes)}", flush=True)
    del qm; torch.cuda.empty_cache()
    np.savez_compressed(cache, scenes=np.array(scenes), margin=M)
    print(f"[cf] cached -> {cache}", flush=True)
    return scenes, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json",
                    default="work_dirs/bevformer_out/official_eval/pts_bbox/results_nusc.json")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--thumbs", default="work_dirs/val150_thumbs")
    ap.add_argument("--qwen", default=QWEN_VL)
    ap.add_argument("--cache", default="work_dirs/val150_camfront_signals.npz")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--far", type=float, default=30.0)
    ap.add_argument("--dist-th", type=float, default=2.0)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results/cvpr/milestone11_camfront_errors.json")
    a = ap.parse_args()

    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.common.loaders import (load_prediction, load_gt,
                                              add_center_dist, filter_eval_boxes)
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.common.data_classes import EvalBoxes

    cfg = config_factory("detection_cvpr_2019")
    nusc = NuScenes(version=a.version, dataroot=a.dataroot, verbose=False)
    scenes, MARG = extract_margins(nusc, a.thumbs, a.qwen, Path(a.cache))
    print(f"[cf] {len(scenes)} val scenes with CAM_FRONT margins", flush=True)

    # CAM_FRONT half-FOV from the camera's own intrinsics
    s0 = nusc.get("scene", scenes[0])
    sd = nusc.get("sample_data", nusc.get("sample", s0["first_sample_token"])["data"]["CAM_FRONT"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    fx = float(np.array(cs["camera_intrinsic"])[0, 0]); Wpx = float(sd["width"])
    half_fov = float(np.arctan(Wpx / (2.0 * fx)))
    print(f"[cf] CAM_FRONT half-FOV {np.degrees(half_fov):.1f} deg "
          f"(fx={fx:.1f}, width={Wpx:.0f})", flush=True)

    pred_boxes, _ = load_prediction(a.results_json, cfg.max_boxes_per_sample,
                                    DetectionBox, verbose=False)
    gt_boxes = load_gt(nusc, "val", DetectionBox, verbose=False)
    pred_boxes = add_center_dist(nusc, pred_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)
    pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, verbose=False)
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=False)

    # ---- restrict every box to the CAM_FRONT sector, in the ego frame --------
    rot = {}
    def front_only(boxes):
        out = EvalBoxes()
        for st in boxes.sample_tokens:
            if st not in rot:
                sdr = nusc.get("sample_data", nusc.get("sample", st)["data"]["LIDAR_TOP"])
                pose = nusc.get("ego_pose", sdr["ego_pose_token"])
                rot[st] = quat_to_yaw_inv_rot(pose["rotation"])
            R = rot[st]
            keep = []
            for b in boxes[st]:
                v = R @ np.array(b.ego_translation[:2])
                if abs(np.arctan2(v[1], v[0])) <= half_fov: keep.append(b)
            if keep: out.add_boxes(st, keep)
        return out

    scene_of = {st: nusc.get("sample", st)["scene_token"] for st in gt_boxes.sample_tokens}
    by_scene = defaultdict(list)
    for st, sc in scene_of.items(): by_scene[sc].append(st)
    GTF, PRF = front_only(gt_boxes), front_only(pred_boxes)
    ngt_f = sum(len(GTF[t]) for t in GTF.sample_tokens)
    ngt_a = sum(len(gt_boxes[t]) for t in gt_boxes.sample_tokens)
    print(f"[cf] GT in front sector: {ngt_f}/{ngt_a} ({100*ngt_f/ngt_a:.1f} %)", flush=True)

    def subset(boxes, toks):
        e = EvalBoxes()
        for t in toks:
            if t in boxes.boxes and boxes[t]: e.add_boxes(t, boxes[t])
        return e

    def greedy(gp, pp, toks, score_thr, far):
        """FP rate at a fixed score, and recall split by range. Greedy matching in
        descending confidence against nearest unmatched same-class GT within
        dist_th, which is the devkit's own matching rule."""
        fp = tp = 0; n_far = m_far = n_near = m_near = 0
        for t in toks:
            G = [b for b in (gp[t] if t in gp.boxes else [])]
            P = sorted([b for b in (pp[t] if t in pp.boxes else [])
                        if b.detection_score >= score_thr],
                       key=lambda b: -b.detection_score)
            taken = set()
            for p in P:
                best, bi = a.dist_th, None
                for i, g in enumerate(G):
                    if i in taken or g.detection_name != p.detection_name: continue
                    d = float(np.linalg.norm(np.array(g.translation[:2])
                                             - np.array(p.translation[:2])))
                    if d < best: best, bi = d, i
                if bi is None: fp += 1
                else: taken.add(bi); tp += 1
            for i, g in enumerate(G):
                r = float(np.linalg.norm(np.array(g.ego_translation[:2])))
                if r > far:
                    n_far += 1; m_far += i in taken
                else:
                    n_near += 1; m_near += i in taken
        npred = tp + fp
        return {"n_pred": npred, "fp": fp, "tp": tp,
                "fp_rate": fp / npred if npred else float("nan"),
                "recall_far": m_far / n_far if n_far else float("nan"),
                "recall_near": m_near / n_near if n_near else float("nan"),
                "n_gt_far": n_far, "n_gt_near": n_near}

    # ---- per-scene front-sector metrics -------------------------------------
    per = {}
    for i, sc in enumerate(scenes):
        toks = by_scene.get(sc, [])
        gp, pp = subset(GTF, toks), subset(PRF, toks)
        aps, ates, aoes = [], [], []
        for cls in cfg.class_names:
            ngt_c = sum(1 for t in toks if t in gp.boxes
                        for b in gp[t] if b.detection_name == cls)
            if ngt_c == 0: continue
            aps.append(float(np.mean([calc_ap(accumulate(gp, pp, cls, cfg.dist_fcn_callable, d),
                                              cfg.min_recall, cfg.min_precision)
                                      for d in cfg.dist_ths])))
            md = accumulate(gp, pp, cls, cfg.dist_fcn_callable, cfg.dist_th_tp)
            ates.append(calc_tp(md, cfg.min_recall, "trans_err"))
            aoes.append(calc_tp(md, cfg.min_recall, "orient_err"))
        g = greedy(gp, pp, toks, a.score_thr, a.far)
        per[sc] = {"mAP_front": float(np.mean(aps)) if aps else float("nan"),
                   "mATE_front": float(np.mean(ates)) if ates else float("nan"),
                   "mAOE_front": float(np.mean(aoes)) if aoes else float("nan"),
                   **g}
        if (i+1) % 25 == 0: print(f"  [cf] {i+1}/{len(scenes)} scenes", flush=True)

    # ---- degraded vs clean, per attribute -----------------------------------
    METS = ["fp_rate", "mATE_front", "mAOE_front", "recall_far", "mAP_front", "recall_near"]
    rng = np.random.default_rng(0)
    out = {"n_scenes": len(scenes), "score_thr": a.score_thr, "far_m": a.far,
           "dist_th_m": a.dist_th, "camfront_half_fov_deg": float(np.degrees(half_fov)),
           "gt_front_fraction": ngt_f / ngt_a, "boot": a.boot,
           "contaminated": False, "per_scene": per, "partitions": {}}
    labels = {at: MARG[:, j] > 0 for j, at in enumerate(ATTRS)}
    labels["any_degradation"] = np.any(MARG > 0, axis=1)
    for name, pos in labels.items():
        A = [scenes[i] for i in range(len(scenes)) if pos[i]]
        B = [scenes[i] for i in range(len(scenes)) if not pos[i]]
        rec = {"n_degraded": len(A), "n_clean": len(B), "metrics": {}}
        for mt in METS:
            va = np.array([per[s][mt] for s in A], float)
            vb = np.array([per[s][mt] for s in B], float)
            va, vb = va[~np.isnan(va)], vb[~np.isnan(vb)]
            if len(va) < 3 or len(vb) < 3: continue
            obs = float(va.mean() - vb.mean())
            pooled = np.concatenate([va, vb])
            # bootstrap CI on the difference, resampling scenes within each group
            bs = [va[rng.integers(0,len(va),len(va))].mean()
                  - vb[rng.integers(0,len(vb),len(vb))].mean() for _ in range(a.boot)]
            # permutation test: reshuffle the degraded/clean assignment
            cnt = 0
            for _ in range(a.boot):
                p_ = rng.permutation(pooled)
                if abs(p_[:len(va)].mean() - p_[len(va):].mean()) >= abs(obs): cnt += 1
            sd = pooled.std() + 1e-12
            rec["metrics"][mt] = {
                "degraded": float(va.mean()), "clean": float(vb.mean()),
                "delta": obs, "delta_sigma": obs / sd,
                "delta_CI95": [float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))],
                "perm_p": (cnt + 1) / (a.boot + 1)}
        out["partitions"][name] = rec

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for name, rec in out["partitions"].items():
        print(f"\n=== {name}: {rec['n_degraded']} degraded vs {rec['n_clean']} clean ===")
        print(f"  {'metric':<14}{'degraded':>10}{'clean':>10}{'delta':>10}"
              f"{'sigma':>8}{'95% CI':>22}{'p':>8}")
        for mt, m in rec["metrics"].items():
            ci = f"[{m['delta_CI95'][0]:+.3f},{m['delta_CI95'][1]:+.3f}]"
            star = " *" if m["perm_p"] < 0.05 else ""
            print(f"  {mt:<14}{m['degraded']:>10.4f}{m['clean']:>10.4f}{m['delta']:>+10.4f}"
                  f"{m['delta_sigma']:>+8.2f}{ci:>22}{m['perm_p']:>8.3f}{star}")
    print(f"\n[cf] wrote {a.out}")


if __name__ == "__main__":
    main()
