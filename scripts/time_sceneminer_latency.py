#!/usr/bin/env python3
"""MILESTONE 27 - SceneMiner latency, online and offline, on one A40.

Online (per query, 1508 cached scenes, query text embedding cached, median over
20 benchmark queries):
  full      prepare_query (gate, routing, FiLM parameters: query-only work, done once
            per query) + rank (per-scene work) replayed as a CUDA graph; one model and
            the 5-seed ensemble; and the reference forward pass for comparison
  portable  the routed base score with abstention (no residual branch)
Offline (Stage 1, once per scene):
  multi-camera path  one 6-camera keyframe through the Stage-1 model (SigLIP2 patches,
                     BEVFormer, motion head, Q-Former, LoRA-Qwen, heads), bf16 autocast
                     as in scripts/export_filter_features_scaled.py, batch 1
  monocular path     SigLIP2 ViT-B/16 image embedding; four Qwen2.5-VL yes/no margins
-> results/cvpr/milestone27_latency.json
"""
from paths import DEVICE
from paths import QWEN_VL
import json, os, sys, time
from types import SimpleNamespace
import numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from eval_semantic_edge_cases import load_v23, text_embed
from scene_mining.filter.fast_index import precompute_scene_index, prepare_query, rank, GraphRanker, fast_base_score, portable_index

dev = torch.device(DEVICE); W = "work_dirs/unified_filter"


def timeit(fn, n=30, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(1000 * (time.perf_counter() - a))
    return float(np.median(ts))


def online():
    z = np.load(f"{W}/features_v2.npz"); T = lambda k: torch.from_numpy(z[k]).float().to(dev)
    Fe = (T("emb"), T("tag_logits"), T("risk"), torch.load(f"{W}/V_frozen_cache.pt", map_location=dev).float(),
          T("siglip_img"), T("vlm_margin_z"), T("prior"))
    M = load_v23(z["risk"].reshape(-1), which=os.environ.get("SM_FINAL", "v26"))
    for m in M: m.use_abstain = True
    idxs = [precompute_scene_index(m, *Fe) for m in M]
    N = Fe[0].shape[0]
    qs = [q["text"] for q in json.load(open("benchmark_cvpr/queries_semantic_edge_cases.json"))["queries"][:10]] + \
         [q["text"] for q in json.load(open("benchmark_cvpr/queries_390.json"))["eval"][:10]]
    QE = text_embed(qs)
    PRm = Fe[6].clone(); PRm[:, :20] = 0
    av = torch.stack([torch.ones(N, device=dev), torch.ones(N, device=dev), torch.zeros(N, device=dev)], -1)
    rec = {k: [] for k in ["prepare_1", "rank_graph_1", "rank_eager_1", "full_1", "full_5", "reference_1", "portable_1",
                           "portable_reference_1"]}
    G = None
    with torch.no_grad():
        for q in QE:
            Ps = [prepare_query(m, q) for m in M]
            if G is None: G = [GraphRanker(m, i_, p_) for m, i_, p_ in zip(M, idxs, Ps)]
            rec["prepare_1"].append(timeit(lambda: prepare_query(M[0], q), 10))
            rec["rank_graph_1"].append(timeit(lambda: G[0](Ps[0])))
            rec["rank_eager_1"].append(timeit(lambda: rank(M[0], idxs[0], Ps[0]), 10))
            rec["full_1"].append(timeit(lambda: G[0](prepare_query(M[0], q)), 10))
            rec["full_5"].append(timeit(lambda: [g_(prepare_query(m, q)) for g_, m in zip(G, M)], 5))
            rec["reference_1"].append(timeit(lambda: M[0](q.expand(N, -1), q.expand(N, -1), *Fe), 5))
            pidx = portable_index(M[0], Fe[4], Fe[5], PRm)
            rec["portable_1"].append(timeit(lambda: fast_base_score(M[0], Fe[4], Fe[5], PRm, q, idx=pidx), 10))
            rec["portable_reference_1"].append(timeit(lambda: M[0].base_score(q.expand(N, -1), Fe[4], Fe[5], PRm, av), 5))
            d = float((G[0](Ps[0]) - M[0](q.expand(N, -1), q.expand(N, -1), *Fe)).abs().max())
            rec.setdefault("max_abs_diff_graph_vs_reference", []).append(d)
    out = {k: float(np.median(v)) for k, v in rec.items() if k != "max_abs_diff_graph_vs_reference"}
    out["max_abs_diff_graph_vs_reference"] = float(max(rec["max_abs_diff_graph_vs_reference"]))
    return out


def offline():
    res = {}
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from eval_external_bdd100k import QUESTIONS
    img = Image.open(sorted(__import__("glob").glob("work_dirs/unified_filter/scene_thumbs/*.jpg"))[0]).convert("RGB")
    m = AutoModel.from_pretrained("google/siglip2-base-patch16-224", use_safetensors=True).to(dev).eval()
    ip = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
    with torch.no_grad():
        res["siglip2_image_ms"] = timeit(lambda: m.get_image_features(**ip(images=[img], return_tensors="pt").to(dev)), 20)
    del m; torch.cuda.empty_cache()
    Q = QWEN_VL
    proc = AutoProcessor.from_pretrained(Q); qm = Qwen2_5_VLForConditionalGeneration.from_pretrained(Q, torch_dtype=torch.bfloat16).to(dev).eval()
    def margins():
        for _, q in QUESTIONS:
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
            t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[t], images=[img], return_tensors="pt").to(dev)
            qm.generate(**inp, max_new_tokens=1, do_sample=False)
    with torch.no_grad():
        res["qwen_four_margins_ms"] = timeit(margins, 5, 1)
    del qm; torch.cuda.empty_cache()
    # multi-camera Stage-1 model
    try:
        from mmengine.config import Config
        from torch.utils.data import DataLoader
        from train_wacv import build_model, load_ckpt, forward_model, TOKENIZER
        from scene_mining.data import WaymoUniversalDataset, universal_collate
        cfg = Config.fromfile("configs/experiment/nuscenes_scenemining_v8_unified.py")
        margs = SimpleNamespace(use_lora=True, freeze_motion=False,
                                base_checkpoint="work_dirs/nuscenes_scenemining_v4_motion_mask/epoch_1.pth",
                                v8_checkpoint="work_dirs/nuscenes_scenemining_v8_unified/epoch_1_trainable.pth")
        model, _ = build_model(cfg, dev, margs, rank=0)
        load_ckpt(model, None, None, "work_dirs/wacv_unified_lora_fm/wacv_epoch2.pth", rank=0); model.eval()
        ds = WaymoUniversalDataset("work_dirs/waymo_zeroshot/extracted_large", tokenizer_path=TOKENIZER)
        batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=universal_collate)))
        def fwd():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                forward_model(model, batch, dev)
        res["stage1_multicamera_keyframe_ms"] = timeit(fwd, 20, 3)
    except Exception as e:
        res["stage1_multicamera_keyframe_ms"] = None; res["stage1_error"] = repr(e)[:300]
    return res


if __name__ == "__main__":
    on = online(); print("online", on, flush=True)
    off = offline(); print("offline", off, flush=True)
    out = {"doc": __doc__, "online": on, "offline": off,
           "SceneMiner": {"ms": on["full_1"], "what": "prepare_query + CUDA-graph rank, one model"},
           "SceneMiner (portable)": {"ms": on["portable_1"], "what": "portable base score, one model"}}
    json.dump(out, open("results/cvpr/milestone27_latency.json", "w"), indent=1)
    print("wrote results/cvpr/milestone27_latency.json")
