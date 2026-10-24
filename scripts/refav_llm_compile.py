#!/usr/bin/env python3
"""MILESTONE 28 - RefAV-style baseline with LLM-compiled programs.

RefAV has an LLM translate a scenario description into an executable program over
perception outputs. Here the local Qwen2.5-VL-3B, in text-only mode with greedy
decoding and a fixed 5-shot prompt, compiles every benchmark query into a JSON
conjunction of predicates over the attributes our corpora expose:
  {"flag": droplets|glare|wet_road|low_contrast, "value": bool}      (VLM decisions)
  {"meta": lighting|weather|density, "value": night|day|rain|dry|dense|sparse}
  {"tag": <one of 20 scene/object tags>, "value": bool}               (corpus only)
The five examples are invented phrasings that are not benchmark queries (checked
below). Output: results/cvpr/refav_llm_programs.json (raw text, parsed program,
validity), reused by eval_related_work_baselines.py. No external API is called.
"""
import json, re, sys
from pathlib import Path
import torch
sys.path.insert(0, "scripts")

TAGS = ["dark", "rainy", "construction zone", "intersection", "parking lot", "highway", "school zone",
        "residential", "pedestrians", "bicycles", "motorcycles", "trucks", "buses", "cars", "emergency vehicles",
        "animals", "jaywalking", "running pedestrians", "erratic driving", "stopped traffic"]
FLAGS = ["droplets", "glare", "wet_road", "low_contrast"]
META = {"lighting": ["night", "day"], "weather": ["rain", "dry"], "density": ["dense", "sparse"]}
SHOTS = [
    ("a dim street with headlights glaring", [{"flag": "glare", "value": True}, {"meta": "lighting", "value": "night"}]),
    ("a truck parked in heavy traffic", [{"tag": "trucks", "value": True}, {"meta": "density", "value": "dense"}]),
    ("a dry sunny road and a clean lens", [{"meta": "weather", "value": "dry"}, {"meta": "lighting", "value": "day"},
                                          {"flag": "droplets", "value": False}]),
    ("kids walking near a school", [{"tag": "school zone", "value": True}, {"tag": "pedestrians", "value": True}]),
    ("a puddle-covered intersection", [{"flag": "wet_road", "value": True}, {"tag": "intersection", "value": True}]),
]
SYSTEM = ("You translate a driving-scene search query into a JSON list of predicates that a matching scene must "
          "satisfy. Allowed predicates:\n"
          f'  {{"flag": one of {FLAGS}, "value": true or false}}  - camera/road degradation seen by a vision model\n'
          '  {"meta": "lighting" | "weather" | "density", "value": "night"|"day" | "rain"|"dry" | "dense"|"sparse"}\n'
          f'  {{"tag": one of {TAGS}, "value": true or false}}\n'
          "Use only these. Use as few predicates as needed. Answer with the JSON list only.")


def all_queries():
    from eval_human_gt_full import QUERY
    from eval_external_bdd100k import EVAL_Q
    qs = list(QUERY.values()) + [q for q, _ in EVAL_Q]
    qs += [q["text"] for q in json.load(open("benchmark_cvpr/queries_semantic_edge_cases.json"))["queries"]]
    qs += [q["text"] for q in json.load(open("benchmark_cvpr/queries_390.json"))["eval"]]
    qs += [q["text"] for q in json.load(open("benchmark_cvpr/bdd_holdout_queries_36.json"))["queries"]]
    return list(dict.fromkeys(qs))


def validate(prog):
    ok = []
    for p in prog if isinstance(prog, list) else []:
        if not isinstance(p, dict): continue
        if "flag" in p and p["flag"] in FLAGS and isinstance(p.get("value"), bool): ok.append(p)
        elif "meta" in p and p["meta"] in META and p.get("value") in META[p["meta"]]: ok.append(p)
        elif "tag" in p and p["tag"] in TAGS and isinstance(p.get("value"), bool): ok.append(p)
    return ok


def main():
    qs = all_queries()
    shots = [s for s, _ in SHOTS]
    assert not set(shots) & set(qs), "a few-shot example is a benchmark query"
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    P = "/nfs/turbo/umd-srawa/hf_cache/Qwen2.5-VL-3B-Instruct"
    proc = AutoProcessor.from_pretrained(P)
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(P, torch_dtype=torch.bfloat16).to("cuda").eval()
    msgs0 = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}]
    for q, prog in SHOTS:
        msgs0 += [{"role": "user", "content": [{"type": "text", "text": q}]},
                  {"role": "assistant", "content": [{"type": "text", "text": json.dumps(prog)}]}]
    out = {"doc": __doc__, "system_prompt": SYSTEM, "shots": SHOTS, "programs": {}}
    for i, q in enumerate(qs):
        msgs = msgs0 + [{"role": "user", "content": [{"type": "text", "text": q}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[t], return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = m.generate(**inp, max_new_tokens=120, do_sample=False)
        raw = proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        mt = re.search(r"\[.*\]", raw, re.S)
        try: parsed = json.loads(mt.group(0)) if mt else None
        except Exception: parsed = None
        prog = validate(parsed)
        out["programs"][q] = {"raw": raw, "program": prog, "parsed": parsed is not None,
                              "n_dropped": (len(parsed) - len(prog)) if isinstance(parsed, list) else None}
        if (i + 1) % 25 == 0: print(f"[refav-llm] {i+1}/{len(qs)}", flush=True)
    n_empty = sum(not v["program"] for v in out["programs"].values())
    out["summary"] = {"n_queries": len(qs), "n_unparsable": sum(not v["parsed"] for v in out["programs"].values()),
                      "n_empty_program": n_empty}
    Path("results/cvpr/refav_llm_programs.json").write_text(json.dumps(out, indent=1))
    print(out["summary"])


if __name__ == "__main__":
    main()
