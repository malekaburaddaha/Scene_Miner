#!/usr/bin/env python3
"""Overnight job orchestrator for Exclusive_Process GPUs.

The cluster runs the A40s in Exclusive_Process mode: exactly ONE CUDA context
per GPU. Launching more than 4 concurrent jobs makes every extra one die with
'CUDA-capable device(s) is/are busy or unavailable'. This runner keeps exactly
one job per GPU, refills a slot the moment it frees, and retries a job once.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

PY = os.environ.get("PY", sys.executable)
SPLIT = "benchmark_cvpr/split_906_602.json"
LOGS = Path("results/overnight_oct18/logs"); LOGS.mkdir(parents=True, exist_ok=True)
COMMON = ["--steps", "3000", "--img-side", "336", "--ckpt-every-steps", "3000",
          "--log-every", "1500", "--warmup-steps", "200", "--split-file", SPLIT]

ROW = {
 "A":  ["--supervision","caption","--legacy-scoring","--legacy-scalar","--freeze-visual","--unconditional-negs","--w-vicreg","0","--tau","1.0"],
 "A1": ["--supervision","caption","--legacy-scoring","--legacy-scalar","--freeze-visual","--w-vicreg","0","--tau","1.0"],
 "A2": ["--supervision","deterministic","--legacy-scoring","--legacy-scalar","--freeze-visual","--unconditional-negs","--w-vicreg","0","--tau","1.0"],
 "B":  ["--supervision","deterministic","--legacy-scoring","--freeze-visual","--w-vicreg","0","--tau","1.0"],
 "E":  ["--supervision","deterministic","--freeze-visual","--w-vicreg","1.0","--w-var","1.0","--w-cov","0.5","--tau","0.4","--tau-final","0.30"],
}
LOAO_BASE = ["--supervision","deterministic","--freeze-visual","--w-vicreg","1.0",
             "--w-var","1.0","--w-cov","0.5","--tau","0.4","--tau-final","0.30"]

def build_jobs():
    jobs = []
    for r in ["A","A1","A2","B","E"]:
        for s in range(5):
            wd = f"work_dirs/cvpr_ablation/{r}_s{s}"
            if Path(wd, "final.pt").exists(): continue
            jobs.append({"name": f"abl_{r}_s{s}", "wd": wd,
                         "args": COMMON + ROW[r] + ["--seed", str(s)]})
    for a in ["droplets","glare","wet_road","low_contrast"]:
        for s in range(5):
            wd = f"work_dirs/cvpr_loao/{a}_s{s}"
            if Path(wd, "final.pt").exists(): continue
            jobs.append({"name": f"loao_{a}_s{s}", "wd": wd,
                         "args": COMMON + LOAO_BASE + ["--drop-attr", a, "--seed", str(s)]})
    return jobs

def launch(job, gpu):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = "src:" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log = open(LOGS / f"{job['name']}.log", "w")
    cmd = [PY, "scripts/train_filter_cvpr.py", "--work-dir", job["wd"]] + job["args"]
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"  [launch] {job['name']:<24} GPU{gpu}  pid {p.pid}", flush=True)
    return {"proc": p, "job": job, "gpu": gpu, "log": log, "t0": time.time()}

def main():
    jobs = build_jobs()
    print(f"[orchestrator] {len(jobs)} jobs queued, 4 GPU slots "
          f"(Exclusive_Process: 1 job/GPU)", flush=True)
    slots = {g: None for g in range(4)}
    queue, retries, done, failed = list(jobs), {}, [], []
    while queue or any(slots.values()):
        for g, run in list(slots.items()):
            if run is None:
                if queue:
                    slots[g] = launch(queue.pop(0), g)
                continue
            rc = run["proc"].poll()
            if rc is None: continue
            run["log"].close()
            el = (time.time() - run["t0"]) / 60
            nm = run["job"]["name"]
            ok = Path(run["job"]["wd"], "final.pt").exists()
            if ok:
                done.append(nm); print(f"  [done]   {nm:<24} {el:.1f} min", flush=True)
            else:
                n = retries.get(nm, 0)
                if n < 1:
                    retries[nm] = n + 1; queue.append(run["job"])
                    print(f"  [retry]  {nm:<24} rc={rc} ({el:.1f} min)", flush=True)
                else:
                    failed.append(nm); print(f"  [FAIL]   {nm:<24} rc={rc}", flush=True)
            slots[g] = None
        time.sleep(10)
    print(f"\n[orchestrator] done={len(done)} failed={len(failed)}", flush=True)
    if failed: print("  failed:", failed, flush=True)
    Path("results/overnight_oct18/orchestrator_status.json").write_text(
        json.dumps({"done": done, "failed": failed, "n_jobs": len(jobs)}, indent=1))
    print("ORCHESTRATOR_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
