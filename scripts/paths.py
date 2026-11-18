"""Machine-specific locations, each overridable by an environment variable.

SCENEMINER_DATA_ROOT  directory holding external_adverse/{bdd100k,bdd100k_holdout} (default: data)
QWEN_VL               Qwen2.5-VL-3B-Instruct checkpoint, hub id or local path
BLIP2                 BLIP-2 OPT-2.7B checkpoint, hub id or local path
SIGLIP2_SO400M        SigLIP2 So400m/14 at 384 px (Milestone 34 backbone study)
SM_DEVICE             torch device for every script (default: cuda if available, else cpu)
"""
import os

DATA_ROOT = os.environ.get("SCENEMINER_DATA_ROOT", "data")
EXT = os.path.join(DATA_ROOT, "external_adverse")
QWEN_VL = os.environ.get("QWEN_VL", "Qwen/Qwen2.5-VL-3B-Instruct")
BLIP2 = os.environ.get("BLIP2", "Salesforce/blip2-opt-2.7b")
SIGLIP2_SO400M = os.environ.get("SIGLIP2_SO400M", "google/siglip2-so400m-patch14-384")
HUMAN_R1 = "benchmark_cvpr/human_rater1_600.json"      # rater 1, 600 scenes (anonymised)
HUMAN_R2 = "benchmark_cvpr/human_rater2_160.json"      # rater 2, 160 scenes (anonymised)


def _default_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = os.environ.get("SM_DEVICE") or _default_device()


def sync():
    """torch.cuda.synchronize() on GPU, a no-op on CPU (for timing)."""
    if DEVICE.startswith("cuda"):
        import torch
        torch.cuda.synchronize()
