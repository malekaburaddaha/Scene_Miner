"""Captioning evaluation metrics: CIDEr, METEOR, ROUGE-L.

Uses ``nltk`` (for METEOR) and ``rouge_score`` (for ROUGE-L), both of
which are already in the project requirements.  CIDEr is computed via
``pycocoevalcap`` when available, with a pure-Python TF-IDF fallback.

Usage::

    from scene_mining.evaluation.captioning_metrics import (
        compute_cider, compute_meteor, compute_rouge_l,
    )

    hyps = ["a vehicle following a cyclist", "a pedestrian crossing"]
    refs = [["a car following a bike rider"], ["a person walking across the road"]]

    print(compute_cider(hyps, refs))
    print(compute_meteor(hyps, refs))
    print(compute_rouge_l(hyps, refs))
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def compute_cider(
    hypotheses: list[str],
    references: list[list[str]],
    n: int = 4,
) -> float:
    """Compute CIDEr-D score (Consensus-based Image Description Evaluation).

    Tries to use ``pycocoevalcap``; falls back to a pure-Python TF-IDF
    implementation if not installed.

    Args:
        hypotheses: List of generated captions.
        references: Per-caption list of reference strings.
        n: Maximum n-gram order (default 4).

    Returns:
        CIDEr-D score (typically 0–10, higher is better).
    """
    try:
        return _cider_pycocoevalcap(hypotheses, references)
    except Exception:
        return _cider_tfidf(hypotheses, references, n=n)


def compute_meteor(
    hypotheses: list[str],
    references: list[list[str]],
) -> float:
    """Compute corpus-level METEOR score.

    Args:
        hypotheses: List of generated captions.
        references: Per-caption list of reference strings.

    Returns:
        METEOR score in [0, 1].
    """
    import nltk

    # Ensure WordNet / tokenise resources are available (no-op if already there)
    for resource in ("wordnet", "punkt", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

    scores = []
    for hyp, refs in zip(hypotheses, references):
        hyp_tokens = hyp.lower().split()
        ref_tokens_list = [r.lower().split() for r in refs]
        # Use the best reference for each hypothesis
        best = max(
            nltk.translate.meteor_score.single_meteor_score(ref, hyp_tokens)
            for ref in ref_tokens_list
        )
        scores.append(best)

    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_rouge_l(
    hypotheses: list[str],
    references: list[list[str]],
) -> float:
    """Compute corpus-level ROUGE-L F1 score.

    Args:
        hypotheses: List of generated captions.
        references: Per-caption list of reference strings.

    Returns:
        ROUGE-L F1 in [0, 1].
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for hyp, refs in zip(hypotheses, references):
        # Use best ROUGE-L across references
        best_f1 = max(
            scorer.score(ref, hyp)["rougeL"].fmeasure for ref in refs
        )
        scores.append(best_f1)

    return float(sum(scores) / len(scores)) if scores else 0.0


# ── CIDEr helpers ─────────────────────────────────────────────────────────────


def _cider_pycocoevalcap(hypotheses: list[str], references: list[list[str]]) -> float:
    """CIDEr via pycocoevalcap (if installed)."""
    from pycocoevalcap.cider.cider import Cider  # type: ignore[import]

    gts = {i: [{"caption": r} for r in refs] for i, refs in enumerate(references)}
    res = {i: [{"caption": h}] for i, h in enumerate(hypotheses)}
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return float(score)


def _cider_tfidf(
    hypotheses: list[str],
    references: list[list[str]],
    n: int = 4,
) -> float:
    """Pure-Python CIDEr-D fallback (no external dependencies).

    Implements the TF-IDF weighted n-gram cosine similarity described in
    Vedantam et al. (2015).
    """
    N = len(hypotheses)
    if N == 0:
        return 0.0

    def ngrams(tokens: list[str], order: int) -> Counter:
        return Counter(
            tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1)
        )

    # Collect all reference n-grams for IDF computation
    df: Counter = Counter()
    for refs in references:
        seen: set = set()
        for ref in refs:
            for order in range(1, n + 1):
                for gram in ngrams(ref.lower().split(), order):
                    if gram not in seen:
                        df[gram] += 1
                        seen.add(gram)

    def tfidf_vec(tokens: list[str], refs_tokens: list[list[str]]) -> dict:
        vec: dict = {}
        for order in range(1, n + 1):
            hyp_counts = ngrams(tokens, order)
            ref_counts: Counter = Counter()
            for ref in refs_tokens:
                ref_counts |= ngrams(ref, order)
            for gram, cnt in hyp_counts.items():
                tf = cnt / max(len(hyp_counts), 1)
                idf = math.log((N + 1.0) / (df[gram] + 1.0))
                ref_tf = ref_counts[gram] / max(len(ref_counts), 1)
                vec[gram] = vec.get(gram, 0.0) + tf * idf * min(1.0, cnt / max(ref_tf, 1e-6))
        return vec

    def cosine(v1: dict, v2: dict) -> float:
        common = set(v1) & set(v2)
        dot = sum(v1[k] * v2[k] for k in common)
        n1 = math.sqrt(sum(x ** 2 for x in v1.values()))
        n2 = math.sqrt(sum(x ** 2 for x in v2.values()))
        return dot / (n1 * n2 + 1e-9)

    scores = []
    for hyp, refs in zip(hypotheses, references):
        hyp_tok = hyp.lower().split()
        refs_tok = [r.lower().split() for r in refs]
        hyp_vec = tfidf_vec(hyp_tok, refs_tok)
        best = max(
            cosine(hyp_vec, tfidf_vec(r, refs_tok)) for r in refs_tok
        ) if refs_tok else 0.0
        scores.append(best)

    return float(sum(scores) / len(scores)) * 10.0  # CIDEr scale [0,10]
