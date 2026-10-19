"""Query router: decides the optimal execution path for a NL query.

Execution paths:
- REFPROG_ONLY:        Pure geometric/motion/map reasoning (no visual needed)
- VLM_ONLY:            Pure visual attribute reasoning (no geometry needed)
- HYBRID_SEQUENTIAL:   RefProg geometric filter → visual check on candidates
- HYBRID_PARALLEL:     Independent geometric + visual branches → merge

The router uses a two-stage approach:
1. Fast keyword heuristic (no LLM call)
2. LLM-based disambiguation for ambiguous queries (optional)
"""

from __future__ import annotations

from enum import Enum, auto


class ExecutionPath(Enum):
    REFPROG_ONLY = auto()
    VLM_ONLY = auto()
    HYBRID_SEQUENTIAL = auto()
    HYBRID_PARALLEL = auto()


# ── Keyword lists ─────────────────────────────────────────────────────────────

# Visual keywords → VLM needed
_VISUAL_KEYWORDS = {
    # Weather
    "fog", "foggy", "rain", "rainy", "raining", "snow", "snowy", "snowing",
    "wet", "icy", "overcast", "cloudy", "sunny", "clear",
    # Time of day
    "night", "nighttime", "dark", "dawn", "dusk", "sunset", "sunrise",
    "daytime", "morning", "evening",
    # Colors
    "red", "blue", "green", "yellow", "white", "black", "silver", "gray",
    "grey", "orange", "brown", "purple",
    # Visual attributes
    "color", "colour", "wearing", "helmet", "vest", "jacket", "uniform",
    "damaged", "dirty", "dirty", "clean", "occluded", "hidden", "blurry",
    "illuminated", "lit", "glowing", "bright",
}

# Geometric/motion keywords → RefProg
_GEOMETRIC_KEYWORDS = {
    "following", "behind", "in front", "ahead", "next to",
    "crossing", "turning", "accelerating", "decelerating", "braking",
    "stopping", "stopped", "stationary", "parked", "moving",
    "intersection", "crosswalk", "stop sign", "lane", "drivable",
    "approaching", "near", "close", "distance", "speed",
    "heading", "facing", "oncoming", "overtaking", "merging",
    "swerving", "changing lane", "yielding",
}


class QueryRouter:
    """Route NL queries to the appropriate execution path.

    Args:
        use_llm_fallback: If True, calls an LLM for ambiguous queries.
        synthesizer: Optional RefProgSynthesizer for LLM routing.
    """

    def __init__(
        self,
        use_llm_fallback: bool = False,
        synthesizer: "RefProgSynthesizer | None" = None,  # noqa: F821
    ) -> None:
        self.use_llm_fallback = use_llm_fallback
        self.synthesizer = synthesizer

    def route(self, query: str) -> ExecutionPath:
        """Determine the execution path for a NL query.

        Args:
            query: Natural language scene description.

        Returns:
            ExecutionPath enum value.
        """
        q_lower = query.lower()
        has_visual = any(kw in q_lower for kw in _VISUAL_KEYWORDS)
        has_geometric = any(kw in q_lower for kw in _GEOMETRIC_KEYWORDS)

        if has_visual and not has_geometric:
            return ExecutionPath.VLM_ONLY
        if has_geometric and not has_visual:
            return ExecutionPath.REFPROG_ONLY
        if has_visual and has_geometric:
            return ExecutionPath.HYBRID_SEQUENTIAL
        # Ambiguous: default to HYBRID_SEQUENTIAL (safest)
        if self.use_llm_fallback and self.synthesizer is not None:
            return self._llm_route(query)
        return ExecutionPath.HYBRID_SEQUENTIAL

    def _llm_route(self, query: str) -> ExecutionPath:
        """Use LLM to disambiguate routing for edge cases."""
        prompt = (
            "Classify this autonomous driving scene query into one of: "
            "REFPROG_ONLY, VLM_ONLY, HYBRID_SEQUENTIAL, HYBRID_PARALLEL.\n\n"
            "Rules:\n"
            "- REFPROG_ONLY: only geometry, motion, or map reasoning needed\n"
            "- VLM_ONLY: only visual/appearance attributes needed\n"
            "- HYBRID_SEQUENTIAL: both, but geometry filter first\n"
            "- HYBRID_PARALLEL: independent geometry + visual branches\n\n"
            f"Query: {query}\n\n"
            "Answer with EXACTLY one of: REFPROG_ONLY, VLM_ONLY, "
            "HYBRID_SEQUENTIAL, HYBRID_PARALLEL"
        )
        try:
            response = self.synthesizer._call_llm("", prompt).strip().upper()
            return ExecutionPath[response]
        except (KeyError, Exception):
            return ExecutionPath.HYBRID_SEQUENTIAL

    def classify_keywords(self, query: str) -> dict[str, bool]:
        """Return a diagnostic dict of keyword matches."""
        q_lower = query.lower()
        return {
            "has_visual": any(kw in q_lower for kw in _VISUAL_KEYWORDS),
            "has_geometric": any(kw in q_lower for kw in _GEOMETRIC_KEYWORDS),
            "visual_keywords": [kw for kw in _VISUAL_KEYWORDS if kw in q_lower],
            "geometric_keywords": [kw for kw in _GEOMETRIC_KEYWORDS if kw in q_lower],
        }
