"""HybridQueryExecutor: top-level orchestrator for dual-path scene mining.

Routes each NL query through the optimal path (RefProg / VLM / Hybrid)
and returns a normalised ScenarioDict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scene_mining.atomic_functions.base import ScenarioDict
from scene_mining.refprog import (
    ExecutionPath,
    IterativeErrorCorrector,
    QueryRouter,
    RefProgExecutor,
    RefProgSynthesizer,
)
from scene_mining.tracks import TrackCollection


@dataclass
class QueryResult:
    """Result from a single HybridQueryExecutor.run() call."""
    query: str
    log_id: str
    execution_path: ExecutionPath
    scenario: ScenarioDict
    success: bool
    error: str | None = None
    refprog_code: str | None = None
    n_referred_tracks: int = 0

    def __post_init__(self) -> None:
        self.n_referred_tracks = len(self.scenario)


class HybridQueryExecutor:
    """Execute scene mining queries using the optimal dual-path strategy.

    Args:
        synthesizer: RefProgSynthesizer for LLM code generation.
        router: QueryRouter for path selection.
        max_retries: Max error correction attempts (passed to corrector).
    """

    def __init__(
        self,
        synthesizer: RefProgSynthesizer | None = None,
        router: QueryRouter | None = None,
        max_retries: int = 5,
    ) -> None:
        self.synthesizer = synthesizer or RefProgSynthesizer()
        self.router = router or QueryRouter()
        self.executor = RefProgExecutor()
        self.corrector = IterativeErrorCorrector(
            self.synthesizer, self.executor, max_retries=max_retries
        )

    def run(
        self,
        query: str,
        tracks: TrackCollection,
        hd_map: Any | None = None,
        feature_cache: Any | None = None,
        log_id: str | None = None,
    ) -> QueryResult:
        """Execute a single query on a driving log.

        Args:
            query: Natural language scene description.
            tracks: TrackCollection for the log.
            hd_map: Optional ArgoverseStaticMap.
            feature_cache: Optional FeatureCache (enables visual functions).
            log_id: Optional log ID for context in LLM prompt.

        Returns:
            QueryResult with ScenarioDict and execution metadata.
        """
        log_id = log_id or tracks.log_id
        path = self.router.route(query)

        if path in (ExecutionPath.REFPROG_ONLY, ExecutionPath.HYBRID_SEQUENTIAL,
                    ExecutionPath.HYBRID_PARALLEL):
            result = self.corrector.run(
                query, tracks, hd_map=hd_map,
                feature_cache=feature_cache if path != ExecutionPath.REFPROG_ONLY else None,
                log_id=log_id,
            )
            return QueryResult(
                query=query,
                log_id=log_id,
                execution_path=path,
                scenario=result.scenario,
                success=result.success,
                error=result.errors[-1] if result.errors else None,
                refprog_code=result.final_code,
            )

        # VLM_ONLY: no geometric code synthesis; return empty (future work)
        return QueryResult(
            query=query,
            log_id=log_id,
            execution_path=path,
            scenario={},
            success=False,
            error="VLM_ONLY path not yet implemented without RefProg fallback.",
        )

    def run_batch(
        self,
        queries: list[str],
        tracks: TrackCollection,
        hd_map: Any | None = None,
        feature_cache: Any | None = None,
        log_id: str | None = None,
    ) -> list[QueryResult]:
        """Run multiple queries on the same log sequentially."""
        return [
            self.run(q, tracks, hd_map=hd_map, feature_cache=feature_cache, log_id=log_id)
            for q in queries
        ]
