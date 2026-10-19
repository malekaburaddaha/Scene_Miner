"""Iterative error corrector for RefProg synthesis.

When the synthesised code fails to execute, this module re-prompts the
LLM with the error traceback and asks it to fix the code. Up to K=5
retries are performed (per the RefAV paper ablation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scene_mining.tracks import TrackCollection

from .executor import ExecutionResult, RefProgExecutor
from .synthesizer import RefProgSynthesizer


@dataclass
class CorrectionResult:
    """Outcome of the iterative error correction process."""
    query: str
    success: bool
    scenario: dict
    final_code: str
    attempts: int
    errors: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "ok" if self.success else "failed"
        return (
            f"CorrectionResult({status}, attempts={self.attempts}, "
            f"n_tracks={len(self.scenario)})"
        )


class IterativeErrorCorrector:
    """Retry RefProg synthesis+execution until success or max_retries.

    Args:
        synthesizer: RefProgSynthesizer instance.
        executor: RefProgExecutor instance.
        max_retries: Maximum number of correction attempts (default 5).
    """

    def __init__(
        self,
        synthesizer: RefProgSynthesizer,
        executor: RefProgExecutor | None = None,
        max_retries: int = 5,
    ) -> None:
        self.synthesizer = synthesizer
        self.executor = executor or RefProgExecutor()
        self.max_retries = max_retries

    def run(
        self,
        query: str,
        tracks: TrackCollection,
        hd_map: Any | None = None,
        feature_cache: Any | None = None,
        log_id: str | None = None,
    ) -> CorrectionResult:
        """Synthesise, execute, and iteratively correct code for a query.

        Args:
            query: Natural language scene description.
            tracks: TrackCollection for the log.
            hd_map: Optional ArgoverseStaticMap.
            feature_cache: Optional FeatureCache.
            log_id: Optional AV2 log ID.

        Returns:
            CorrectionResult with the final scenario and execution history.
        """
        errors: list[str] = []
        last_code = ""
        extra_context: str | None = None

        for attempt in range(1, self.max_retries + 1):
            code = self.synthesizer.synthesize(
                query, log_id=log_id, extra_context=extra_context
            )
            last_code = code
            result: ExecutionResult = self.executor.execute(
                code, tracks, hd_map=hd_map, feature_cache=feature_cache
            )

            if result.success:
                return CorrectionResult(
                    query=query,
                    success=True,
                    scenario=result.scenario,
                    final_code=code,
                    attempts=attempt,
                    errors=errors,
                )

            errors.append(result.error or "Unknown error")
            extra_context = (
                f"Your previous code produced the following error "
                f"(attempt {attempt}/{self.max_retries}):\n"
                f"```\n{result.error}\n```\n"
                f"Previous code:\n```python\n{code}\n```\n"
                f"Please fix the error and rewrite the complete `run` function."
            )

        return CorrectionResult(
            query=query,
            success=False,
            scenario={},
            final_code=last_code,
            attempts=self.max_retries,
            errors=errors,
        )
