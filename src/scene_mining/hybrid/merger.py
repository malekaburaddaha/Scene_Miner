"""ScenarioDictMerger: combine outputs from multiple execution paths."""

from __future__ import annotations

from scene_mining.atomic_functions.base import ScenarioDict


class ScenarioDictMerger:
    """Merge ScenarioDicts from parallel execution paths."""

    @staticmethod
    def intersect(a: ScenarioDict, b: ScenarioDict) -> ScenarioDict:
        """Keep tracks present in BOTH a and b (AND semantics)."""
        result: ScenarioDict = {}
        for tid in set(a) & set(b):
            # Merge related dicts — keep only related objects present in both
            a_rel, b_rel = a[tid], b[tid]
            if not a_rel and not b_rel:
                result[tid] = {}
            elif not a_rel:
                result[tid] = b_rel
            elif not b_rel:
                result[tid] = a_rel
            else:
                # Intersect related object IDs; intersect timestamps
                common: dict[str, list[int]] = {}
                for rid in set(a_rel) & set(b_rel):
                    ts = sorted(set(a_rel[rid]) & set(b_rel[rid]))
                    if ts:
                        common[rid] = ts
                if common:
                    result[tid] = common
        return result

    @staticmethod
    def union(a: ScenarioDict, b: ScenarioDict) -> ScenarioDict:
        """Keep tracks from EITHER a or b (OR semantics)."""
        result: ScenarioDict = dict(a)
        for tid, rel in b.items():
            if tid not in result:
                result[tid] = rel
            else:
                for rid, ts_list in rel.items():
                    if rid in result[tid]:
                        result[tid][rid] = sorted(set(result[tid][rid]) | set(ts_list))
                    else:
                        result[tid][rid] = list(ts_list)
        return result

    @staticmethod
    def filter_timestamps(
        scenario: ScenarioDict,
        allowed_timestamps: set[int],
    ) -> ScenarioDict:
        """Remove timestamps not in allowed_timestamps from all entries."""
        result: ScenarioDict = {}
        for tid, related in scenario.items():
            clean: dict[str, list[int]] = {}
            for rid, ts_list in related.items():
                filtered = [t for t in ts_list if t in allowed_timestamps]
                if filtered:
                    clean[rid] = filtered
            if clean or not related:
                result[tid] = clean
        return result
