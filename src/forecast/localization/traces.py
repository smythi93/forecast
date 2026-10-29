"""
Spectrum-based fault localization over FORECAST's traced executions.

FORECAST traces each failing and passing test once, for its diagnosis. The
same traces record every function entry, so a method-level spectrum -- and
hence a classic SBFL ranking -- comes for free, from exactly the executions the
diagnosis is built from. Agents that use test information (AutoCodeRover's
analysis-augmented retrieval) receive this ranking, in both modes: comparing
it with the cause chain isolates what the chain adds beyond the spectrum.
"""

from __future__ import annotations

from typing import Sequence


def rank_functions(
    event_files: Sequence,
    *,
    formula: str = "Ochiai",
    top: int = 50,
    processes: int = 1,
) -> list[dict]:
    """
    Rank the functions of a traced program by suspiciousness.

    :param event_files: sflkit ``EventFile`` objects, failing and passing
        (their ``failing`` flag tells them apart).
    :param formula: Name of an sflkit spectrum metric (``Ochiai``,
        ``Tarantula``, ...).
    :param top: How many functions to return.
    :param processes: Worker processes to spread the runs over; decoding the
        traces dominates, and large instances hold gigabytes of them.
    :returns: Dicts with ``file``, ``method``, ``line``, ``suspiciousness``,
        most suspicious first; ties keep a stable order by location.
    """
    from sflkit.analysis.analysis_type import AnalysisType
    from sflkit.analysis.analyzer import Analyzer
    from sflkit.analysis.factory import FunctionFactory
    from sflkit.analysis.spectra import Spectrum

    failing = [f for f in event_files if getattr(f, "failing", False)]
    passing = [f for f in event_files if not getattr(f, "failing", False)]
    if not failing:
        return []
    analyzer = Analyzer(failing, passing, FunctionFactory(), processes=max(1, processes))
    analyzer.analyze()
    metric = getattr(Spectrum, formula)
    ranked = []
    for function in analyzer.get_analysis_by_type(AnalysisType.FUNCTION):
        ranked.append(
            {
                "file": str(function.file),
                "method": str(function.function),
                "line": int(function.line),
                "suspiciousness": float(function.get_metric(metric)),
            }
        )
    ranked.sort(key=lambda f: (-f["suspiciousness"], f["file"], f["line"]))
    return ranked[:top]
