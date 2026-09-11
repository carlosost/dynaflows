"""Known-answer calibration for the worker. ADR-022.

"No findings" is unfalsifiable on its own: a clean codebase and a worker that
cannot find anything produce identical output. This package plants defects in
a fixture, records where they are, and scores what comes back.

It measures the WORKER, not the truth. Five planted defects say nothing about
the defects nobody planted, and full recall here must never be read as "the
pipeline finds bugs". It is a floor.
"""

from dynaflows.calibration.harness import (
    Defect,
    Scorecard,
    fixture_paths,
    load_manifest,
    score,
)

__all__ = ["Defect", "Scorecard", "fixture_paths", "load_manifest", "score"]
