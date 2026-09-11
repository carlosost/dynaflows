"""The only package permitted to touch the filesystem outside the databases.

ADR-016 forbids a worker from writing anywhere but `.dynaflows/`. ADR-017
confines reading the plan's `inputs` to one place rather than N parallel
branches. Both are enforced by `scripts/lint_architecture.py`, which rejects
filesystem calls in any module outside this package.
"""

from dynaflows.store.run_store import RunStore, get_run_store
from dynaflows.store.sources import SourceRefusal, read_sources

__all__ = ["RunStore", "SourceRefusal", "get_run_store", "read_sources"]
