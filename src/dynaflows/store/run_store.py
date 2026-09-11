"""Where worker output lives. ADR-008: state carries references, not payloads.

LangGraph checkpoints the entire state at every superstep. Twelve workers each
putting a few thousand tokens of model output into state means re-serialising
all of it, repeatedly, through SQLite's single writer. So the payload goes to a
file and the state carries an `ArtifactRef`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dynaflows.contracts.state import ArtifactRef
from dynaflows.playbook.tokens import estimate_tokens

_PREVIEW_CHARS = 280


class RunStore:
    """Writes artifacts under `.dynaflows/runs/<run_id>/`. Nowhere else.

    The root is fixed at construction rather than passed per call: a path
    argument is an invitation to pass a different one, and ADR-016's boundary
    is only as strong as the narrowest thing that can express a violation.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def write(self, run_id: str, task_id: str, text: str, *, kind: str = "markdown") -> ArtifactRef:
        """Content-addressed, so an identical retry overwrites rather than
        accumulates, and the sha in state is verifiable against the file."""
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        directory = self._root / "runs" / _safe(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe(task_id)}.{sha}.md"
        path.write_text(text, encoding="utf-8")
        return ArtifactRef(
            sha=sha,
            path=str(path),
            kind=kind,
            tokens=estimate_tokens(text),
            preview=text.strip()[:_PREVIEW_CHARS],
        )

    def write_data(self, run_id: str, task_id: str, payload: object) -> ArtifactRef:
        """The machine-readable half, beside the human-readable one.

        The synthesizer needs findings as claims, not as prose it has to parse
        back out of markdown -- and ADR-008 keeps them off the state, because
        twelve workers' findings re-serialised at every superstep is the exact
        pressure the run store exists to remove.
        """
        text = json.dumps(payload, indent=2, sort_keys=True)
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        directory = self._root / "runs" / _safe(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe(task_id)}.{sha}.json"
        path.write_text(text, encoding="utf-8")
        return ArtifactRef(
            sha=sha, path=str(path), kind="json", tokens=estimate_tokens(text), preview=""
        )

    def read_data(self, ref: ArtifactRef) -> object:
        """Never raises on a missing or corrupt file.

        This is read at synthesis time, after every worker has finished and
        been paid for. Losing the whole run because one artifact is unreadable
        would discard work that succeeded.
        """
        try:
            return json.loads(Path(ref.path).read_text(encoding="utf-8"))
        except OSError, ValueError:
            return None

    def read(self, ref: ArtifactRef) -> str:
        """The synthesizer's side of ADR-008: refs go through state, the text
        is fetched once at the end."""
        return Path(ref.path).read_text(encoding="utf-8")


def _safe(value: str) -> str:
    """A task id comes from a model. It is normalised in planning, but this is
    a filesystem path and a second check here costs nothing."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in value).strip("-")
    return cleaned or "unnamed"


def get_run_store(root: Path | None = None) -> RunStore:
    """Sole construction path (playbook §3.1). Tests patch THIS."""
    if root is None:
        from dynaflows.settings import get_settings

        root = get_settings().home
    return RunStore(root)
