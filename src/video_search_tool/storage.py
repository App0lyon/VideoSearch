"""Persistence for generated indexes."""

from __future__ import annotations

import json
from pathlib import Path

from .models import VideoIndex


class IndexStore:
    """Serialize indexes to JSON for later search."""

    def save(self, index: VideoIndex, path: str | Path) -> Path:
        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(json.dumps(index.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return resolved_path

    def load(self, path: str | Path) -> VideoIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return VideoIndex.from_dict(payload)
