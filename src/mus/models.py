from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any, Iterable

@dataclass(slots=True)
class Constraints:
    max_melodic_step: int = 2
    forbidden_parallels: list[str] = field(default_factory=lambda: ["P5", "P8"])
    max_k: int | None = 3
    max_k1: int | None = None
    max_k2: int | None = None
    max_k3: int | None = None

@dataclass(slots=True)
class ProblemInstance:
    problem_id: str
    voices: int
    chord_roots: list[str]
    chord_durations: list[float]
    constraints: Constraints
    starting_pitches: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Keep JSON compact by omitting null-valued optional constraints.
        payload["constraints"] = {
            key: value
            for key, value in payload["constraints"].items()
            if value is not None
        }
        if payload.get("starting_pitches") is None:
            payload.pop("starting_pitches")
        if not payload.get("metadata"):
            payload.pop("metadata")
        return payload

def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")))
            handle.write("\n")
